"""Production-operation tests for durable request acceptance and executor leases.

Every test drives real temporary Corpora through the production ``CorpusGraph``
interface and its private coordination implementation. Cross-process behavior is
exercised with real subprocesses that are really terminated, because a durable
queue and a recoverable lease only mean anything against a process that dies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _child_env() -> dict[str, str]:
    """Return an environment that lets a child subprocess import graphify."""
    repository = Path(__file__).parents[1]
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repository) + (os.pathsep + prior if prior else "")
    return env


def _start_child(script: Path, body: str, *args: str) -> subprocess.Popen[str]:
    """Write ``body`` to ``script`` and start it as a real child process."""
    script.write_text(body.lstrip(), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=str(script.parent),
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _run_child(script: Path, body: str, *args: str) -> str:
    """Run ``body`` to completion in a real child process and return its stdout."""
    process = _start_child(script, body, *args)
    stdout, stderr = process.communicate(timeout=300)
    assert process.returncode == 0, f"child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    return stdout


def _wait_for(predicate, *, timeout: float = 60.0, description: str = "condition"):
    """Poll ``predicate`` until it returns a truthy value, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {description}")


def _coordinator(root: Path, output: Path):
    """Return the production coordinator bound to one temporary Corpus."""
    from graphify.generation._coordination import _RequestCoordinator
    from graphify.generation._types import Corpus

    return _RequestCoordinator(Corpus(root=root, output=output))


def _graph_labels(output: Path) -> set[str]:
    """Return the node labels of the active materialized graph."""
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    return {node["label"] for node in graph["nodes"]}


# --- durable acceptance -------------------------------------------------------


def test_queued_is_returned_only_after_the_request_is_durable(tmp_path) -> None:
    """Persist a background request before acknowledging it as ``Queued``."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        Queued,
        ReturnWhenQueued,
    )

    (tmp_path / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest((Path("service.py"),)),
        completion=ReturnWhenQueued(),
    )

    assert isinstance(outcome, Queued)
    assert outcome.request_id
    # Durable means "already on disk when the caller was told Queued", so the
    # record is read back from the filesystem rather than from the coordinator's
    # own in-memory state.
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / ".graphify_requests").iterdir())
    ]
    assert [record["request_id"] for record in records] == [outcome.request_id]
    assert records[0]["operation"] == "code-update"
    assert records[0]["changed_paths"] == ["service.py"]
    # Acceptance is not execution: nothing was published by the acknowledgment.
    assert not (output / "graph.json").exists()


def test_an_exited_submitter_leaves_its_accepted_request_durable(tmp_path) -> None:
    """Survive the submitting process exiting right after acceptance."""
    output = tmp_path / "graphify-out"
    (tmp_path / "submitted.py").write_text("def submitted():\n    return 1\n", encoding="utf-8")

    stdout = _run_child(
        tmp_path / "submit.py",
        """
import sys
from pathlib import Path

from graphify.generation import (
    CodeUpdateRequest,
    Corpus,
    CorpusGraph,
    Queued,
    ReturnWhenQueued,
)

root, output = Path(sys.argv[1]), Path(sys.argv[2])
outcome = CorpusGraph(Corpus(root=root, output=output)).code_update(
    CodeUpdateRequest((Path("submitted.py"),)),
    completion=ReturnWhenQueued(),
)
assert isinstance(outcome, Queued), outcome
print(outcome.request_id, flush=True)
""",
        str(tmp_path),
        str(output),
    )
    request_id = stdout.strip()

    pending = _coordinator(tmp_path, output).pending()
    assert [request.request_id for request in pending] == [request_id]
    assert pending[0].operation == "code-update"


def test_a_later_executor_covers_work_accepted_by_an_exited_submitter(tmp_path) -> None:
    """Execute queued work in a process that never submitted it."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    output = tmp_path / "graphify-out"
    (tmp_path / "submitted.py").write_text(
        "class Submitted:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    _run_child(
        tmp_path / "submit.py",
        """
import sys
from pathlib import Path

from graphify.generation import CodeUpdateRequest, Corpus, CorpusGraph, ReturnWhenQueued

root, output = Path(sys.argv[1]), Path(sys.argv[2])
CorpusGraph(Corpus(root=root, output=output)).code_update(
    CodeUpdateRequest((Path("submitted.py"),)),
    completion=ReturnWhenQueued(),
)
""",
        str(tmp_path),
        str(output),
    )
    assert _coordinator(tmp_path, output).pending()

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert "Submitted" in _graph_labels(output)
    # The queued request is covered by the generation that included its source,
    # so it is retired rather than replayed forever.
    assert _coordinator(tmp_path, output).pending() == ()


def test_an_operation_without_an_executor_cannot_be_durably_queued(tmp_path) -> None:
    """Refuse to acknowledge work no executor could ever carry out.

    A prepared candidate lives in the calling process's memory, and Full
    extraction and Reclustering are still adapter-prepared rather than executed
    by this module. Recording either durably would produce a queued
    acknowledgment nothing on the Corpus is able to cover.
    """
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        ReclusteringRequest,
        ReturnWhenQueued,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))

    with pytest.raises(ValueError):
        owner.full_extraction(
            FullExtractionRequest(),
            completion=ReturnWhenQueued(),
            _publication=_Publication(build_config={"generation": "prepared"}),
        )
    with pytest.raises(ValueError):
        owner.full_extraction(
            FullExtractionRequest(),
            completion=ReturnWhenQueued(),
        )
    with pytest.raises(ValueError):
        owner.reclustering(
            ReclusteringRequest(),
            completion=ReturnWhenQueued(),
        )

    # Refused before anything was written, so no record is left to be covered.
    assert not (output / ".graphify_requests").exists()


def test_a_prepared_publication_covers_no_queued_work(tmp_path) -> None:
    """Never retire work on behalf of a candidate prepared before the request.

    A compatibility adapter builds its candidate before handing it over, so the
    owning module cannot know which moment of the Corpus that candidate
    describes. Covering a queued request from it would retire work nobody did.
    """
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
        Queued,
        ReturnWhenQueued,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    source = tmp_path / "prepared.py"
    source.write_text("PREPARED = True\n", encoding="utf-8")
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    queued = owner.code_update(
        CodeUpdateRequest((Path("later.py"),)),
        completion=ReturnWhenQueued(),
    )
    assert isinstance(queued, Queued)

    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(
                        {
                            "id": "prepared",
                            "label": "Prepared",
                            "source_file": str(source),
                            "file_type": "code",
                        },
                    ),
                ),
            ),
            root_marker=str(tmp_path),
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    pending = _coordinator(tmp_path, output).pending()
    assert [request.request_id for request in pending] == [queued.request_id]


def test_a_request_accepted_mid_execution_is_covered_by_an_extra_pass(
    tmp_path,
    monkeypatch,
) -> None:
    """Cover a request that arrives mid-run with work that could have seen it.

    A hook that commits while this executor is scanning cannot be covered by the
    generation whose discovery already happened, so the executor makes another
    pass for it rather than either retiring it unearned or leaving it queued.
    """
    from graphify import detect as detect_module
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )
    from graphify.generation._coordination import _AcceptedRequest

    output = tmp_path / "graphify-out"
    (tmp_path / "first.py").write_text(
        "class First:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    coordinator = _coordinator(tmp_path, output)
    accepted_before = coordinator.accept("code-update", changed_paths=("first.py",))
    real_detect = detect_module.detect
    discoveries: list[None] = []
    late: list[_AcceptedRequest] = []

    def accept_a_late_request(*args, **kwargs):
        discoveries.append(None)
        if len(discoveries) == 1:
            # Stands in for a second hook accepted while this scan is running.
            (tmp_path / "second.py").write_text(
                "class Second:\n    def run(self):\n        return 1\n",
                encoding="utf-8",
            )
            late.append(
                coordinator.accept("code-update", changed_paths=("second.py",))
            )
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(detect_module, "detect", accept_a_late_request)

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    # A second discovery ran: the first one predated the late request entirely.
    assert len(discoveries) >= 2
    assert coordinator.is_covered(accepted_before)
    assert coordinator.is_covered(late[0])
    assert coordinator.pending() == ()
    assert _graph_labels(output) >= {"First", "Second"}


# --- one executor lease -------------------------------------------------------

_HOLD_LEASE_SCRIPT = """
import sys
import time
from pathlib import Path

from graphify.generation._coordination import _LeaseOutcome, _RequestCoordinator
from graphify.generation._types import Corpus

root, output, ready, stop = (Path(argument) for argument in sys.argv[1:5])
coordinator = _RequestCoordinator(Corpus(root=root, output=output))
with coordinator.executor_lease() as lease:
    assert lease is _LeaseOutcome.HELD, lease
    ready.write_text("held\\n", encoding="utf-8")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not stop.exists():
        time.sleep(0.05)
print("released", flush=True)
"""


def test_a_second_executor_cannot_publish_while_the_lease_is_held(
    tmp_path,
    monkeypatch,
) -> None:
    """Give exactly one process publication ownership for a Corpus."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        OperationFailed,
    )

    output = tmp_path / "graphify-out"
    (tmp_path / "contended.py").write_text(
        "class Contended:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    holder = _start_child(
        tmp_path / "hold.py",
        _HOLD_LEASE_SCRIPT,
        str(tmp_path),
        str(output),
        str(ready),
        str(stop),
    )
    try:
        _wait_for(ready.exists, description="the child to hold the executor lease")
        monkeypatch.setenv("GRAPHIFY_EXECUTOR_LEASE_TIMEOUT", "1")

        blocked = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
            CodeUpdateRequest()
        )

        assert isinstance(blocked, OperationFailed)
        assert "lease" in blocked.reason
        # A refused executor must not have published anything behind the holder.
        assert not (output / "graph.json").exists()
        # The work is still accepted, so releasing the lease is enough to finish it.
        assert _coordinator(tmp_path, output).pending()
    finally:
        stop.write_text("stop\n", encoding="utf-8")
        stdout, stderr = holder.communicate(timeout=180)
        assert holder.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"

    monkeypatch.setenv("GRAPHIFY_EXECUTOR_LEASE_TIMEOUT", "120")
    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert "Contended" in _graph_labels(output)
    assert _coordinator(tmp_path, output).pending() == ()
    assert not (output / ".graphify_executor.json").exists()


def test_competing_executors_publish_one_valid_generation(tmp_path) -> None:
    """Serialize real concurrent executors onto one coherent Graph generation."""
    from graphify.generation import Corpus
    from graphify.generation._transaction import _PublicationTransaction

    output = tmp_path / "graphify-out"
    script = tmp_path / "compete.py"
    script.write_text(
        """
import sys
from pathlib import Path

from graphify.generation import CodeUpdateRequest, Corpus, CorpusGraph

root, output, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
# Write this executor's source before contending, so whichever executor runs
# discovery last observes the whole Corpus.
(root / f"{name}.py").write_text(
    f"class {name.capitalize()}:\\n    def run(self):\\n        return 1\\n",
    encoding="utf-8",
)
outcome = CorpusGraph(Corpus(root=root, output=output)).code_update(CodeUpdateRequest())
print(type(outcome).__name__, flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), str(output), name],
            cwd=str(tmp_path),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for name in ("alpha", "beta", "gamma")
    ]

    outcomes = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=300)
        assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
        outcomes.append(stdout.strip())

    assert set(outcomes) <= {"GenerationPublished", "CorpusStateAdvanced", "AlreadyCurrent"}
    transaction_corpus = Corpus(root=tmp_path, output=output)
    assert _PublicationTransaction.recover(transaction_corpus) is None
    assert _graph_labels(output) >= {"Alpha", "Beta", "Gamma"}
    assert _coordinator(tmp_path, output).pending() == ()
    assert not (output / ".graphify_executor.json").exists()


# --- lease recovery -----------------------------------------------------------


def test_a_terminated_executor_leaves_recoverable_work_and_lease(tmp_path) -> None:
    """Recover a lease and its accepted work after the executor is killed."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    output = tmp_path / "graphify-out"
    (tmp_path / "abandoned.py").write_text(
        "class Abandoned:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    ready = tmp_path / "ready"
    holder = _start_child(
        tmp_path / "abandon.py",
        """
import sys
import time
from pathlib import Path

from graphify.generation._coordination import _LeaseOutcome, _RequestCoordinator
from graphify.generation._types import Corpus

root, output, ready = (Path(argument) for argument in sys.argv[1:4])
coordinator = _RequestCoordinator(Corpus(root=root, output=output))
coordinator.accept("code-update", changed_paths=("abandoned.py",))
with coordinator.executor_lease() as lease:
    assert lease is _LeaseOutcome.HELD, lease
    ready.write_text("held\\n", encoding="utf-8")
    time.sleep(600)
""",
        str(tmp_path),
        str(output),
        str(ready),
    )
    _wait_for(ready.exists, description="the child to hold the executor lease")
    holder.kill()
    holder.communicate(timeout=60)
    assert holder.returncode != 0
    # The killed executor left both its lease and its accepted work behind.
    assert (output / ".graphify_executor.json").is_file()
    assert _coordinator(tmp_path, output).pending()

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert "Abandoned" in _graph_labels(output)
    assert _coordinator(tmp_path, output).pending() == ()
    assert not (output / ".graphify_executor.json").exists()


def test_an_expired_lease_from_another_host_is_recovered(tmp_path, monkeypatch) -> None:
    """Take over a lease whose holder can no longer be proven alive."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    output = tmp_path / "graphify-out"
    output.mkdir(parents=True)
    (tmp_path / "expired.py").write_text(
        "class Expired:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    # A lease held on another host cannot be checked for process liveness, so
    # only its heartbeat can prove it is still owned.
    (output / ".graphify_executor.json").write_text(
        json.dumps(
            {
                "schema": "graphify-executor-lease",
                "version": 1,
                "lease_id": "0" * 32,
                "pid": 1,
                "host": "a-host-that-is-not-this-one",
                "acquired_at": time.time() - 3600,
                "renewed_at": time.time() - 3600,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GRAPHIFY_EXECUTOR_LEASE_TTL", "5")

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert "Expired" in _graph_labels(output)
    assert not (output / ".graphify_executor.json").exists()


def test_a_live_foreign_lease_is_left_alone(tmp_path, monkeypatch) -> None:
    """Never take over a lease whose heartbeat is still current."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        OperationFailed,
    )

    output = tmp_path / "graphify-out"
    output.mkdir(parents=True)
    (tmp_path / "held.py").write_text("HELD = True\n", encoding="utf-8")
    lease = output / ".graphify_executor.json"
    lease.write_text(
        json.dumps(
            {
                "schema": "graphify-executor-lease",
                "version": 1,
                "lease_id": "1" * 32,
                "pid": 1,
                "host": "a-host-that-is-not-this-one",
                "acquired_at": time.time(),
                "renewed_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GRAPHIFY_EXECUTOR_LEASE_TTL", "600")
    monkeypatch.setenv("GRAPHIFY_EXECUTOR_LEASE_TIMEOUT", "1")

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, OperationFailed)
    assert not (output / "graph.json").exists()
    assert json.loads(lease.read_text(encoding="utf-8"))["lease_id"] == "1" * 32


def test_a_waiting_executor_returns_when_its_request_is_covered(tmp_path) -> None:
    """Complete a waiting caller as soon as another executor covers its request."""
    from graphify.generation._coordination import _LeaseOutcome

    output = tmp_path / "graphify-out"
    (tmp_path / "covered.py").write_text(
        "class Covered:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    coordinator = _coordinator(tmp_path, output)
    accepted = coordinator.accept("code-update", changed_paths=("covered.py",))
    ready = tmp_path / "ready"
    holder = _start_child(
        tmp_path / "hold.py",
        """
import sys
import time
from pathlib import Path

from graphify.generation._coordination import _LeaseOutcome, _RequestCoordinator
from graphify.generation._types import Corpus

root, output, ready = (Path(argument) for argument in sys.argv[1:4])
coordinator = _RequestCoordinator(Corpus(root=root, output=output))
with coordinator.executor_lease() as lease:
    assert lease is _LeaseOutcome.HELD, lease
    ready.write_text("held\\n", encoding="utf-8")
    # Cover the waiting caller's request the way a real executor does, then keep
    # holding the lease so the waiter cannot mistake acquisition for coverage.
    time.sleep(2)
    coordinator.cover(coordinator.pending(operations=frozenset({"code-update"})))
    time.sleep(5)
print("released", flush=True)
""",
        str(tmp_path),
        str(output),
        str(ready),
    )
    try:
        _wait_for(ready.exists, description="the child to hold the executor lease")

        with coordinator.executor_lease(until_covered=accepted) as lease:
            assert lease is _LeaseOutcome.COVERED
    finally:
        stdout, stderr = holder.communicate(timeout=180)
        assert holder.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"


# --- cross-platform process liveness -----------------------------------------


def test_process_liveness_distinguishes_a_live_process_from_an_exited_one() -> None:
    """Detect executor death the same way on Windows and POSIX."""
    from graphify.generation._coordination import _HOST, _process_is_alive

    assert _process_is_alive(os.getpid(), _HOST) is True
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=60)
    assert _process_is_alive(exited.pid, _HOST) is False
    # A holder on another host cannot be proven dead by a local pid check.
    assert _process_is_alive(exited.pid, "a-host-that-is-not-this-one") is True
