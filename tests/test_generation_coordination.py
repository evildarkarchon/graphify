"""Production-operation tests for request acceptance, coalescing, and leases.

Every test drives real temporary Corpora through the production ``CorpusGraph``
interface and its private coordination implementation. Cross-process behavior is
exercised with real subprocesses that are really terminated, because a durable
queue, a recoverable lease, and a coalescing rule that must not lose work only
mean anything against a process that dies.
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


def _plan(root: Path, output: Path):
    """Return the coalesced plan the next executor of this Corpus would run."""
    from graphify.generation._coordination import _coalesce

    return _coalesce(_coordinator(root, output).pending())


_QUEUE_CODE_UPDATE_SCRIPT = """
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
hints = tuple(Path(hint) for hint in sys.argv[3:] if hint)
outcome = CorpusGraph(Corpus(root=root, output=output)).code_update(
    CodeUpdateRequest(hints),
    completion=ReturnWhenQueued(),
)
assert isinstance(outcome, Queued), outcome
print(outcome.request_id, flush=True)
"""


# --- coalescing changed-path hints -------------------------------------------


def test_changed_path_hints_from_separate_processes_union_deterministically(
    tmp_path,
) -> None:
    """Union every submitter's hints once, in acceptance order, every time.

    Three real processes each accept a request and exit. The executor that
    eventually runs sees one merged change set containing every hint exactly
    once, and reading the same queue again produces the identical plan — a hint
    must not depend on which process happened to survive to run the work.
    """
    output = tmp_path / "graphify-out"
    for index, hints in enumerate(
        (("alpha.py", "shared.py"), ("beta.py",), ("shared.py", "gamma.py"))
    ):
        _run_child(
            tmp_path / f"queue{index}.py",
            _QUEUE_CODE_UPDATE_SCRIPT,
            str(tmp_path),
            str(output),
            *hints,
        )

    plan = _plan(tmp_path, output)

    assert len(plan) == 1
    unit = plan[0]
    assert unit.operation == "code-update"
    # Every hint survives, exactly once, and in the order the requests were
    # accepted rather than the order the processes happened to finish in.
    accepted = _coordinator(tmp_path, output).pending()
    expected: list[str] = []
    for request in accepted:
        expected.extend(
            hint for hint in request.changed_paths if hint not in expected
        )
    assert list(unit.changed_paths) == expected
    assert set(unit.changed_paths) == {
        "alpha.py",
        "beta.py",
        "gamma.py",
        "shared.py",
    }
    assert len(unit.covers) == 3
    # Deterministic: the same queue plans the same way on every executor.
    assert _plan(tmp_path, output) == plan


def test_a_request_without_hints_widens_the_union_to_the_whole_corpus(
    tmp_path,
) -> None:
    """Treat a hintless request as asking about every source, not about none.

    A post-checkout hook accepts a whole-Corpus rebuild with no hints. Merging
    that in as zero files would quietly narrow the covering operation to the
    other submitters' paths — the exact loss the path-only pending file used to
    cause.
    """
    output = tmp_path / "graphify-out"
    _run_child(
        tmp_path / "hinted.py",
        _QUEUE_CODE_UPDATE_SCRIPT,
        str(tmp_path),
        str(output),
        "narrow.py",
    )
    _run_child(
        tmp_path / "whole.py",
        _QUEUE_CODE_UPDATE_SCRIPT,
        str(tmp_path),
        str(output),
    )

    plan = _plan(tmp_path, output)

    assert len(plan) == 1
    assert plan[0].changed_paths == ()
    assert len(plan[0].covers) == 2


# --- coalescing operations ----------------------------------------------------


@pytest.mark.parametrize(
    "arrival",
    [
        ("code-update", "full-extraction", "reclustering"),
        ("code-update", "reclustering", "full-extraction"),
        ("reclustering", "code-update", "full-extraction"),
        ("full-extraction", "code-update", "reclustering"),
        ("reclustering", "full-extraction", "code-update"),
        ("full-extraction", "reclustering", "code-update"),
    ],
)
def test_a_queued_full_extraction_covers_ordinary_code_update_and_reclustering(
    tmp_path,
    arrival,
) -> None:
    """Let one Full extraction absorb the ordinary work it would redo anyway.

    Every arrival order is exercised because coalescing must depend only on what
    is outstanding. Neither Code update nor Reclustering covers the other, so an
    order that opens both units before the Full extraction arrives is the case
    where a merge that never revisits an earlier unit would plan two operations
    for work one operation does.
    """
    output = tmp_path / "graphify-out"
    coordinator = _coordinator(tmp_path, output)
    accepted = [
        coordinator.accept(
            operation,
            changed_paths=("changed.py",) if operation == "code-update" else (),
        )
        for operation in arrival
    ]

    plan = _plan(tmp_path, output)

    assert len(plan) == 1
    unit = plan[0]
    assert unit.operation == "full-extraction"
    assert {covered.request_id for covered in unit.covers} == {
        request.request_id for request in accepted
    }
    # The Full extraction carried no hints, so the merged unit reconciles the
    # whole Corpus rather than only the Code update's changed file.
    assert unit.changed_paths == ()


def test_code_update_and_reclustering_never_cover_each_other(tmp_path) -> None:
    """Keep operations that reconcile different halves of a generation apart.

    Reclustering republishes community identity from the active Source
    contributions and performs no discovery; a Code update does the opposite.
    Merging them would mean one of the two requests was answered by work that
    never did it.
    """
    output = tmp_path / "graphify-out"
    coordinator = _coordinator(tmp_path, output)
    update = coordinator.accept("code-update", changed_paths=("changed.py",))
    recluster = coordinator.accept("reclustering")

    plan = _plan(tmp_path, output)

    assert [unit.operation for unit in plan] == ["code-update", "reclustering"]
    assert [covered.request_id for unit in plan for covered in unit.covers] == [
        update.request_id,
        recluster.request_id,
    ]


# --- coalescing explicit authority --------------------------------------------


def test_coalescing_carries_every_explicit_authority_onto_the_covering_operation(
    tmp_path,
) -> None:
    """Never optimize away a safety rule a caller took responsibility for."""
    from graphify.generation._coordination import (
        _PolicyReplacement,
        _RequestedAuthority,
        _SemanticLabeling,
    )

    output = tmp_path / "graphify-out"
    coordinator = _coordinator(tmp_path, output)
    policy = _PolicyReplacement(excludes=("vendor/**",), gitignore=False)
    labeling = _SemanticLabeling(backend="anthropic", model="a-model", refresh_all=True)
    coordinator.accept(
        "code-update",
        changed_paths=("changed.py",),
        authority=_RequestedAuthority(force=True),
    )
    coordinator.accept(
        "full-extraction",
        authority=_RequestedAuthority(
            allow_partial_publication=True,
            policy_replacement=policy,
        ),
    )
    coordinator.accept(
        "reclustering",
        authority=_RequestedAuthority(
            adopt_curated_labels=True,
            semantic_labeling=labeling,
        ),
    )

    plan = _plan(tmp_path, output)

    assert len(plan) == 1
    unit = plan[0]
    assert unit.operation == "full-extraction"
    assert unit.authority == _RequestedAuthority(
        force=True,
        allow_partial_publication=True,
        adopt_curated_labels=True,
        policy_replacement=policy,
        semantic_labeling=labeling,
    )


def test_disagreeing_exclusive_policies_are_kept_as_separate_units(tmp_path) -> None:
    """Refuse to invent a merged policy neither caller asked for.

    A Semantic-labeling policy is one-shot and a policy replacement redefines
    the Corpus, so two differing ones cannot be folded together. Each keeps its
    own unit instead of one of them being silently discarded.
    """
    from graphify.generation._coordination import (
        _PolicyReplacement,
        _RequestedAuthority,
        _SemanticLabeling,
    )

    output = tmp_path / "graphify-out"
    coordinator = _coordinator(tmp_path, output)
    accepted = [
        coordinator.accept("full-extraction", authority=authority)
        for authority in (
            _RequestedAuthority(
                policy_replacement=_PolicyReplacement(excludes=("vendor/**",))
            ),
            _RequestedAuthority(
                policy_replacement=_PolicyReplacement(excludes=("build/**",))
            ),
        )
    ] + [
        coordinator.accept("reclustering", authority=authority)
        for authority in (
            _RequestedAuthority(semantic_labeling=_SemanticLabeling(backend="openai")),
            _RequestedAuthority(
                semantic_labeling=_SemanticLabeling(backend="anthropic")
            ),
        )
    ]

    plan = _plan(tmp_path, output)

    # Two units, because each disagreeing pair had to split — but a Reclustering
    # whose labeling policy does not disagree still joins a Full extraction, so
    # splitting is what disagreement costs rather than the default.
    assert len(plan) == 2
    assert [unit.operation for unit in plan] == ["full-extraction"] * 2
    assert [unit.authority.policy_replacement for unit in plan] == [
        _PolicyReplacement(excludes=("vendor/**",)),
        _PolicyReplacement(excludes=("build/**",)),
    ]
    assert [unit.authority.semantic_labeling for unit in plan] == [
        _SemanticLabeling(backend="openai"),
        _SemanticLabeling(backend="anthropic"),
    ]
    # Every request is still covered by exactly one unit, and no unit carries a
    # policy that disagrees with one a request it covers asked for.
    assert sorted(covered.request_id for unit in plan for covered in unit.covers) == (
        sorted(request.request_id for request in accepted)
    )
    for unit in plan:
        for covered in unit.covers:
            assert covered.authority.policy_replacement in (
                None,
                unit.authority.policy_replacement,
            )
            assert covered.authority.semantic_labeling in (
                None,
                unit.authority.semantic_labeling,
            )


def test_an_operation_cannot_accept_authority_it_cannot_exercise(tmp_path) -> None:
    """Refuse an impossible promise before it becomes durable state.

    Code update is fail closed and never replaces Corpus policy, and no
    operation but Reclustering or Full extraction publishes labels. Recording
    such a request would put an authority in the queue that nothing can honor.
    """
    from graphify.generation._coordination import (
        _PolicyReplacement,
        _RequestedAuthority,
        _SemanticLabeling,
    )

    output = tmp_path / "graphify-out"
    coordinator = _coordinator(tmp_path, output)
    refused = [
        ("code-update", _RequestedAuthority(allow_partial_publication=True)),
        ("code-update", _RequestedAuthority(policy_replacement=_PolicyReplacement())),
        ("code-update", _RequestedAuthority(adopt_curated_labels=True)),
        ("code-update", _RequestedAuthority(semantic_labeling=_SemanticLabeling())),
        ("reclustering", _RequestedAuthority(allow_partial_publication=True)),
        ("reclustering", _RequestedAuthority(policy_replacement=_PolicyReplacement())),
    ]

    for operation, authority in refused:
        with pytest.raises(ValueError):
            coordinator.accept(operation, authority=authority)

    # ``force`` is the one authority every operation may exercise, so it is the
    # control: the refusals above are about which authority, not about carrying
    # any authority at all.
    coordinator.accept("code-update", authority=_RequestedAuthority(force=True))
    assert [request.operation for request in coordinator.pending()] == ["code-update"]


def test_requested_authority_survives_the_process_that_asked_for_it(
    tmp_path,
) -> None:
    """Read every authority back from disk after its submitter has exited."""
    from graphify.generation._coordination import (
        _PolicyReplacement,
        _RequestedAuthority,
        _SemanticLabeling,
    )

    output = tmp_path / "graphify-out"
    _run_child(
        tmp_path / "authorize.py",
        """
import sys
from pathlib import Path

from graphify.generation._coordination import (
    _PolicyReplacement,
    _RequestCoordinator,
    _RequestedAuthority,
    _SemanticLabeling,
)
from graphify.generation._types import Corpus

root, output = Path(sys.argv[1]), Path(sys.argv[2])
coordinator = _RequestCoordinator(Corpus(root=root, output=output))
coordinator.accept(
    "full-extraction",
    changed_paths=("changed.py",),
    authority=_RequestedAuthority(
        force=True,
        allow_partial_publication=True,
        adopt_curated_labels=True,
        policy_replacement=_PolicyReplacement(
            excludes=("vendor/**",),
            gitignore=False,
        ),
        semantic_labeling=_SemanticLabeling(
            backend="anthropic",
            model="a-model",
            concurrency=4,
            refresh_all=True,
        ),
    ),
)
""",
        str(tmp_path),
        str(output),
    )

    (request,) = _coordinator(tmp_path, output).pending()

    assert request.operation == "full-extraction"
    assert request.changed_paths == ("changed.py",)
    assert request.authority == _RequestedAuthority(
        force=True,
        allow_partial_publication=True,
        adopt_curated_labels=True,
        policy_replacement=_PolicyReplacement(
            excludes=("vendor/**",),
            gitignore=False,
        ),
        semantic_labeling=_SemanticLabeling(
            backend="anthropic",
            model="a-model",
            concurrency=4,
            refresh_all=True,
        ),
    )


def test_a_queued_force_authorization_reaches_the_executor_that_covers_it(
    tmp_path,
    monkeypatch,
) -> None:
    """Run the covering operation with the authority the queued request carried.

    A hook that authorized replacing the graph with a smaller one may lose the
    lease and exit. The executor that covers its change set never asked for
    ``force`` itself, so the authority has to travel with the request.

    The request the production executor builds is inspected rather than an
    output artifact, because a Code update reconciled from authoritative
    discovery accounts for its own shrinkage per source — ``force`` changes no
    artifact unless a loss occurs that reconciliation cannot explain, which this
    operation is designed never to produce. Coverage of the queued request is
    still asserted through the public queue state.
    """
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )
    from graphify.generation import _corpus_graph as corpus_graph_module

    output = tmp_path / "graphify-out"
    (tmp_path / "forced.py").write_text(
        "class Forced:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    _run_child(
        tmp_path / "force.py",
        """
import sys
from pathlib import Path

from graphify.generation._coordination import _RequestCoordinator, _RequestedAuthority
from graphify.generation._types import Corpus

root, output = Path(sys.argv[1]), Path(sys.argv[2])
_RequestCoordinator(Corpus(root=root, output=output)).accept(
    "code-update",
    changed_paths=("forced.py",),
    authority=_RequestedAuthority(force=True),
)
""",
        str(tmp_path),
        str(output),
    )
    executed: list[CodeUpdateRequest] = []
    real_execute = corpus_graph_module._execute_code_update

    def record(corpus, request):
        executed.append(request)
        return real_execute(corpus, request)

    monkeypatch.setattr(corpus_graph_module, "_execute_code_update", record)

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert executed and all(request.force for request in executed)
    assert _coordinator(tmp_path, output).pending() == ()


# --- waiting until covered ----------------------------------------------------


def test_wait_until_covered_returns_only_after_its_own_request_is_covered(
    tmp_path,
) -> None:
    """Complete a foreground call only once its own accepted request is retired.

    Work queued by processes that have already exited is covered by the same
    generation, so the interactive caller's completion still means "the Corpus
    now describes what I asked about" rather than "some rebuild happened".
    """
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )
    from graphify.generation._coordination import _coalesce

    output = tmp_path / "graphify-out"
    for name in ("earlier", "later"):
        (tmp_path / f"{name}.py").write_text(
            f"class {name.capitalize()}:\n    def run(self):\n        return 1\n",
            encoding="utf-8",
        )
        _run_child(
            tmp_path / f"queue_{name}.py",
            _QUEUE_CODE_UPDATE_SCRIPT,
            str(tmp_path),
            str(output),
            f"{name}.py",
        )
    (tmp_path / "waiting.py").write_text(
        "class Waiting:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest((Path("waiting.py"),))
    )

    assert isinstance(outcome, GenerationPublished)
    # Nothing is left accepted, so the waiting caller's own request was covered
    # by the generation this call published rather than deferred to a later one.
    assert _coordinator(tmp_path, output).pending() == ()
    assert _coalesce(_coordinator(tmp_path, output).pending()) == ()
    assert _graph_labels(output) >= {"Earlier", "Later", "Waiting"}


def test_a_killed_executor_strands_no_coalesced_request(tmp_path) -> None:
    """Leave every request of an interrupted plan for the next executor.

    The killed executor had already coalesced three submitters' work into one
    unit. Coverage happens only after the operation succeeds, so all three stay
    durably accepted and the next executor covers them together.
    """
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    output = tmp_path / "graphify-out"
    for name in ("alpha", "beta", "gamma"):
        (tmp_path / f"{name}.py").write_text(
            f"class {name.capitalize()}:\n    def run(self):\n        return 1\n",
            encoding="utf-8",
        )
        _run_child(
            tmp_path / f"queue_{name}.py",
            _QUEUE_CODE_UPDATE_SCRIPT,
            str(tmp_path),
            str(output),
            f"{name}.py",
        )
    assert len(_plan(tmp_path, output)[0].covers) == 3
    ready = tmp_path / "ready"
    executor = _start_child(
        tmp_path / "die.py",
        """
import sys
import time
from pathlib import Path

from graphify.generation._coordination import (
    _LeaseOutcome,
    _RequestCoordinator,
    _coalesce,
)
from graphify.generation._types import Corpus

root, output, ready = (Path(argument) for argument in sys.argv[1:4])
coordinator = _RequestCoordinator(Corpus(root=root, output=output))
with coordinator.executor_lease() as lease:
    assert lease is _LeaseOutcome.HELD, lease
    assert len(_coalesce(coordinator.pending())[0].covers) == 3
    ready.write_text("planned\\n", encoding="utf-8")
    time.sleep(600)
""",
        str(tmp_path),
        str(output),
        str(ready),
    )
    _wait_for(ready.exists, description="the child to plan its coalesced work")
    executor.kill()
    executor.communicate(timeout=60)
    assert len(_plan(tmp_path, output)[0].covers) == 3

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished)
    assert _graph_labels(output) >= {"Alpha", "Beta", "Gamma"}
    assert _coordinator(tmp_path, output).pending() == ()
    assert not (output / ".graphify_executor.json").exists()


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
