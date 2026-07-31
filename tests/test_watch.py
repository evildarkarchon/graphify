"""Adapter tests for watch.py.

Code update is a production operation owned by ``graphify.generation``; every
function in ``graphify.watch`` is now an adapter over it. These tests cover what
the adapters are responsible for — translating a caller's ask into a request,
rendering the terminal outcome that comes back, and mapping it to the boolean
the compatibility helper still returns — and deliberately not the lifecycle
behind it. Discovery, reconciliation, semantic custody, shrink safety,
publication, and recovery are covered against real Corpora in
tests/test_generation_code_update.py, tests/test_generation_contributions.py,
tests/test_generation_semantic_freshness.py, tests/test_generation_recovery.py,
and tests/test_generation_coordination.py.
"""
import json
import time
from pathlib import Path

import pytest

from graphify.watch import _WATCHED_EXTENSIONS


def _corpus(tmp_path: Path) -> Path:
    """Create a one-file Corpus and return its root."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _break_the_ledger(out: Path) -> None:
    """Corrupt the authoritative Source-contribution ledger.

    Publication refuses rather than reconciling against evidence it cannot
    validate, which is how these tests reach a refusal outcome through the
    production interface instead of inventing a failure seam.
    """
    (out / ".graphify_contributions.jsonl").write_text("not a ledger\n", encoding="utf-8")


# --- _WATCHED_EXTENSIONS ---

def test_watched_extensions_includes_code():
    assert ".py" in _WATCHED_EXTENSIONS
    assert ".ts" in _WATCHED_EXTENSIONS
    assert ".go" in _WATCHED_EXTENSIONS
    assert ".rs" in _WATCHED_EXTENSIONS

def test_watched_extensions_includes_docs():
    assert ".md" in _WATCHED_EXTENSIONS
    assert ".txt" in _WATCHED_EXTENSIONS
    assert ".pdf" in _WATCHED_EXTENSIONS

def test_watched_extensions_includes_images():
    assert ".png" in _WATCHED_EXTENSIONS
    assert ".jpg" in _WATCHED_EXTENSIONS

def test_watched_extensions_excludes_noise():
    # .json is now indexed (bash/JSON extractors added in #866)
    assert ".json" in _WATCHED_EXTENSIONS
    assert ".sh" in _WATCHED_EXTENSIONS
    assert ".pyc" not in _WATCHED_EXTENSIONS
    assert ".log" not in _WATCHED_EXTENSIONS


# --- check-update: reports the compatibility marker, never writes it ---

def test_check_update_no_flag_returns_true(tmp_path):
    """check_update returns True and is silent when needs_update flag is absent."""
    from graphify.watch import check_update
    assert check_update(tmp_path) is True


def test_check_update_with_flag_returns_true_and_prints(tmp_path, capsys):
    """check_update returns True and prints notification when flag exists."""
    from graphify.watch import check_update
    flag = tmp_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1")
    result = check_update(tmp_path)
    assert result is True
    out = capsys.readouterr().out
    assert "graphify --update" in out


def test_check_update_does_not_clear_flag(tmp_path):
    """check_update never removes the needs_update flag (clearing is LLM's job)."""
    from graphify.watch import check_update
    flag = tmp_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1")
    check_update(tmp_path)
    assert flag.exists()


def test_watch_raises_without_watchdog(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "watchdog.observers" or name == "watchdog.events":
            raise ImportError("mocked missing watchdog")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from graphify.watch import watch
    with pytest.raises(ImportError, match="watchdog not installed"):
        watch(tmp_path)


def _watchdog_available() -> bool:
    try:
        import watchdog  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _watchdog_available(), reason="watchdog not installed")
def test_watch_handler_honors_graphifyignore(tmp_path, monkeypatch):
    """gh-928: the watch Handler must short-circuit paths matching
    .graphifyignore so busy volumes (node_modules churn, build artefacts,
    Time Machine writes, …) don't wake the submission pipeline.
    """
    import threading
    from graphify import watch as watch_mod

    watch_root = tmp_path / ".hidden-parent" / "corpus"
    watch_root.mkdir(parents=True)
    (watch_root / ".graphifyignore").write_text("node_modules/\nbuild/\n", encoding="utf-8")
    (watch_root / "node_modules").mkdir()
    (watch_root / "build").mkdir()

    submitted: list[Path] = []
    monkeypatch.setattr(
        watch_mod,
        "_submit_code_update",
        lambda p, **kw: submitted.append(p) or True,
    )

    # Run watch() in a thread with a short debounce so we can verify the
    # post-debounce dispatch path actually runs on real events.
    t = threading.Thread(
        target=watch_mod.watch,
        args=(watch_root,),
        kwargs={"debounce": 0.2},
        daemon=True,
    )
    t.start()
    time.sleep(0.5)  # let observer.start() settle

    # Ignored writes — handler must drop these.
    (watch_root / "node_modules" / "junk.js").write_text("// noise\n", encoding="utf-8")
    (watch_root / "build" / "out.py").write_text("x = 1\n", encoding="utf-8")
    time.sleep(1.0)
    assert submitted == [], "ignored writes triggered a submission"

    # Non-ignored write — handler must accept and (after debounce) dispatch.
    (watch_root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not submitted:
        time.sleep(0.1)
    assert submitted, "non-ignored .py write should have been submitted"


@pytest.mark.skipif(not _watchdog_available(), reason="watchdog not installed")
def test_watch_loads_graphifyignore_once(tmp_path, monkeypatch):
    """gh-928: .graphifyignore must be parsed exactly once at watch() startup,
    not per filesystem event. Otherwise busy volumes re-read the file
    thousands of times per second.
    """
    import threading
    from graphify import watch as watch_mod
    from graphify import detect as detect_mod

    (tmp_path / ".graphifyignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()

    calls = {"n": 0}
    real_loader = detect_mod._load_graphifyignore

    def counting_loader(root, **kwargs):
        calls["n"] += 1
        return real_loader(root, **kwargs)

    # Patch the symbol the watch module imported at module-load time.
    monkeypatch.setattr(watch_mod, "_load_graphifyignore", counting_loader)
    monkeypatch.setattr(watch_mod, "_submit_code_update", lambda p, **kw: True)

    t = threading.Thread(target=watch_mod.watch, args=(tmp_path,), kwargs={"debounce": 0.2}, daemon=True)
    t.start()
    time.sleep(0.5)

    # Generate many events; loader must not be called again.
    for i in range(50):
        (tmp_path / "ignored" / f"f{i}.py").write_text("x\n", encoding="utf-8")
    time.sleep(0.7)
    assert calls["n"] == 1, f"_load_graphifyignore called {calls['n']} times; expected 1"


# The per-repo ``.rebuild.lock`` advisory flock these tests used to cover was
# retired for the cross-platform ``CorpusGraph`` executor lease (#7); its holder
# identity, release, and non-clobbering contention are covered by
# tests/test_generation_coordination.py against real competing subprocesses.


# --- the watcher submits changes; it does not own semantic-pending state ---

def test_watcher_submission_covers_every_relevant_change(tmp_path):
    """One submission reconciles the whole batch, code and document alike.

    The watcher no longer sorts a batch into "code" and "needs an LLM": it hands
    the changed paths to the operation, which decides from the active Graph
    generation what its evidence means.
    """
    from graphify.watch import _submit_code_update

    root = _corpus(tmp_path)
    (root / "guide.md").write_text("# Guide\n\nText.\n", encoding="utf-8")

    assert _submit_code_update(root, changed_paths=[root / "lib.py", root / "guide.md"])

    data = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    sources = {node.get("source_file") for node in data["nodes"]}
    assert "lib.py" in sources
    assert "guide.md" in sources


def test_watcher_submission_does_not_raise_the_pending_marker_itself(tmp_path):
    """A document with no Semantic evidence is not pending; nothing claims it is.

    The retired ``_notify_only`` raised the marker for any non-code change,
    including one the graph could represent structurally. The marker is now a
    projection of the generation's own Stale semantic evidence, so a document
    the Code update just represented leaves it lowered.
    """
    from graphify.watch import _submit_code_update

    root = _corpus(tmp_path)
    (root / "guide.md").write_text("# Guide\n", encoding="utf-8")

    assert _submit_code_update(root, changed_paths=[root / "guide.md"])
    assert not (root / "graphify-out" / "needs_update").exists()


def test_watcher_submission_discloses_stale_semantic_evidence(tmp_path, capsys):
    """Changed semantic sources are disclosed in the generation's own words."""
    from graphify.generation import stale_semantic_disclosure
    from graphify.watch import _submit_code_update

    root = _corpus(tmp_path)
    doc = root / "notes.md"
    doc.write_text("# Notes\n\nOriginal.\n", encoding="utf-8")
    out = root / "graphify-out"

    _publish_semantic_evidence_for(root, doc)
    doc.write_text("# Notes\n\nRewritten after interpretation.\n", encoding="utf-8")
    capsys.readouterr()

    assert _submit_code_update(root, changed_paths=[doc])

    printed = capsys.readouterr().out
    assert stale_semantic_disclosure(("notes.md",)) in printed
    # The operation, not the watcher, decided the marker should be raised.
    assert (out / "needs_update").read_text(encoding="utf-8") == "1"


def _publish_semantic_evidence_for(root: Path, doc: Path) -> None:
    """Publish a generation whose evidence for ``doc`` is Semantic, then stamp it.

    Uses the production Full-extraction compatibility handoff and the production
    manifest writer, so the resulting active generation is the same shape a real
    ``graphify extract`` leaves behind: a Semantic contribution for the document
    and a manifest that vouches for its interpretation.
    """
    from graphify.detect import detect
    from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _ManifestUpdate, _Publication

    detected = detect(root)
    CorpusGraph(Corpus(root=root, output=root / "graphify-out")).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=str(doc),
                    interpretation=_InterpretationKind.SEMANTIC,
                    nodes=(
                        {
                            "id": "notes:concept",
                            "label": "Interpreted idea",
                            "file_type": "concept",
                            "source_file": str(doc),
                        },
                    ),
                ),
            ),
            manifest=_ManifestUpdate(
                files=detected["files"],
                kind="semantic",
                root=root,
                scan_corpus={f for group in detected["files"].values() for f in group},
            ),
        ),
    )


# --- _rebuild_code: the compatibility adapter and its Boolean mapping ---

def test_rebuild_code_maps_a_published_generation_to_true(tmp_path, capsys):
    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path)
    assert _rebuild_code(root) is True
    assert "Graph generation published" in capsys.readouterr().out
    assert (root / "graphify-out" / "graph.json").is_file()


def test_rebuild_code_maps_an_already_current_corpus_to_true(tmp_path, capsys):
    """Nothing to do is not a failure: the request is covered either way."""
    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path)
    assert _rebuild_code(root) is True
    capsys.readouterr()

    assert _rebuild_code(root) is True
    assert "already covers this request" in capsys.readouterr().out


def test_rebuild_code_maps_a_refused_publication_to_false(tmp_path, capsys):
    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path)
    assert _rebuild_code(root) is True
    _break_the_ledger(root / "graphify-out")
    capsys.readouterr()

    assert _rebuild_code(root) is False
    assert "Publication refused" in capsys.readouterr().err


def _surviving_requests(out: Path) -> list[dict]:
    """Return the durable request records an uncovered submission left behind."""
    return [
        json.loads(record.read_text(encoding="utf-8"))
        for record in sorted((out / ".graphify_requests").iterdir())
        if record.suffix == ".json"
    ]


def test_rebuild_code_translates_hints_and_force_into_the_request(tmp_path):
    """The caller's change set and its ``force`` reach the durable request.

    Both are the adapter's whole job: ``force`` is authority the *request*
    carries, so whichever executor ends up covering it has to be able to read it
    even though this process was the one told to use it. Reading it back from
    the record an uncovered submission leaves behind is how that is visible
    without reaching into the operation.
    """
    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path)
    assert _rebuild_code(root) is True
    out = root / "graphify-out"
    _break_the_ledger(out)

    assert _rebuild_code(root, changed_paths=[root / "lib.py"], force=True) is False

    records = _surviving_requests(out)
    assert len(records) == 1
    assert records[0]["operation"] == "code-update"
    assert records[0]["force"] is True
    assert records[0]["changed_paths"] == [str(root / "lib.py")]


def test_rebuild_code_keeps_the_retired_options_import_compatible(tmp_path):
    """An installed hook's call still binds after the lifecycle moved.

    None of these can change what the operation does any more, which is the
    point: they exist so a hook written against the old helper keeps working
    through the transition window rather than failing with a TypeError.
    """
    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path)
    assert _rebuild_code(
        root,
        changed_paths=[root / "lib.py"],
        follow_symlinks=True,
        force=False,
        no_cluster=True,
        acquire_lock=False,
        block_on_lock=True,
    ) is True


def test_rebuild_code_deleted_cwd_without_repo_root_returns_false(tmp_path, monkeypatch, capsys):
    """A detached hook whose CWD was deleted fails before touching the Corpus."""
    from graphify.watch import _rebuild_code

    doomed = tmp_path / "doomed"
    doomed.mkdir()
    monkeypatch.chdir(doomed)
    monkeypatch.delenv("GRAPHIFY_REPO_ROOT", raising=False)
    try:
        import os
        os.rmdir(doomed)
    except OSError:
        pytest.skip("platform keeps a deleted CWD alive")

    assert _rebuild_code(Path("."), acquire_lock=False) is False
    assert "current working directory" in capsys.readouterr().out


def test_rebuild_code_deleted_cwd_uses_graphify_repo_root(tmp_path, monkeypatch):
    """GRAPHIFY_REPO_ROOT recovers the relative-path case hooks actually hit."""
    import os

    from graphify.watch import _rebuild_code

    root = _corpus(tmp_path / "repo")
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    monkeypatch.chdir(doomed)
    monkeypatch.setenv("GRAPHIFY_REPO_ROOT", str(root))
    try:
        os.rmdir(doomed)
    except OSError:
        pytest.skip("platform keeps a deleted CWD alive")

    assert _rebuild_code(Path(".")) is True
    assert (root / "graphify-out" / "graph.json").is_file()


# --- _background_code_update: the completion policy hooks submit with ---

def test_background_code_update_queues_durably_before_executing(tmp_path, capsys):
    """A hook is told ``Queued`` first, then covers the request as executor."""
    from graphify.watch import _background_code_update

    root = _corpus(tmp_path)
    assert _background_code_update(root, changed_paths=[root / "lib.py"]) is True

    printed = capsys.readouterr().out
    assert "Queued as " in printed
    assert "Graph generation published" in printed
    assert (root / "graphify-out" / "graph.json").is_file()


def test_background_code_update_leaves_its_request_accepted_when_it_cannot_publish(
    tmp_path,
    capsys,
):
    """The change set outlives the process that submitted it.

    This is the whole reason a background caller queues durably first: the
    executor pass refuses, and the request is still on disk for whichever
    executor manages to cover it, instead of dying with this hook.
    """
    from graphify.watch import _background_code_update

    root = _corpus(tmp_path)
    assert _background_code_update(root) is True
    out = root / "graphify-out"
    assert not list((out / ".graphify_requests").iterdir())

    _break_the_ledger(out)
    capsys.readouterr()

    assert _background_code_update(root, changed_paths=[root / "lib.py"]) is False
    assert "Publication refused" in capsys.readouterr().err
    assert list((out / ".graphify_requests").iterdir())
