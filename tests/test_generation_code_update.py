"""Production-operation tests for the deterministic, LLM-free Code update."""

import contextlib
import json
import os
from pathlib import Path

import pytest


def _listing_fails(directory: Path) -> bool:
    """Return whether this host actually refuses to list ``directory``."""
    try:
        os.listdir(directory)
    except OSError:
        return True
    return False


@contextlib.contextmanager
def _unreadable(directory: Path):
    """Deny directory enumeration the way a real permission error does.

    Discovery incompleteness has to come from the filesystem, not from a stubbed
    scan, so the fail-closed rule runs against production ``detect``. The
    mechanism is the same ``chmod`` the detect-level walk-error test uses, so the
    test is skipped where it has no effect: as root, and on Windows, where the
    equivalent ACL deny also blocks the right to remove it again and would leave
    an undeletable temporary tree behind.
    """
    if os.name == "nt":
        pytest.skip("no reversible way to deny directory enumeration on Windows")
    if os.geteuid() == 0:
        pytest.skip("running as root: an unreadable directory cannot be simulated")
    os.chmod(directory, 0o000)
    try:
        if not _listing_fails(directory):
            pytest.skip("this host still enumerates an unreadable directory")
        yield
    finally:
        os.chmod(directory, 0o755)


def _ledger_records(output: Path) -> list[dict]:
    """Return the active Source-contribution records for one Corpus output."""
    lines = (
        (output / ".graphify_contributions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return [json.loads(line) for line in lines[1:]]


def test_first_run_code_update_initializes_a_code_only_generation(tmp_path) -> None:
    """Publish a valid code-only generation without a prior Full extraction."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "service.py").write_text(
        "class Service:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert "Service" in {node["label"] for node in graph["nodes"]}
    records = _ledger_records(output)
    assert {record["source"] for record in records} == {"service.py"}
    assert {record["interpretation"] for record in records} == {"structural"}
    assert (output / "manifest.json").is_file()
    assert (output / ".graphify_generation_complete").is_file()


def test_authoritative_discovery_reconciles_a_shrinking_deletion(tmp_path) -> None:
    """Retire a deleted source's evidence without needing shrink authority."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    (tmp_path / "gone.py").write_text("class Gone:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    (tmp_path / "gone.py").unlink()

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    assert {record["source"] for record in _ledger_records(output)} == {"kept.py"}
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert {node["source_file"] for node in graph["nodes"]} == {"kept.py"}


def test_authoritative_discovery_reconciles_additions_and_renames(tmp_path) -> None:
    """Follow a rename and admit a new source from discovery alone."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "old_name.py").write_text("class Moved:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    (tmp_path / "old_name.py").rename(tmp_path / "new_name.py")
    (tmp_path / "added.py").write_text("class Added:\n    pass\n", encoding="utf-8")

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    assert {record["source"] for record in _ledger_records(output)} == {
        "new_name.py",
        "added.py",
    }
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert {node["source_file"] for node in graph["nodes"]} == {
        "new_name.py",
        "added.py",
    }


def test_authoritative_discovery_retires_a_newly_excluded_source(tmp_path) -> None:
    """Remove contributions from a live file the Corpus no longer includes."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "bundled.py").write_text("class Bundled:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    assert "vendor/bundled.py" in {
        record["source"] for record in _ledger_records(output)
    }

    (tmp_path / ".graphifyignore").write_text("vendor/\n", encoding="utf-8")

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    assert {record["source"] for record in _ledger_records(output)} == {"kept.py"}
    assert (vendor / "bundled.py").is_file()


def test_an_omitted_changed_path_is_not_deletion_authority(tmp_path) -> None:
    """Reconcile a deleted source the caller's hints never mentioned."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    (tmp_path / "gone.py").write_text("class Gone:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    (tmp_path / "kept.py").write_text(
        "class Kept:\n    def added(self):\n        return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "gone.py").unlink()

    outcome = owner.code_update(CodeUpdateRequest((tmp_path / "kept.py",)))

    assert isinstance(outcome, GenerationPublished)
    assert {record["source"] for record in _ledger_records(output)} == {"kept.py"}


def test_a_changed_path_hint_outside_the_corpus_retires_nothing(tmp_path) -> None:
    """Treat an unknown hinted path as a hint, never as removal evidence."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    before = _ledger_records(output)

    owner.code_update(CodeUpdateRequest((tmp_path / "never_existed.py",)))

    assert _ledger_records(output) == before


def test_code_update_preserves_semantic_evidence_for_live_sources(tmp_path) -> None:
    """Keep evidence a Code update cannot re-derive and retire departed evidence."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    (tmp_path / "guide.md").write_text("# Guide\n\nText.\n", encoding="utf-8")
    (tmp_path / "retired.md").write_text("# Retired\n\nText.\n", encoding="utf-8")
    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))

    def _semantic(source: str, node_id: str) -> _SourceContribution:
        return _SourceContribution(
            source=source,
            interpretation=_InterpretationKind.SEMANTIC,
            nodes=(
                {
                    "id": node_id,
                    "label": node_id,
                    "source_file": source,
                    "file_type": "concept",
                },
            ),
        )

    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _semantic("guide.md", "guide_concept"),
                _semantic("retired.md", "retired_concept"),
            ),
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    (tmp_path / "retired.md").unlink()

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    records = {
        (record["source"], record["interpretation"])
        for record in _ledger_records(output)
    }
    assert ("guide.md", "semantic") in records
    assert ("service.py", "structural") in records
    assert not any(source == "retired.md" for source, _ in records)
    # A semantically interpreted document is not also AST quick-scanned: its
    # Semantic evidence is that source's whole representation.
    assert ("guide.md", "structural") not in records
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in graph["nodes"]}
    assert "guide_concept" in node_ids
    assert "retired_concept" not in node_ids


def test_an_unenumerable_scan_path_fails_closed_even_with_force(tmp_path) -> None:
    """Refuse publication when a path discovery must scan cannot be enumerated."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        PublicationRefused,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    active_graph = (output / "graph.json").read_bytes()
    active_ledger = (output / ".graphify_contributions.jsonl").read_bytes()

    # Discovery always scans the graph memory directory as well as the Corpus
    # root. A plain file in its place is a real filesystem state that leaves the
    # enumeration provably incomplete on every platform.
    memory = output / "memory"
    memory.write_text("not a directory", encoding="utf-8")

    refused = owner.code_update(CodeUpdateRequest())
    forced = owner.code_update(CodeUpdateRequest(force=True))

    assert isinstance(refused, PublicationRefused)
    assert "discovery was incomplete" in refused.reason
    # `force` authorizes a smaller graph, never a graph built from an unknown
    # Corpus, so it must not turn the refusal into a publication.
    assert isinstance(forced, PublicationRefused)
    assert (output / "graph.json").read_bytes() == active_graph
    assert (output / ".graphify_contributions.jsonl").read_bytes() == active_ledger


def test_an_unreadable_subtree_fails_closed_even_with_force(tmp_path) -> None:
    """Refuse publication when a permission error hid part of the Corpus."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        PublicationRefused,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    active_graph = (output / "graph.json").read_bytes()
    active_ledger = (output / ".graphify_contributions.jsonl").read_bytes()

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("class Hidden:\n    pass\n", encoding="utf-8")

    with _unreadable(locked):
        refused = owner.code_update(CodeUpdateRequest())
        forced = owner.code_update(CodeUpdateRequest(force=True))

    assert isinstance(refused, PublicationRefused)
    assert "discovery was incomplete" in refused.reason
    # `force` authorizes a smaller graph, never a graph built from an unknown
    # Corpus, so it must not turn the refusal into a publication.
    assert isinstance(forced, PublicationRefused)
    assert (output / "graph.json").read_bytes() == active_graph
    assert (output / ".graphify_contributions.jsonl").read_bytes() == active_ledger


def test_code_update_distinguishes_no_op_state_advance_and_graph_change(
    tmp_path,
) -> None:
    """Return a distinct outcome for each kind of Code update result."""
    from graphify.generation import (
        AlreadyCurrent,
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        GenerationPublished,
    )

    service = tmp_path / "service.py"
    service.write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    graph_path = output / "graph.json"
    ledger_path = output / ".graphify_contributions.jsonl"
    published_graph = graph_path.read_bytes()
    published_ledger = ledger_path.read_bytes()

    assert isinstance(owner.code_update(CodeUpdateRequest()), AlreadyCurrent)

    # A source with no structural evidence joins the Corpus: the Corpus state
    # advances while the materialized graph stays exactly as published.
    (tmp_path / "notes.txt").write_text("Release notes.\n", encoding="utf-8")
    advanced = owner.code_update(CodeUpdateRequest())

    assert isinstance(advanced, CorpusStateAdvanced)
    assert "manifest.json" in advanced.changed_artifacts
    assert "graph.json" not in advanced.changed_artifacts
    assert ".graphify_contributions.jsonl" not in advanced.changed_artifacts
    assert graph_path.read_bytes() == published_graph
    assert ledger_path.read_bytes() == published_ledger

    service.write_text(
        "class Service:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    changed = owner.code_update(CodeUpdateRequest())

    assert isinstance(changed, GenerationPublished)
    assert "graph.json" in changed.changed_artifacts
    assert ".graphify_contributions.jsonl" in changed.changed_artifacts


def test_an_unreadable_active_ledger_refuses_instead_of_retiring_evidence(
    tmp_path,
) -> None:
    """Refuse rather than reconcile against a ledger that cannot be validated."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        PublicationRefused,
    )

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    active_graph = (output / "graph.json").read_bytes()
    ledger = output / ".graphify_contributions.jsonl"
    ledger.write_text("not a contribution ledger\n", encoding="utf-8")

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, PublicationRefused)
    assert (output / "graph.json").read_bytes() == active_graph


def test_a_failed_structural_extraction_returns_a_terminal_failure(
    tmp_path, monkeypatch
) -> None:
    """Report an accepted operation's preparation failure as a closed outcome."""
    from graphify import extract as extract_module
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        OperationFailed,
    )

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    active_graph = (output / "graph.json").read_bytes()

    def _fail(*args, **kwargs):
        raise RuntimeError("tree-sitter grammar unavailable")

    # The whole structural extractor pass failing is the one preparation error a
    # Corpus cannot produce on its own; everything downstream of it is real.
    monkeypatch.setattr(extract_module, "extract", _fail)

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, OperationFailed)
    assert "tree-sitter grammar unavailable" in outcome.reason
    assert (output / "graph.json").read_bytes() == active_graph


def test_code_update_honors_the_active_corpus_build_policy(tmp_path) -> None:
    """Discover under the recorded policy instead of re-including excluded paths."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._publication import _Publication

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "bundled.py").write_text("class Bundled:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    recorded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            build_config={"excludes": ["vendor/"], "gitignore": False},
        ),
    )
    assert not isinstance(recorded, GenerationPublished)

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    assert {record["source"] for record in _ledger_records(output)} == {"kept.py"}


def test_an_unreadable_build_policy_refuses_rather_than_widening_the_corpus(
    tmp_path,
) -> None:
    """Fail closed when the recorded policy cannot be read."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        PublicationRefused,
    )

    (tmp_path / "kept.py").write_text("class Kept:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    (output / ".graphify_build.json").write_text("{not json", encoding="utf-8")

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, PublicationRefused)
    assert "build policy" in outcome.reason


def test_raw_publication_retires_the_previous_clustered_artifacts(tmp_path) -> None:
    """Stop presenting community identity that no longer describes the graph."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _Publication

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)
    clustered = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            report="# Service graph\n",
            analysis={"communities": {"0": ["service"]}},
            labels={"0": "Service"},
            label_signatures={"0": "sig"},
        ),
    )
    assert not isinstance(clustered, GenerationPublished)
    for name in (
        "GRAPH_REPORT.md",
        ".graphify_analysis.json",
        ".graphify_labels.json",
        ".graphify_labels.json.sig",
    ):
        assert (output / name).is_file(), name

    (tmp_path / "added.py").write_text("class Added:\n    pass\n", encoding="utf-8")
    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    for name in (
        "GRAPH_REPORT.md",
        ".graphify_analysis.json",
        ".graphify_labels.json",
        ".graphify_labels.json.sig",
    ):
        assert not (output / name).exists(), name
        assert name in outcome.changed_artifacts, name


def test_code_update_adopts_legacy_output_without_destroying_its_evidence(
    tmp_path,
) -> None:
    """Reconcile a pre-ledger Corpus instead of treating it as having no evidence."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        GenerationPublished,
    )

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    output.mkdir()
    # A graph written before Source contributions existed: it carries semantic
    # evidence for a live document that a Code update cannot re-derive.
    (tmp_path / "guide.md").write_text("# Guide\n\nText.\n", encoding="utf-8")
    (output / "graph.json").write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {
                        "id": "guide_concept",
                        "label": "Service concept",
                        "file_type": "concept",
                        "source_file": "guide.md",
                    },
                    {"id": "orphan", "label": "Unattributed", "file_type": "concept"},
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).code_update(
        CodeUpdateRequest()
    )

    assert isinstance(outcome, GenerationPublished), outcome
    records = {
        (record["source"], record["interpretation"])
        for record in _ledger_records(output)
    }
    assert ("service.py", "structural") in records
    assert ("guide.md", "legacy-attributed") in records
    assert ("@legacy/unattributed", "legacy-unattributed") in records
    node_ids = {
        node["id"]
        for node in json.loads((output / "graph.json").read_text(encoding="utf-8"))[
            "nodes"
        ]
    }
    assert {"guide_concept", "orphan"} <= node_ids
