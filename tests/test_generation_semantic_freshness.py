"""Production-operation tests for per-source Stale semantic evidence.

Every test drives the real ``CorpusGraph`` operations over a real temporary
Corpus: semantic evidence is seeded through the production Full-extraction
handoff, freshness is advanced by editing files on disk, and the assertions read
the published Graph generation's own artifacts.
"""

import contextlib
import json
from pathlib import Path


def _ledger_records(output: Path) -> list[dict]:
    """Return the active Source-contribution records for one Corpus output."""
    lines = (
        (output / ".graphify_contributions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return [json.loads(line) for line in lines[1:]]


def _record_for(output: Path, source: str, interpretation: str) -> dict:
    """Return one ledger record, failing the test when it is absent."""
    for record in _ledger_records(output):
        if record["source"] == source and record["interpretation"] == interpretation:
            return record
    raise AssertionError(
        f"no {interpretation} contribution for {source!r} in {_ledger_records(output)}"
    )


def _manifest(output: Path) -> dict:
    """Return the published manifest keyed by its portable relative paths."""
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _semantic_contribution(source: str, node_id: str):
    """Build one interpreted contribution for a document source."""
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )

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


def _carried_forward(output: Path, *, replacing: str) -> tuple:
    """Return the active contributions except those attributed to ``replacing``."""
    from graphify.generation._contributions import _iter_contribution_ledger

    return tuple(
        _as_contribution(record)
        for record in _iter_contribution_ledger(
            output / ".graphify_contributions.jsonl"
        )
        if record.source != replacing
    )


def _as_contribution(record):
    """Return one ledger record as a publishable contribution."""
    from graphify.generation._contributions import _SourceContribution

    return _SourceContribution(
        source=record.source,
        interpretation=record.interpretation,
        nodes=record.nodes,
        edges=record.edges,
        hyperedges=record.hyperedges,
        provisional=record.provisional,
    )


def _seed_interpreted_corpus(tmp_path: Path, *documents: str):
    """Publish a Full extraction that interprets ``documents`` and stamps them.

    The manifest is written with ``kind="both"`` so each document starts with a
    current semantic hash — the same state a successful Full extraction leaves
    behind. Without it every seeded source would already be pending and the
    tests could not tell "was interpreted, then changed" from "was never
    interpreted".
    """
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._publication import _ManifestUpdate, _Publication

    (tmp_path / "service.py").write_text(
        "class Service:\n    pass\n",
        encoding="utf-8",
    )
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    files = {
        "code": [str(tmp_path / "service.py")],
        "document": [str(tmp_path / name) for name in documents],
    }
    corpus_paths = {path for group in files.values() for path in group}
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=tuple(
                _semantic_contribution(name, f"{Path(name).stem}_concept")
                for name in documents
            ),
            manifest=_ManifestUpdate(
                files=files,
                kind="both",
                root=tmp_path,
                scan_corpus=corpus_paths,
            ),
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    return owner, output


def test_a_changed_semantic_source_keeps_its_evidence_as_stale(tmp_path) -> None:
    """Retain a changed document's interpretation as Stale semantic evidence."""
    from graphify.generation import (
        CodeUpdateRequest,
        GenerationPublished,
        stale_semantic_sources,
    )

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    assert stale_semantic_sources(output) == ()

    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")

    outcome = owner.code_update(CodeUpdateRequest())

    assert isinstance(outcome, GenerationPublished)
    record = _record_for(output, "guide.md", "semantic")
    assert record["stale"] is True
    # Custody, not a freshness claim: the evidence is still queryable.
    assert [node["id"] for node in record["nodes"]] == ["guide_concept"]
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert "guide_concept" in {node["id"] for node in graph["nodes"]}
    assert stale_semantic_sources(output) == ("guide.md",)


def test_a_changed_semantic_source_becomes_pending(tmp_path) -> None:
    """Record the changed document as needing reinterpretation, not reinterpreted."""
    from graphify.generation import CodeUpdateRequest, GenerationPublished

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    assert _manifest(output)["guide.md"]["semantic_hash"]

    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")

    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    assert _manifest(output)["guide.md"]["semantic_hash"] == ""
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_an_unchanged_semantic_source_stays_current(tmp_path) -> None:
    """Leave an untouched document's interpretation current and undisclosed."""
    from graphify.generation import CodeUpdateRequest, stale_semantic_sources

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")

    owner.code_update(CodeUpdateRequest())

    record = _record_for(output, "guide.md", "semantic")
    assert record.get("stale", False) is False
    assert _manifest(output)["guide.md"]["semantic_hash"]
    assert not (output / "needs_update").exists()
    assert stale_semantic_sources(output) == ()


def test_a_deleted_semantic_source_loses_every_attributed_contribution(
    tmp_path,
) -> None:
    """Remove all evidence attributed to a source that left the Corpus."""
    from graphify.generation import (
        CodeUpdateRequest,
        GenerationPublished,
        stale_semantic_sources,
    )

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")

    (tmp_path / "guide.md").unlink()

    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    assert not any(
        record["source"] == "guide.md" for record in _ledger_records(output)
    )
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert "guide_concept" not in {node["id"] for node in graph["nodes"]}
    # A departed source is not pending work: there is nothing left to reinterpret.
    assert stale_semantic_sources(output) == ()
    assert not (output / "needs_update").exists()


def test_a_newly_excluded_semantic_source_loses_every_attributed_contribution(
    tmp_path,
) -> None:
    """Drop evidence for a live file the active build policy no longer admits."""
    from graphify.generation import (
        CodeUpdateRequest,
        GenerationPublished,
        stale_semantic_sources,
    )

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (output / ".graphify_build.json").write_text(
        json.dumps({"excludes": ["guide.md"], "gitignore": True}),
        encoding="utf-8",
    )

    assert isinstance(owner.code_update(CodeUpdateRequest()), GenerationPublished)

    assert not any(
        record["source"] == "guide.md" for record in _ledger_records(output)
    )
    assert (tmp_path / "guide.md").is_file()
    assert stale_semantic_sources(output) == ()


def test_code_update_never_clears_pending_semantic_state(tmp_path) -> None:
    """Leave a pending marker raised by other work alone across Code updates."""
    from graphify.generation import CodeUpdateRequest, Corpus, CorpusGraph

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    owner.code_update(CodeUpdateRequest())
    # A pending source recorded outside this operation — a watcher notification,
    # for instance. A Code update performs no interpretation, so it has no
    # evidence that the work is done.
    (output / "needs_update").write_text("1", encoding="utf-8")

    (tmp_path / "service.py").write_text(
        "class Service:\n    def added(self):\n        return 1\n",
        encoding="utf-8",
    )
    owner.code_update(CodeUpdateRequest())

    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_the_code_rebuild_entrypoint_never_clears_pending_semantic_state(
    tmp_path,
) -> None:
    """Keep pending state across `graphify update`'s LLM-free rebuild path.

    This is the entrypoint a user actually runs, and it still publishes through
    the compatibility handoff rather than the owned operation. It performs no
    interpretation either, so it must not retire the pending marker.
    """
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    output = corpus / "graphify-out"
    output.mkdir()
    (output / "needs_update").write_text("1", encoding="utf-8")

    assert _rebuild_code(corpus, acquire_lock=False) is True

    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_stale_semantic_evidence_survives_repeated_code_updates(tmp_path) -> None:
    """Keep a source stale and pending until interpretation actually succeeds."""
    from graphify.generation import (
        AlreadyCurrent,
        CodeUpdateRequest,
        stale_semantic_sources,
    )

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")
    owner.code_update(CodeUpdateRequest())

    repeated = owner.code_update(CodeUpdateRequest())

    assert isinstance(repeated, AlreadyCurrent)
    assert _record_for(output, "guide.md", "semantic")["stale"] is True
    assert stale_semantic_sources(output) == ("guide.md",)
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_a_reinterpreted_source_clears_its_stale_marking(tmp_path) -> None:
    """Return a source to current once a Full extraction replaces its evidence."""
    from graphify.generation import (
        CodeUpdateRequest,
        FullExtractionRequest,
        GenerationPublished,
        stale_semantic_sources,
    )
    from graphify.generation._publication import _ManifestUpdate, _Publication

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")
    owner.code_update(CodeUpdateRequest())
    assert stale_semantic_sources(output) == ("guide.md",)

    files = {
        "code": [str(tmp_path / "service.py")],
        "document": [str(tmp_path / "guide.md")],
    }
    # A Full extraction republishes the whole Corpus, so the structural evidence
    # the intervening Code update derived is carried alongside the freshly
    # interpreted document rather than dropped.
    reinterpreted = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=_carried_forward(output, replacing="guide.md")
            + (_semantic_contribution("guide.md", "guide_concept"),),
            manifest=_ManifestUpdate(
                files=files,
                kind="both",
                root=tmp_path,
                scan_corpus={path for group in files.values() for path in group},
            ),
        ),
    )
    assert isinstance(reinterpreted, GenerationPublished)

    assert stale_semantic_sources(output) == ()
    # The next Code update observes current interpretation and agrees.
    owner.code_update(CodeUpdateRequest())
    assert _record_for(output, "guide.md", "semantic").get("stale", False) is False
    assert stale_semantic_sources(output) == ()


def test_stale_semantic_sources_reads_a_ledgerless_corpus_as_empty(tmp_path) -> None:
    """Report no stale evidence rather than raising when nothing is published."""
    from graphify.generation import stale_semantic_sources

    assert stale_semantic_sources(tmp_path / "graphify-out") == ()


# --- disclosure -------------------------------------------------------------


def test_query_discloses_stale_semantic_evidence(tmp_path, monkeypatch, capsys) -> None:
    """Announce stale evidence at the top of a query answer."""
    import sys

    from graphify.cli import dispatch_command
    from graphify.generation import CodeUpdateRequest

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")
    owner.code_update(CodeUpdateRequest())

    monkeypatch.setattr(
        sys,
        "argv",
        ["graphify", "query", "guide", "--graph", str(output / "graph.json")],
    )
    dispatch_command("query")

    out = capsys.readouterr().out
    assert "STALE SEMANTIC EVIDENCE" in out
    assert "guide.md" in out


def test_query_is_silent_when_no_semantic_evidence_is_stale(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Leave a current generation's query answer untouched."""
    import sys

    from graphify.cli import dispatch_command

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    _owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")

    monkeypatch.setattr(
        sys,
        "argv",
        ["graphify", "query", "guide", "--graph", str(output / "graph.json")],
    )
    dispatch_command("query")

    assert "STALE SEMANTIC EVIDENCE" not in capsys.readouterr().out


def test_explain_discloses_stale_semantic_evidence(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Disclose stale evidence on the focused-concept reader too."""
    import sys

    from graphify.cli import dispatch_command
    from graphify.generation import CodeUpdateRequest

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")
    owner.code_update(CodeUpdateRequest())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graphify",
            "explain",
            "guide_concept",
            "--graph",
            str(output / "graph.json"),
        ],
    )
    # `explain` exits early for an unmatched label; either way the disclosure
    # has to have been printed before any node detail.
    with contextlib.suppress(SystemExit):
        dispatch_command("explain")

    assert "STALE SEMANTIC EVIDENCE" in capsys.readouterr().out


def test_path_discloses_stale_semantic_evidence_before_its_answer(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Disclose stale evidence even when the relationship reader finds nothing."""
    import sys

    from graphify.cli import dispatch_command
    from graphify.generation import CodeUpdateRequest

    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    owner, output = _seed_interpreted_corpus(tmp_path, "guide.md")
    (tmp_path / "guide.md").write_text("# Guide\n\nRewritten.\n", encoding="utf-8")
    owner.code_update(CodeUpdateRequest())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graphify",
            "path",
            "guide_concept",
            "Service",
            "--graph",
            str(output / "graph.json"),
        ],
    )
    with contextlib.suppress(SystemExit):
        dispatch_command("path")

    assert "STALE SEMANTIC EVIDENCE" in capsys.readouterr().out


def test_report_discloses_stale_semantic_evidence() -> None:
    """Name every source whose interpretation is pending in the report."""
    import networkx as nx

    from graphify.report import generate

    G = nx.Graph()
    G.add_node("guide_concept", label="Guide", file_type="concept")
    detection = {"total_files": 1, "total_words": 100, "warning": None}

    report = generate(
        G,
        {0: ["guide_concept"]},
        {0: 0.5},
        {0: "Docs"},
        [],
        [],
        detection,
        {"input": 0, "output": 0},
        ".",
        stale_semantic_sources=["guide.md"],
    )

    assert "## Semantic Freshness" in report
    assert "guide.md" in report


def test_report_omits_the_section_without_stale_evidence() -> None:
    """Keep a fully current generation's report byte-identical to before."""
    import networkx as nx

    from graphify.report import generate

    G = nx.Graph()
    G.add_node("guide_concept", label="Guide", file_type="concept")
    detection = {"total_files": 1, "total_words": 100, "warning": None}
    args = (
        G,
        {0: ["guide_concept"]},
        {0: 0.5},
        {0: "Docs"},
        [],
        [],
        detection,
        {"input": 0, "output": 0},
        ".",
    )

    assert generate(*args, stale_semantic_sources=[]) == generate(*args)
    assert "## Semantic Freshness" not in generate(*args)
