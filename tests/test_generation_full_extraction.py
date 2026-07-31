"""Production-operation tests for successful Full extraction.

These tests drive the real ``CorpusGraph.full_extraction`` operation over a real
temporary Corpus and read the published Graph generation's own artifacts. The
Semantic provider used here is a real implementation of the production provider
seam — it interprets Markdown headings deterministically — so the operation
contract is exercised end to end without an LLM or a network.

One test seeds a source system's evidence through the production Full-extraction
handoff the extraction CLI still uses, because no source system this suite can
reach offline produces evidence keyed to a system address. It is noted where it
happens.
"""

import json
import os
import shutil
from pathlib import Path

import pytest


class _HeadingInterpreter:
    """Interpret Markdown headings into concept nodes.

    A genuine :class:`graphify.generation.SemanticProvider`: it reads the sources
    it is given and answers with the evidence it derived from them. ``refuse``
    names Corpus-relative sources it reports as *not* completely interpreted, by
    leaving them out of its answer — the same way a real provider reports a chunk
    that failed or truncated.

    A refused source still had a fragment extracted before it was cut short, and
    that fragment is written to this provider's own cache under the Corpus output
    — which is exactly where a partial result is allowed to live, and nowhere
    else.
    """

    def __init__(self, *, refuse: frozenset[str] | set[str] = frozenset()) -> None:
        """Interpret every requested source except those named in ``refuse``."""
        self.refuse = frozenset(refuse)
        self.requested: list[str] = []

    def interpret(self, request):
        """Return concept evidence for every requested source it could read."""
        from graphify.generation import SemanticInterpretation, SourceEvidence

        interpreted: dict[str, SourceEvidence] = {}
        for source in request.sources:
            identity = Path(os.path.relpath(source, request.root)).as_posix()
            self.requested.append(identity)
            nodes = tuple(
                {
                    "id": f"{identity}::{heading}",
                    "label": heading,
                    "file_type": "concept",
                }
                for heading in [
                    line[2:].strip()
                    for line in Path(source).read_text(encoding="utf-8").splitlines()
                    if line.startswith("# ")
                ]
            )
            if identity in self.refuse:
                self._cache_fragment(request, identity, nodes)
                continue
            interpreted[identity] = SourceEvidence(nodes=nodes)
        return SemanticInterpretation(interpreted=interpreted)

    @staticmethod
    def fragment_path(output: Path, identity: str) -> Path:
        """Return where this provider caches the fragment of a cut-short source."""
        return output / "cache" / "fragments" / f"{identity.replace('/', '_')}.json"

    def _cache_fragment(self, request, identity: str, nodes: tuple) -> None:
        """Keep what was extracted before the interpretation was cut short."""
        path = self.fragment_path(request.output, identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": list(nodes)}), encoding="utf-8")


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


def _has_record(output: Path, source: str, interpretation: str) -> bool:
    """Return whether the active ledger carries one contribution."""
    return any(
        record["source"] == source and record["interpretation"] == interpretation
        for record in _ledger_records(output)
    )


def _node_ids(output: Path) -> set[str]:
    """Return the node identities in the published materialized graph."""
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    return {node["id"] for node in graph["nodes"]}


def _manifest(output: Path) -> dict:
    """Return the published manifest keyed by its portable relative paths."""
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _corpus(tmp_path: Path):
    """Return an owner over a small mixed Corpus, with its output directory."""
    from graphify.generation import Corpus, CorpusGraph

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    return CorpusGraph(Corpus(root=tmp_path, output=output)), output


# --- every requested evidence source completes before commit ----------------


def test_full_extraction_publishes_structural_and_semantic_evidence_together(
    tmp_path,
) -> None:
    """Commit one generation carrying every requested source's evidence."""
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    provider = _HeadingInterpreter()

    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=provider),))
    )

    assert isinstance(outcome, GenerationPublished)
    assert provider.requested == ["guide.md"]
    assert _has_record(output, "service.py", "structural")
    assert _record_for(output, "guide.md", "semantic")["nodes"] == [
        {
            "id": "guide.md::Guide",
            "label": "Guide",
            "file_type": "concept",
            "source_file": "guide.md",
        }
    ]
    # One materialized graph, describing one moment of the Corpus.
    ids = _node_ids(output)
    assert "guide.md::Guide" in ids
    structural = _record_for(output, "service.py", "structural")
    assert structural["nodes"]
    assert {node["id"] for node in structural["nodes"]} <= ids


def test_full_extraction_waits_for_a_requested_cargo_workspace(tmp_path) -> None:
    """Include Cargo evidence in the same commit as the filesystem's."""
    try:
        import tomllib  # noqa: F401
    except ModuleNotFoundError:  # Python 3.10 reads TOML through tomli instead.
        pytest.importorskip("tomli", reason="Cargo introspection needs a TOML reader")
    from graphify.generation import (
        CargoSource,
        FullExtractionRequest,
        GenerationPublished,
    )

    owner, output = _corpus(tmp_path)
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "app"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    outcome = owner.full_extraction(FullExtractionRequest(sources=(CargoSource(),)))

    assert isinstance(outcome, GenerationPublished)
    assert "crate:app" in _node_ids(output)
    assert _has_record(output, "Cargo.toml", "structural")


def test_a_requested_source_that_cannot_complete_commits_nothing(tmp_path) -> None:
    """Preserve the active generation when a requested source never answers."""
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        OperationFailed,
        PostgresSource,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    published = (output / "graph.json").read_bytes()

    # A DSN the driver cannot even parse, so the source fails immediately rather
    # than after a connect timeout. Whether the driver is installed or not, this
    # requested source cannot complete.
    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(PostgresSource(dsn="graphify-unreachable"),))
    )

    assert isinstance(outcome, OperationFailed)
    assert (output / "graph.json").read_bytes() == published


def test_a_source_systems_evidence_survives_a_later_code_update(tmp_path) -> None:
    """Keep database evidence a filesystem scan can say nothing about.

    Discovery is authoritative about files. A schema is not a file, so its
    absence from a scan is not evidence it went away — only a Full extraction
    that asked that system again may replace what it said.

    The schema is seeded through the production Full-extraction handoff the
    extraction CLI still uses, because reaching a real PostgreSQL server is not
    something this suite can do offline. What is under test is the reconciliation
    that follows, which runs entirely through the owned operations.
    """
    from graphify.generation import (
        CodeUpdateRequest,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    owner, output = _corpus(tmp_path)
    schema = "postgresql://db.example/orders"
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=schema,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(
                        {
                            "id": "table:orders",
                            "label": "orders",
                            "source_file": schema,
                        },
                    ),
                ),
            )
        ),
    )
    assert isinstance(seeded, GenerationPublished)

    owner.code_update(CodeUpdateRequest())

    assert _has_record(output, schema, "structural")
    assert "table:orders" in _node_ids(output)


# --- source-atomic replacement ----------------------------------------------


def test_a_reinterpreted_source_replaces_its_prior_contribution_atomically(
    tmp_path,
) -> None:
    """Replace one source's evidence completely without touching the others."""
    from graphify.generation import FullExtractionRequest, SemanticSource

    owner, output = _corpus(tmp_path)
    (tmp_path / "design.md").write_text("# Design\n", encoding="utf-8")
    owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert "guide.md::Guide" in _node_ids(output)

    (tmp_path / "guide.md").write_text("# Rewritten\n", encoding="utf-8")
    owner.full_extraction(
        FullExtractionRequest(
            sources=(SemanticSource(provider=_HeadingInterpreter()),),
            # The replacement is smaller than the evidence it retires, which is
            # accounted for by the source it replaced rather than by an override.
            force=False,
        )
    )

    record = _record_for(output, "guide.md", "semantic")
    assert [node["id"] for node in record["nodes"]] == ["guide.md::Rewritten"]
    ids = _node_ids(output)
    assert "guide.md::Guide" not in ids
    assert "guide.md::Rewritten" in ids
    # The untouched source kept exactly the evidence it had.
    assert "design.md::Design" in ids


# --- failed interpretation --------------------------------------------------


def test_a_failed_interpretation_keeps_its_prior_evidence_as_stale(tmp_path) -> None:
    """Retain the last complete interpretation, disclosed as stale and pending."""
    from graphify.generation import (
        FullExtractionRequest,
        SemanticSource,
        stale_semantic_sources,
    )

    owner, output = _corpus(tmp_path)
    (tmp_path / "design.md").write_text("# Design\n", encoding="utf-8")
    owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert stale_semantic_sources(output) == ()

    (tmp_path / "guide.md").write_text("# Rewritten\n", encoding="utf-8")
    owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        )
    )

    record = _record_for(output, "guide.md", "semantic")
    assert record["stale"] is True
    assert [node["id"] for node in record["nodes"]] == ["guide.md::Guide"]
    assert "guide.md::Guide" in _node_ids(output)
    assert stale_semantic_sources(output) == ("guide.md",)
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"
    assert _manifest(output)["guide.md"]["semantic_hash"] == ""
    # The source that did interpret is current, not dragged down with it.
    assert _record_for(output, "design.md", "semantic").get("stale", False) is False


def test_a_never_interpreted_source_that_fails_contributes_nothing(tmp_path) -> None:
    """Leave a source absent rather than inventing evidence for it."""
    from graphify.generation import (
        FullExtractionRequest,
        SemanticSource,
        stale_semantic_sources,
    )

    owner, output = _corpus(tmp_path)

    owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        )
    )

    assert not _has_record(output, "guide.md", "semantic")
    assert "guide.md::Guide" not in _node_ids(output)
    # Nothing is stale, because nothing was ever interpreted — but the source is
    # still awaiting interpretation.
    assert stale_semantic_sources(output) == ()
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"
    # No interpretation ever succeeded for it, so the manifest either has no row
    # at all or one that vouches for nothing.
    assert not _manifest(output).get("guide.md", {}).get("semantic_hash")


def test_a_partial_fragment_stays_in_the_providers_cache(tmp_path) -> None:
    """Keep an incomplete interpretation out of the published generation.

    The provider extracted something for the source before it was cut short. That
    fragment belongs in its cache, so a retry can use it — never in the ledger or
    the graph, where it would read as the source's complete evidence.
    """
    from graphify.generation import FullExtractionRequest, SemanticSource

    owner, output = _corpus(tmp_path)
    provider = _HeadingInterpreter(refuse={"guide.md"})

    owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=provider),))
    )

    fragment = provider.fragment_path(output, "guide.md")
    assert json.loads(fragment.read_text(encoding="utf-8"))["nodes"], (
        "the provider should have cached what it did extract"
    )
    assert not _has_record(output, "guide.md", "semantic")
    assert "guide.md::Guide" not in _node_ids(output)


def test_a_failed_google_workspace_export_leaves_the_corpus_pending(tmp_path) -> None:
    """Treat a shortcut that could not be exported as outstanding work.

    Requesting Google Workspace admits shortcuts as Corpus documents by exporting
    them to Markdown sidecars. An export that fails leaves no sidecar, so the
    Corpus is missing a source it was asked to reconcile — and must say so rather
    than publishing as though the shortcut were not there.
    """
    if shutil.which("gws") is not None:
        pytest.skip("the Google Workspace exporter is installed; this would go online")
    from graphify.generation import (
        FullExtractionRequest,
        GoogleWorkspaceSource,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert not (output / "needs_update").exists()
    (tmp_path / "notes.gdoc").write_text(
        json.dumps({"url": "https://docs.google.com/document/d/abc123/edit"}),
        encoding="utf-8",
    )

    owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter()),
                GoogleWorkspaceSource(),
            )
        )
    )

    assert (output / "needs_update").read_text(encoding="utf-8") == "1"
    # The rest of the Corpus is still published, and the evidence it already had
    # is intact — a failed export is not a reason to lose a document.
    assert "guide.md::Guide" in _node_ids(output)


# --- clearing pending state -------------------------------------------------


def test_only_a_complete_interpretation_clears_pending_state(tmp_path) -> None:
    """Retire the pending marker once every live source is interpreted again."""
    from graphify.generation import (
        FullExtractionRequest,
        SemanticSource,
        stale_semantic_sources,
    )

    owner, output = _corpus(tmp_path)
    owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        )
    )
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"

    owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )

    assert not (output / "needs_update").exists()
    assert stale_semantic_sources(output) == ()
    assert _manifest(output)["guide.md"]["semantic_hash"]


def test_a_full_extraction_without_interpretation_never_clears_pending_state(
    tmp_path,
) -> None:
    """Leave pending state alone when no Semantic evidence was requested."""
    from graphify.generation import FullExtractionRequest, SemanticSource

    owner, output = _corpus(tmp_path)
    owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        )
    )
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"

    owner.full_extraction(FullExtractionRequest())

    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


# --- build policy -----------------------------------------------------------


def test_full_extraction_preserves_the_active_build_policy(tmp_path) -> None:
    """Keep honoring a recorded policy no request asked to change."""
    from graphify.generation import FullExtractionRequest

    owner, output = _corpus(tmp_path)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "bundled.py").write_text("class Bundled:\n    pass\n", encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    policy = json.dumps({"excludes": ["vendor/**"], "gitignore": True})
    (output / ".graphify_build.json").write_text(policy, encoding="utf-8")

    owner.full_extraction(FullExtractionRequest())

    assert not _has_record(output, "vendor/bundled.py", "structural")
    assert (output / ".graphify_build.json").read_text(encoding="utf-8") == policy


def test_full_extraction_replaces_the_build_policy_only_when_asked(tmp_path) -> None:
    """Record and apply a new Corpus shape when replacement is explicit."""
    from graphify.generation import (
        BuildPolicy,
        FullExtractionRequest,
        ReplaceBuildPolicy,
    )

    owner, output = _corpus(tmp_path)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "bundled.py").write_text("class Bundled:\n    pass\n", encoding="utf-8")

    owner.full_extraction(
        FullExtractionRequest(
            build_policy=ReplaceBuildPolicy(
                BuildPolicy(excludes=("vendor/**",), gitignore=False)
            )
        )
    )

    assert not _has_record(output, "vendor/bundled.py", "structural")
    recorded = json.loads((output / ".graphify_build.json").read_text(encoding="utf-8"))
    assert recorded == {"excludes": ["vendor/**"], "gitignore": False}


def test_full_extraction_clears_the_build_policy_when_asked(tmp_path) -> None:
    """Return the Corpus to the documented defaults on an explicit clear."""
    from graphify.generation import ClearBuildPolicy, FullExtractionRequest

    owner, output = _corpus(tmp_path)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "bundled.py").write_text("class Bundled:\n    pass\n", encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".graphify_build.json").write_text(
        json.dumps({"excludes": ["vendor/**"], "gitignore": True}), encoding="utf-8"
    )

    owner.full_extraction(FullExtractionRequest(build_policy=ClearBuildPolicy()))

    assert _has_record(output, "vendor/bundled.py", "structural")
    assert not (output / ".graphify_build.json").exists()
