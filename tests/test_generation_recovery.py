"""Production-operation tests for recoverable Graph-generation publication."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


def _seed_protected_generation(root: Path, output: Path) -> dict[str, bytes]:
    """Publish one real protected prior generation and return its stable bytes."""
    from graphify.generation import (
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

    source = root / "prior.py"
    source.write_text("PRIOR = True\n", encoding="utf-8")
    outcome = CorpusGraph(Corpus(root=root, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(
                        {
                            "id": "prior",
                            "label": "Prior",
                            "source_file": str(source),
                            "file_type": "code",
                        },
                    ),
                ),
            ),
            report="# Prior generation\n",
            analysis={"generation": "prior"},
            labels={"0": "Prior"},
            build_config={"generation": "prior"},
            root_marker=str(root),
            semantic_marker={"output_tokens": 12},
        ),
    )
    assert isinstance(outcome, GenerationPublished)
    return {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }


def _start_candidate_publication(root: Path, output: Path) -> subprocess.Popen[str]:
    """Start a real production publication with a long observable promotion."""
    script = root / "publish_candidate.py"
    script.write_text(
        """
import sys
from pathlib import Path

from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
from graphify.generation._contributions import _InterpretationKind, _SourceContribution
from graphify.generation._publication import _Publication

root = Path(sys.argv[1])
output = Path(sys.argv[2])
source = root / "candidate.py"
source.write_text("CANDIDATE = True\\n", encoding="utf-8")
report = "# Candidate generation\\n" + ("x" * (32 * 1024 * 1024))
publication = _Publication(
    contributions=(
        _SourceContribution(
            source=source,
            interpretation=_InterpretationKind.STRUCTURAL,
            nodes=(
                {
                    "id": "candidate",
                    "label": "Candidate",
                    "source_file": str(source),
                    "file_type": "code",
                },
            ),
        ),
    ),
    report=report,
    analysis={"generation": "candidate"},
    labels={"0": "Candidate"},
    build_config={"generation": "candidate"},
    root_marker=str(root),
    semantic_marker={"output_tokens": 24},
    protect_previous=True,
)
outcome = CorpusGraph(Corpus(root=root, output=output)).full_extraction(
    FullExtractionRequest(),
    _publication=publication,
)
print(type(outcome).__name__, flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    repository = Path(__file__).parents[1]
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository)
        if not prior_pythonpath
        else str(repository) + os.pathsep + prior_pythonpath
    )
    return subprocess.Popen(
        [sys.executable, str(script), str(root), str(output)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _kill_during_promotion(
    process: subprocess.Popen[str],
    output: Path,
) -> dict:
    """Kill ``process`` during the durable production promotion phase."""
    journal_path = output / ".graphify_publication.json"
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    "publication exited before termination phase\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                if journal["phase"] == "promoting":
                    process.kill()
                    process.communicate(timeout=20)
                    assert process.returncode != 0
                    assert journal_path.exists()
                    return journal
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, KeyError):
                pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)
    raise AssertionError("timed out waiting for observable publication promotion")


def test_complete_candidate_is_published_with_a_digest_backed_completion_marker(
    tmp_path,
) -> None:
    """Write the completion marker only for the fully validated candidate."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._publication import _GraphData, _Publication

    source = tmp_path / "service.py"
    source.write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    graph = {
        "directed": False,
        "multigraph": False,
        "nodes": [
            {
                "id": "service",
                "label": "Service",
                "source_file": "service.py",
                "file_type": "code",
            }
        ],
        "links": [],
    }

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(graph=_GraphData(graph, force=True)),
    )

    assert isinstance(outcome, GenerationPublished)
    marker = json.loads(
        (output / ".graphify_generation_complete").read_text(encoding="utf-8")
    )
    assert marker["schema"] == "graphify-generation-completion"
    assert marker["version"] == 1
    assert set(marker["artifacts"]) == {
        ".graphify_contributions.jsonl",
        "graph.json",
    }
    for artifact, metadata in marker["artifacts"].items():
        published = output / artifact
        assert metadata == {
            "location": "output",
            "sha256": hashlib.sha256(published.read_bytes()).hexdigest(),
            "size": published.stat().st_size,
        }
    assert not (output / ".graphify_publication").exists()
    assert not (output / ".graphify_publication.json").exists()


def test_protected_backup_failure_refuses_without_changing_the_active_generation(
    tmp_path,
) -> None:
    """Fail closed when the protected prior generation cannot be backed up."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
    )
    from graphify.generation._publication import _GraphData, _Publication

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    prior_graph = {
        "directed": False,
        "multigraph": False,
        "nodes": [{"id": "prior", "label": "Prior"}],
        "links": [],
    }
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(prior_graph, force=True),
            report="# Prior\n",
            semantic_marker={"output_tokens": 12},
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    stable_before = {
        path.name: path.read_bytes()
        for path in output.iterdir()
        if path.is_file()
    }
    blocked_backup = output / date.today().isoformat()
    blocked_backup.write_text("not a directory", encoding="utf-8")
    candidate_graph = {
        "directed": False,
        "multigraph": False,
        "nodes": [
            {"id": "candidate", "label": "Candidate"},
            {"id": "second", "label": "Second"},
        ],
        "links": [],
    }

    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(candidate_graph),
            report="# Candidate\n",
            protect_previous=True,
        ),
    )

    assert isinstance(outcome, PublicationRefused)
    assert "protected backup failed" in outcome.reason
    assert {
        name: (output / name).read_bytes()
        for name in stable_before
    } == stable_before
    assert blocked_backup.read_text(encoding="utf-8") == "not a directory"
    assert not (output / ".graphify_publication").exists()
    assert not (output / ".graphify_publication.json").exists()


def test_explicit_no_backup_policy_allows_protected_publication(
    tmp_path,
    monkeypatch,
) -> None:
    """Honor only the explicit policy override while retaining journal rollback."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._publication import _GraphData, _Publication

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "prior"}],
                    "links": [],
                },
                force=True,
            ),
            semantic_marker={"output_tokens": 12},
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    blocked_backup = output / date.today().isoformat()
    blocked_backup.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("GRAPHIFY_NO_BACKUP", "1")

    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "candidate"}, {"id": "second"}],
                    "links": [],
                }
            ),
            protect_previous=True,
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    published = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert {node["id"] for node in published["nodes"]} == {
        "candidate",
        "second",
    }
    assert blocked_backup.read_text(encoding="utf-8") == "not a directory"


def test_valid_external_label_edit_is_admitted_as_a_legacy_prior(tmp_path) -> None:
    """Keep current curated-label compatibility without trusting a stale marker."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        FullExtractionRequest,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _GraphData, _Publication

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "service"}],
                    "links": [],
                },
                force=True,
            ),
            labels={"0": "Community 0"},
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    labels_path = output / ".graphify_labels.json"
    labels_path.write_text('{"0":"Curated Service"}', encoding="utf-8")

    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            report="# Curated service\n",
            labels={"0": "Curated Service"},
        ),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    assert outcome.changed_artifacts == (
        "GRAPH_REPORT.md",
        ".graphify_labels.json",
    )
    assert json.loads(labels_path.read_text(encoding="utf-8")) == {
        "0": "Curated Service"
    }


def test_code_update_preserves_authoritative_ledger_after_label_curation(
    tmp_path,
) -> None:
    """Refresh label custody without degrading authoritative Source identity."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    source = tmp_path / "service.py"
    source.write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(
                        {
                            "id": "service",
                            "label": "Service",
                            "source_file": str(source),
                            "file_type": "code",
                        },
                    ),
                ),
            ),
            labels={"0": "Community 0"},
            root_marker=str(tmp_path),
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    ledger_path = output / ".graphify_contributions.jsonl"
    authoritative_ledger = ledger_path.read_bytes()
    labels_path = output / ".graphify_labels.json"
    labels_path.write_text('{"0":"Curated Service"}', encoding="utf-8")

    outcome = owner.code_update(
        CodeUpdateRequest(),
        _publication=_Publication(root_marker=str(tmp_path)),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    assert ledger_path.read_bytes() == authoritative_ledger
    marker = json.loads(
        (output / ".graphify_generation_complete").read_text(encoding="utf-8")
    )
    assert marker["artifacts"][".graphify_labels.json"]["sha256"] == hashlib.sha256(
        labels_path.read_bytes()
    ).hexdigest()


def test_external_graph_edit_is_not_admitted_as_curated_legacy_state(
    tmp_path,
) -> None:
    """Refuse a stale marker when Graph topology contradicts its ledger."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _GraphData, _Publication

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "original"}],
                    "links": [],
                },
                force=True,
            ),
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    (output / "graph.json").write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "nodes": [{"id": "tampered"}],
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(report="# Must not publish\n"),
    )

    assert isinstance(outcome, PublicationRefused)
    assert "did not validate" in outcome.reason


def test_active_completion_placements_survive_sidecar_only_publication(
    tmp_path,
) -> None:
    """Keep compatibility-root artifacts tracked until explicitly migrated."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        FullExtractionRequest,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._publication import (
        _CanonicalArtifact,
        _GraphData,
        _Publication,
    )

    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    seeded = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "service"}],
                    "links": [],
                },
                force=True,
            ),
            analysis={"generation": "seed"},
            artifact_paths={
                _CanonicalArtifact.ANALYSIS: Path(".graphify_analysis.json")
            },
        ),
    )
    assert isinstance(seeded, GenerationPublished)
    compatibility_analysis = tmp_path / ".graphify_analysis.json"
    assert compatibility_analysis.exists()

    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(report="# Refreshed report\n"),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    assert json.loads(compatibility_analysis.read_text(encoding="utf-8")) == {
        "generation": "seed"
    }
    assert not (output / ".graphify_analysis.json").exists()
    marker = json.loads(
        (output / ".graphify_generation_complete").read_text(encoding="utf-8")
    )
    assert marker["artifacts"][".graphify_analysis.json"]["location"] == (
        "compatibility-root"
    )


def test_interrupted_publication_rolls_an_intact_candidate_forward(tmp_path) -> None:
    """Recover an authorized candidate after real subprocess termination."""
    from graphify.generation import (
        AlreadyCurrent,
        Corpus,
        CorpusGraph,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    _seed_protected_generation(tmp_path, output)
    process = _start_candidate_publication(tmp_path, output)
    _kill_during_promotion(process, output)
    candidate_marker = (
        output
        / ".graphify_publication"
        / "candidate"
        / ".graphify_generation_complete"
    ).read_bytes()

    recovered = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(),
    )

    assert isinstance(recovered, AlreadyCurrent)
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert [node["id"] for node in graph["nodes"]] == ["candidate"]
    assert (output / "GRAPH_REPORT.md").read_text(encoding="utf-8").startswith(
        "# Candidate generation\n"
    )
    assert json.loads(
        (output / ".graphify_analysis.json").read_text(encoding="utf-8")
    ) == {"generation": "candidate"}
    assert (output / ".graphify_generation_complete").read_bytes() == candidate_marker
    assert not (output / ".graphify_publication").exists()
    assert not (output / ".graphify_publication.json").exists()


def test_interrupted_publication_restores_prior_when_candidate_is_invalid(
    tmp_path,
) -> None:
    """Restore the protected prior generation when staged evidence is corrupt."""
    from graphify.generation import (
        AlreadyCurrent,
        Corpus,
        CorpusGraph,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    prior = _seed_protected_generation(tmp_path, output)
    process = _start_candidate_publication(tmp_path, output)
    _kill_during_promotion(process, output)
    candidate_ledger = (
        output
        / ".graphify_publication"
        / "candidate"
        / ".graphify_contributions.jsonl"
    )
    candidate_ledger.write_text('{"corrupt":true}\n', encoding="utf-8")

    recovered = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(),
    )

    assert isinstance(recovered, AlreadyCurrent)
    assert {
        name: (output / name).read_bytes()
        for name in prior
    } == prior
    assert not (output / ".graphify_publication").exists()
    assert not (output / ".graphify_publication.json").exists()
    assert not list(output.glob(".GRAPH_REPORT.md.*.tmp"))


def test_abort_does_not_remove_another_transaction_journal(tmp_path) -> None:
    """Leave durable state untouched when this transaction never owned it."""
    from graphify.generation import Corpus
    from graphify.generation._layout import _PublicationLayout
    from graphify.generation._transaction import _PublicationTransaction

    corpus = Corpus(root=tmp_path, output=tmp_path / "graphify-out")
    layout = _PublicationLayout(root=corpus.root, output=corpus.output, overrides={})
    owner = _PublicationTransaction(
        corpus,
        layout,
        operation="full-extraction",
        protect_previous=False,
    )
    owner.begin()
    foreign = _PublicationTransaction(
        corpus,
        layout,
        operation="reclustering",
        protect_previous=False,
    )

    foreign.abort()

    assert (corpus.output / ".graphify_publication.json").exists()
    assert (corpus.output / ".graphify_publication" / "candidate").is_dir()
    owner.abort()


def test_prepare_refuses_graph_not_materialized_from_candidate_contributions(
    tmp_path,
) -> None:
    """Reject a well-formed candidate whose graph contradicts its ledger."""
    from graphify.generation import Corpus, PublicationRefused
    from graphify.generation._layout import _PublicationLayout
    from graphify.generation._publication import _CanonicalArtifact
    from graphify.generation._transaction import _PublicationTransaction

    output = tmp_path / "graphify-out"
    _seed_protected_generation(tmp_path, output)
    corpus = Corpus(root=tmp_path, output=output)
    layout = _PublicationLayout(root=corpus.root, output=corpus.output, overrides={})
    transaction = _PublicationTransaction(
        corpus,
        layout,
        operation="full-extraction",
        protect_previous=False,
    )
    staged = transaction.begin()
    staged.path_for(_CanonicalArtifact.GRAPH).write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "nodes": [{"id": "unrelated"}],
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        refusal = transaction.prepare()
        assert isinstance(refusal, PublicationRefused)
        assert "candidate" in refusal.reason
    finally:
        transaction.abort()
