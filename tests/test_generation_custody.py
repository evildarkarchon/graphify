"""Production-operation tests for exclusive Graph-generation publication custody."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import get_args


def test_corpus_graph_operation_contract_is_closed() -> None:
    """Expose only the agreed request, completion, and terminal outcome variants."""
    from graphify.generation import (
        AlreadyCurrent,
        Cancelled,
        CodeUpdateRequest,
        Completion,
        CorpusStateAdvanced,
        FullExtractionRequest,
        GenerationPublished,
        OperationFailed,
        PublicationRefused,
        Queued,
        ReclusteringRequest,
        ReturnWhenQueued,
        TerminalOutcome,
        WaitUntilCovered,
    )

    assert FullExtractionRequest is not CodeUpdateRequest
    assert CodeUpdateRequest is not ReclusteringRequest
    assert set(get_args(Completion)) == {WaitUntilCovered, ReturnWhenQueued}
    assert set(get_args(TerminalOutcome)) == {
        GenerationPublished,
        CorpusStateAdvanced,
        AlreadyCurrent,
        Queued,
        PublicationRefused,
        OperationFailed,
        Cancelled,
    }


def test_full_extraction_publishes_canonical_state_for_a_real_corpus(tmp_path) -> None:
    """Publish graph and Corpus metadata together through the production owner."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._publication import (
        _GraphData,
        _ManifestUpdate,
        _Publication,
    )

    source = tmp_path / "a.py"
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    graph = {
        "directed": True,
        "multigraph": False,
        "nodes": [
            {
                "id": "a_answer",
                "label": "answer",
                "source_file": "a.py",
                "file_type": "code",
            }
        ],
        "links": [],
    }
    publication = _Publication(
        graph=_GraphData(graph, force=True),
        analysis={"communities": {"0": ["a_answer"]}},
        build_config={"excludes": ["vendor/"], "gitignore": False},
        root_marker=str(tmp_path),
        semantic_marker={"output_tokens": 12},
        manifest=_ManifestUpdate(
            files={"code": [str(source)]},
            kind="both",
            root=tmp_path,
            scan_corpus={str(source)},
        ),
        protect_previous=True,
    )

    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=publication,
    )

    assert isinstance(outcome, GenerationPublished)
    assert json.loads((output / "graph.json").read_text(encoding="utf-8")) == graph
    assert json.loads((output / ".graphify_analysis.json").read_text(encoding="utf-8")) == {
        "communities": {"0": ["a_answer"]}
    }
    assert json.loads((output / ".graphify_build.json").read_text(encoding="utf-8")) == {
        "excludes": ["vendor/"],
        "gitignore": False,
    }
    assert (output / ".graphify_root").read_text(encoding="utf-8") == str(tmp_path)
    assert json.loads((output / ".graphify_semantic_marker").read_text(encoding="utf-8")) == {
        "output_tokens": 12
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"a.py"}


def test_code_update_and_reclustering_share_the_corpus_owner(tmp_path) -> None:
    """Publish both operation families against one real Corpus filesystem."""
    from graphify.generation import (
        CodeUpdateRequest,
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _GraphData, _Publication

    source = tmp_path / "service.py"
    source.write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
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

    update = owner.code_update(
        CodeUpdateRequest((source,)),
        _publication=_Publication(
            graph=_GraphData(graph, force=True),
            root_marker=str(tmp_path),
        ),
    )
    recluster = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            report="# Service graph\n",
            labels={"0": "Service"},
        ),
    )

    assert isinstance(update, GenerationPublished)
    assert isinstance(recluster, CorpusStateAdvanced)
    assert json.loads((output / "graph.json").read_text(encoding="utf-8")) == graph
    assert (output / "GRAPH_REPORT.md").read_text(encoding="utf-8") == ("# Service graph\n")
    assert json.loads((output / ".graphify_labels.json").read_text(encoding="utf-8")) == {
        "0": "Service"
    }


def test_owner_preserves_a_legacy_runbook_sidecar_location(tmp_path) -> None:
    """Keep a stable legacy sidecar path without returning write custody."""
    from graphify.generation import Corpus, CorpusGraph, CorpusStateAdvanced
    from graphify.generation import FullExtractionRequest
    from graphify.generation._publication import (
        _CanonicalArtifact,
        _Publication,
    )

    source_root = tmp_path / "source"
    invocation_root = tmp_path / "invocation"
    source_root.mkdir()
    owner = CorpusGraph(
        Corpus(root=source_root, output=invocation_root / "graphify-out")
    )
    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            analysis={"communities": {}},
            artifact_paths={
                _CanonicalArtifact.ANALYSIS: Path(".graphify_analysis.json"),
            },
        ),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    assert json.loads(
        (invocation_root / ".graphify_analysis.json").read_text(encoding="utf-8")
    ) == {"communities": {}}
    assert not (source_root / ".graphify_analysis.json").exists()
    assert not (invocation_root / "graphify-out" / ".graphify_analysis.json").exists()


def test_owner_freezes_relative_corpus_identity_across_chdir(tmp_path, monkeypatch) -> None:
    """Keep a bound Corpus stable when an adapter later changes process CWD."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        FullExtractionRequest,
    )
    from graphify.generation._publication import _Publication

    source = tmp_path / "source"
    source.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    owner = CorpusGraph(Corpus(root=Path("source"), output=Path("graphify-out")))

    monkeypatch.chdir(other)
    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(analysis={"communities": {}}),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    assert owner.corpus.root == source
    assert owner.corpus.output == tmp_path / "graphify-out"
    assert (tmp_path / "graphify-out" / ".graphify_analysis.json").exists()
    assert not (other / "graphify-out").exists()


def test_generation_owner_has_no_adapter_dependency() -> None:
    """Keep the owner independent of CLI, watcher, and hook adapters."""
    package = Path(__file__).parents[1] / "graphify" / "generation"
    forbidden = {"graphify.cli", "graphify.watch", "graphify.hooks"}
    imported: set[str] = set()

    for module_path in package.glob("*.py"):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

    assert forbidden.isdisjoint(imported)


def test_canonical_adapters_have_no_independent_writer() -> None:
    """Keep production adapters and runbook sources on the owner boundary."""
    repository = Path(__file__).parents[1]
    targets = [
        repository / "graphify" / "cli.py",
        repository / "graphify" / "watch.py",
        repository / "tools" / "skillgen" / "fragments" / "core" / "core.md",
        repository / "tools" / "skillgen" / "fragments" / "core" / "aider.md",
        repository / "tools" / "skillgen" / "fragments" / "core" / "devin.md",
        repository
        / "tools"
        / "skillgen"
        / "fragments"
        / "references"
        / "shared"
        / "update.md",
        repository / "tools" / "skillgen" / "fragments" / "shell" / "posix.md",
        repository / "tools" / "skillgen" / "fragments" / "shell" / "powershell.md",
    ]
    forbidden = (
        "write_json_atomic(_current_path",
        "graph_tmp.replace(existing_graph)",
        "to_json(G, communities, 'graphify-out/graph.json')",
        "save_manifest(_manifest_files",
        "Path('graphify-out/GRAPH_REPORT.md').write_text(report)",
        "Path('graphify-out/.graphify_analysis.json').write_text(",
        "Path('graphify-out/.graphify_labels.json').write_text(",
        "> graphify-out/.graphify_root",
        "Out-File -FilePath graphify-out\\.graphify_root",
        "graphify-out/.graphify_semantic.json "
        "graphify-out/.graphify_analysis.json",
        "rm -f .graphify_detect.json .graphify_extract.json .graphify_ast.json "
        ".graphify_semantic.json .graphify_analysis.json",
    )

    for target in targets:
        content = target.read_text(encoding="utf-8")
        assert "CorpusGraph" in content, target
        assert not any(pattern in content for pattern in forbidden), target

    for target in targets[2:5]:
        content = target.read_text(encoding="utf-8")
        assert "CodeUpdateRequest" in content, target
        assert "retire=frozenset" in content, target
        assert ".graphify_code_update').unlink(missing_ok=True)" in content, target


def test_merge_driver_publishes_current_graph_through_corpus_owner(tmp_path) -> None:
    """Exercise the production merge driver against real graph files."""
    base = tmp_path / "base.json"
    current = tmp_path / "graphify-out" / "graph.json"
    other = tmp_path / "other.json"
    current.parent.mkdir()
    for path, node_id in ((base, "base"), (current, "current"), (other, "other")):
        path.write_text(
            json.dumps(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": node_id}],
                    "links": [],
                }
            ),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphify",
            "merge-driver",
            str(base),
            str(current),
            str(other),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    merged = json.loads(current.read_text(encoding="utf-8"))
    # The existing three-way merge algorithm composes current + other; base is
    # conflict context, not an additional graph input.
    assert {node["id"] for node in merged["nodes"]} == {"current", "other"}
