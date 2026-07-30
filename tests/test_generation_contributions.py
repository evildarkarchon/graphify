"""Production-operation tests for authoritative Source-contribution custody."""

import json
import os

import pytest


def _publish_ordered_contributions(root, *, reverse: bool) -> bytes:
    """Publish equivalent evidence and return the deterministic ledger bytes."""
    from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    source = root / "nested" / "source.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_identity = source
    source_file = str(source)
    if reverse:
        source_identity = r"nested\source.py"
        source_file = r"nested\source.py"
    structural_node = {
        "source_file": source_file,
        "label": "Source",
        "id": "source",
        "file_type": "code",
    }
    semantic_node = {
        "file_type": "concept",
        "id": "source_meaning",
        "label": "Source meaning",
        "source_file": source_file,
    }
    if reverse:
        structural_node = dict(reversed(tuple(structural_node.items())))
        semantic_node = dict(reversed(tuple(semantic_node.items())))
    contributions = [
        _SourceContribution(
            source=source_identity,
            interpretation=_InterpretationKind.STRUCTURAL,
            nodes=(structural_node,),
        ),
        _SourceContribution(
            source=source_identity,
            interpretation=_InterpretationKind.SEMANTIC,
            nodes=(semantic_node,),
        ),
    ]
    if reverse:
        contributions.reverse()
    output = root / "graphify-out"
    CorpusGraph(Corpus(root=root, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(contributions=tuple(contributions)),
    )
    return (output / ".graphify_contributions.jsonl").read_bytes()


def test_full_extraction_persists_pre_dedup_source_contributions(tmp_path) -> None:
    """Retain shared evidence per root-relative source and interpretation kind."""
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

    source_a = tmp_path / "a.py"
    source_b = tmp_path / "b.md"
    source_a.write_text("class Shared:\n    pass\n", encoding="utf-8")
    source_b.write_text("# Shared\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    shared_from_a = {
        "id": "shared",
        "label": "Shared",
        "file_type": "code",
        "source_file": str(source_a),
    }
    shared_from_b = {
        "id": "shared",
        "label": "Shared",
        "file_type": "document",
        "source_file": str(source_b),
    }
    semantic_from_a = {
        "id": "shared",
        "label": "Shared concept",
        "file_type": "concept",
        "source_file": str(source_a),
    }

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source_b,
                    interpretation=_InterpretationKind.SEMANTIC,
                    nodes=(shared_from_b,),
                ),
                _SourceContribution(
                    source=source_a,
                    interpretation=_InterpretationKind.SEMANTIC,
                    nodes=(semantic_from_a,),
                ),
                _SourceContribution(
                    source=source_a,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(shared_from_a,),
                ),
            ),
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    records = [
        json.loads(line)
        for line in (output / ".graphify_contributions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        (record["source"], record["interpretation"])
        for record in records[1:]
    ] == [
        ("a.py", "semantic"),
        ("a.py", "structural"),
        ("b.md", "semantic"),
    ]
    assert [record["nodes"][0]["id"] for record in records[1:]] == [
        "shared",
        "shared",
        "shared",
    ]
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"] == [
        {
            "id": "shared",
            "label": "Shared",
            "file_type": "document",
            "source_file": "b.md",
        }
    ]


def test_replacement_contributions_rematerialize_an_existing_graph(tmp_path) -> None:
    """Keep the derived graph synchronized when authority is replaced alone."""
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

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))

    def _contribution(node_id: str) -> _SourceContribution:
        return _SourceContribution(
            source=source,
            interpretation=_InterpretationKind.STRUCTURAL,
            nodes=(
                {
                    "id": node_id,
                    "label": node_id.title(),
                    "source_file": str(source),
                },
            ),
        )

    owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(contributions=(_contribution("old"),)),
    )

    outcome = owner.code_update(
        CodeUpdateRequest(),
        _publication=_Publication(contributions=(_contribution("new"),)),
    )

    assert isinstance(outcome, GenerationPublished)
    assert outcome.changed_artifacts[:2] == (
        ".graphify_contributions.jsonl",
        "graph.json",
    )
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert [node["id"] for node in graph["nodes"]] == ["new"]


def test_newer_interpretation_wins_duplicate_edge_attributes(tmp_path) -> None:
    """Apply explicit interpretation precedence to last-writer edge deduplication."""
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

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"

    def _contribution(
        interpretation: _InterpretationKind,
        authority: str,
        *,
        provisional: bool = False,
    ) -> _SourceContribution:
        return _SourceContribution(
            source=source,
            interpretation=interpretation,
            nodes=(
                {"id": "source", "source_file": str(source)},
                {"id": "target", "source_file": str(source)},
            ),
            edges=(
                {
                    "source": "source",
                    "target": "target",
                    "relation": "calls",
                    "authority": authority,
                    "source_file": str(source),
                },
            ),
            hyperedges=(
                {
                    "id": "flow",
                    "nodes": ["source", "target"],
                    "authority": authority,
                    "source_file": str(source),
                },
            ),
            provisional=provisional,
        )

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _contribution(
                    _InterpretationKind.LEGACY_ATTRIBUTED,
                    "legacy",
                    provisional=True,
                ),
                _contribution(_InterpretationKind.STRUCTURAL, "structural"),
                _contribution(_InterpretationKind.SEMANTIC, "semantic"),
            ),
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["links"]) == 1
    assert graph["links"][0]["authority"] == "semantic"
    assert len(graph["hyperedges"]) == 1
    assert graph["hyperedges"][0]["authority"] == "semantic"


@pytest.mark.parametrize("graph_kind", ["data", "model"])
def test_multigraph_generation_retains_parallel_contribution_edges(
    tmp_path,
    graph_kind,
) -> None:
    """Keep legitimate parallel evidence when the requested view is a multigraph."""
    import networkx as nx

    from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import (
        _GraphData,
        _GraphModel,
        _Publication,
    )

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    contribution = _SourceContribution(
        source=source,
        interpretation=_InterpretationKind.STRUCTURAL,
        nodes=(
            {"id": "source", "source_file": str(source)},
            {"id": "target", "source_file": str(source)},
        ),
        edges=(
            {
                "source": "source",
                "target": "target",
                "relation": "calls",
                "evidence": "first",
                "source_file": str(source),
            },
            {
                "source": "source",
                "target": "target",
                "relation": "calls",
                "evidence": "second",
                "source_file": str(source),
            },
        ),
        hyperedges=(
            {
                "id": "parallel-flow",
                "nodes": ["source", "target"],
                "source_file": str(source),
            },
        ),
    )
    if graph_kind == "data":
        graph = _GraphData(
            {
                "directed": False,
                "multigraph": True,
                "nodes": [],
                "links": [],
            },
            force=True,
        )
    else:
        prepared = nx.MultiGraph()
        prepared.add_nodes_from(("source", "target"))
        graph = _GraphModel(prepared, communities={}, force=True)

    CorpusGraph(Corpus(root=tmp_path, output=tmp_path / "graphify-out")).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=graph,
            contributions=(contribution,),
        ),
    )

    published = json.loads(
        (tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8")
    )
    assert published["multigraph"] is True
    assert {edge["evidence"] for edge in published["links"]} == {
        "first",
        "second",
    }
    assert [hyperedge["id"] for hyperedge in published["hyperedges"]] == [
        "parallel-flow"
    ]


def test_graph_model_refusal_does_not_replace_authoritative_contributions(
    tmp_path,
) -> None:
    """Preflight model safety before changing required generation state."""
    import networkx as nx

    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        PublicationRefused,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _GraphModel, _Publication

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    initial = _SourceContribution(
        source=source,
        interpretation=_InterpretationKind.STRUCTURAL,
        nodes=(
            {"id": "first", "source_file": str(source)},
            {"id": "second", "source_file": str(source)},
        ),
    )
    owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(contributions=(initial,)),
    )
    ledger_path = output / ".graphify_contributions.jsonl"
    graph_path = output / "graph.json"
    ledger_before = ledger_path.read_bytes()
    graph_before = graph_path.read_bytes()

    replacement = _SourceContribution(
        source=source,
        interpretation=_InterpretationKind.STRUCTURAL,
        nodes=({"id": "replacement", "source_file": str(source)},),
    )
    prepared_graph = nx.Graph()
    prepared_graph.add_node("replacement")

    outcome = owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            graph=_GraphModel(prepared_graph, communities={}),
            contributions=(replacement,),
        ),
    )

    assert isinstance(outcome, PublicationRefused)
    assert ledger_path.read_bytes() == ledger_before
    assert graph_path.read_bytes() == graph_before


def test_contribution_ledger_is_byte_deterministic_and_streamable(tmp_path) -> None:
    """Compare portable bytes and consume one contribution record at a time."""
    from graphify.generation._contributions import _iter_contribution_ledger

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    first_bytes = _publish_ordered_contributions(first, reverse=False)
    second_bytes = _publish_ordered_contributions(second, reverse=True)

    assert first_bytes == second_bytes
    assert b"\r\n" not in first_bytes
    records = _iter_contribution_ledger(
        first / "graphify-out" / ".graphify_contributions.jsonl"
    )
    first_record = next(records)
    assert (first_record.source, first_record.interpretation.value) == (
        "nested/source.py",
        "semantic",
    )
    assert [record.interpretation.value for record in records] == ["structural"]


def test_contribution_identity_preserves_in_root_symlink_spelling(tmp_path) -> None:
    """Keep distinct lexical Corpus sources distinct after containment checks."""
    from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
        _iter_contribution_ledger,
    )
    from graphify.generation._publication import _Publication

    target = tmp_path / "sub" / "target.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")
    alias = tmp_path / "alias.py"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def _contribution(path, node_id: str) -> _SourceContribution:
        return _SourceContribution(
            source=path,
            interpretation=_InterpretationKind.STRUCTURAL,
            nodes=(
                {
                    "id": node_id,
                    "source_file": str(path),
                },
            ),
        )

    output = tmp_path / "graphify-out"
    CorpusGraph(Corpus(root=tmp_path, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _contribution(alias, "alias"),
                _contribution(target, "target"),
            ),
        ),
    )

    records = list(
        _iter_contribution_ledger(output / ".graphify_contributions.jsonl")
    )
    assert [record.source for record in records] == ["alias.py", "sub/target.py"]
    assert [record.nodes[0]["source_file"] for record in records] == [
        "alias.py",
        "sub/target.py",
    ]


def test_foreign_absolute_source_is_not_rebased_under_the_corpus(tmp_path) -> None:
    """Reject an absolute spelling from the other path dialect as non-portable."""
    from graphify.generation._contributions import (
        _legacy_source_identity,
        _relative_source_identity,
    )

    foreign = (
        "/outside/source.py"
        if os.name == "nt"
        else r"C:\outside\source.py"
    )

    with pytest.raises(ValueError, match="outside the Corpus root"):
        _relative_source_identity(foreign, tmp_path)
    assert _legacy_source_identity(foreign, tmp_path) == "@legacy/unattributed"


def test_reclustering_adopts_valid_legacy_graph_without_inventing_sidecars(
    tmp_path,
) -> None:
    """Adopt attributed and unattributed legacy evidence provisionally."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        CorpusStateAdvanced,
        ReclusteringRequest,
    )
    from graphify.generation._contributions import _iter_contribution_ledger
    from graphify.generation._publication import _Publication

    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text("class Service:\n    pass\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    output.mkdir()
    legacy_graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "service",
                "label": "Service",
                "file_type": "code",
                "source_file": "src/service.py",
            },
            {
                "id": "external",
                "label": "External",
                "file_type": "concept",
            },
        ],
        "links": [
            {
                "source": "service",
                "target": "external",
                "relation": "calls",
                "source_file": "src/service.py",
            }
        ],
    }
    graph_path = output / "graph.json"
    graph_path.write_text(json.dumps(legacy_graph, indent=2), encoding="utf-8")
    original_graph_bytes = graph_path.read_bytes()

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(),
    )

    assert isinstance(outcome, CorpusStateAdvanced)
    records = list(
        _iter_contribution_ledger(output / ".graphify_contributions.jsonl")
    )
    assert [
        (record.source, record.interpretation.value, record.provisional)
        for record in records
    ] == [
        ("@legacy/unattributed", "legacy-unattributed", True),
        ("src/service.py", "legacy-attributed", True),
    ]
    assert [node["id"] for node in records[0].nodes] == ["external"]
    assert [node["id"] for node in records[1].nodes] == ["service"]
    assert [edge["relation"] for edge in records[1].edges] == ["calls"]
    assert graph_path.read_bytes() == original_graph_bytes
    assert {path.name for path in output.iterdir()} == {
        ".graphify_contributions.jsonl",
        "graph.json",
    }


def test_reclustering_refuses_corrupt_legacy_graph_without_writes(tmp_path) -> None:
    """Fail closed when pre-ledger canonical output cannot be validated."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        PublicationRefused,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    output.mkdir()
    graph_path = output / "graph.json"
    corrupt_bytes = b'{"nodes":[{"id":"unfinished"}'
    graph_path.write_bytes(corrupt_bytes)

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(),
    )

    assert isinstance(outcome, PublicationRefused)
    assert graph_path.read_bytes() == corrupt_bytes
    assert {path.name for path in output.iterdir()} == {"graph.json"}


def test_compatibility_graph_is_admitted_before_its_generation_is_written(
    tmp_path,
) -> None:
    """Give every compatibility generation a required provisional ledger."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
    )
    from graphify.generation._contributions import _iter_contribution_ledger
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
            },
            {"id": "external", "label": "External"},
        ],
        "links": [],
    }

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(graph=_GraphData(graph, force=True)),
    )

    assert isinstance(outcome, GenerationPublished)
    assert outcome.changed_artifacts[:2] == (
        ".graphify_contributions.jsonl",
        "graph.json",
    )
    records = list(
        _iter_contribution_ledger(output / ".graphify_contributions.jsonl")
    )
    assert [
        (record.source, record.interpretation.value, record.provisional)
        for record in records
    ] == [
        ("@legacy/unattributed", "legacy-unattributed", True),
        ("service.py", "legacy-attributed", True),
    ]
    published = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    assert {
        key: value for key, value in published.items() if key != "nodes"
    } == {
        key: value for key, value in graph.items() if key != "nodes"
    }
    assert {node["id"]: node for node in published["nodes"]} == {
        node["id"]: node for node in graph["nodes"]
    }


def test_corrupt_authoritative_ledger_is_refused_without_graph_fallback(
    tmp_path,
) -> None:
    """Never recover required generation state from its materialized view."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        PublicationRefused,
        ReclusteringRequest,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _GraphData, _Publication

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source,
                    interpretation=_InterpretationKind.STRUCTURAL,
                    nodes=(
                        {
                            "id": "source",
                            "label": "Source",
                            "source_file": str(source),
                        },
                    ),
                ),
            ),
        ),
    )
    graph_path = output / "graph.json"
    graph_before = graph_path.read_bytes()
    contribution_path = output / ".graphify_contributions.jsonl"
    corrupt_ledger = b'{"schema":"graphify-source-contributions","version":1}\n{bad'
    contribution_path.write_bytes(corrupt_ledger)

    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "fallback"}],
                    "links": [],
                },
                force=True,
            ),
        ),
    )

    assert isinstance(outcome, PublicationRefused)
    assert contribution_path.read_bytes() == corrupt_ledger
    assert graph_path.read_bytes() == graph_before


@pytest.mark.parametrize(
    ("bucket", "invalid_item"),
    [
        ("nodes", {}),
        ("edges", {"source": "source"}),
        ("hyperedges", {"id": "flow"}),
        ("hyperedges", {"nodes": ["source"]}),
    ],
)
def test_structurally_invalid_authoritative_ledger_is_refused(
    tmp_path,
    bucket,
    invalid_item,
) -> None:
    """Refuse well-formed JSONL whose graph evidence violates the ledger schema."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        PublicationRefused,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _GraphData, _Publication

    output = tmp_path / "graphify-out"
    output.mkdir()
    graph_path = output / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "nodes": [{"id": "existing"}],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    record = {
        "source": "source.py",
        "interpretation": "structural",
        "provisional": False,
        "nodes": [],
        "edges": [],
        "hyperedges": [],
    }
    record[bucket] = [invalid_item]
    ledger_path = output / ".graphify_contributions.jsonl"
    ledger_path.write_text(
        "\n".join(
            (
                '{"schema":"graphify-source-contributions","version":1}',
                json.dumps(record, separators=(",", ":")),
                "",
            )
        ),
        encoding="utf-8",
    )
    ledger_before = ledger_path.read_bytes()
    graph_before = graph_path.read_bytes()

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "nodes": [{"id": "fallback"}],
                    "links": [],
                },
                force=True,
            )
        ),
    )

    assert isinstance(outcome, PublicationRefused)
    assert ledger_path.read_bytes() == ledger_before
    assert graph_path.read_bytes() == graph_before


def test_noncanonical_source_alias_in_authoritative_ledger_is_refused(
    tmp_path,
) -> None:
    """Prevent alternate spellings from bypassing composite-key uniqueness."""
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        PublicationRefused,
        ReclusteringRequest,
    )
    from graphify.generation._publication import _Publication

    output = tmp_path / "graphify-out"
    output.mkdir()
    graph_path = output / "graph.json"
    graph_path.write_text(
        '{"directed":false,"multigraph":false,"nodes":[],"links":[]}',
        encoding="utf-8",
    )
    record = {
        "source": "./source.py",
        "interpretation": "structural",
        "provisional": False,
        "nodes": [],
        "edges": [],
        "hyperedges": [],
    }
    ledger_path = output / ".graphify_contributions.jsonl"
    ledger_path.write_text(
        "\n".join(
            (
                '{"schema":"graphify-source-contributions","version":1}',
                json.dumps(record, separators=(",", ":")),
                "",
            )
        ),
        encoding="utf-8",
    )

    outcome = CorpusGraph(Corpus(root=tmp_path, output=output)).reclustering(
        ReclusteringRequest(),
        _publication=_Publication(),
    )

    assert isinstance(outcome, PublicationRefused)


def test_corrupt_cache_warns_and_cannot_redefine_ledger_backed_graph(
    tmp_path,
) -> None:
    """Treat a corrupt accelerator as a miss and retain authoritative evidence."""
    from graphify.cache import load_cached, save_cached
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import (
        _GraphData,
        _Publication,
    )

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    contribution = _SourceContribution(
        source=source,
        interpretation=_InterpretationKind.STRUCTURAL,
        nodes=(
            {
                "id": "authoritative",
                "label": "Authoritative",
                "file_type": "code",
                "source_file": str(source),
            },
        ),
    )
    owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(contributions=(contribution,)),
    )
    graph_before = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    ledger_before = (output / ".graphify_contributions.jsonl").read_bytes()

    save_cached(
        source,
        {
            "nodes": [
                {
                    "id": "cache_poison",
                    "source_file": str(source),
                }
            ],
            "edges": [],
        },
        root=tmp_path,
        cache_root=tmp_path,
    )
    cache_entries = list((output / "cache").rglob("*.json"))
    assert len(cache_entries) == 1
    cache_entries[0].write_text("{broken", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="corrupt cache entry"):
        assert (
            load_cached(
                source,
                root=tmp_path,
                cache_root=tmp_path,
            )
            is None
        )
    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [{"id": "cache_poison"}],
                    "links": [],
                    "hyperedges": [],
                },
                force=True,
            )
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    assert json.loads((output / "graph.json").read_text(encoding="utf-8")) == graph_before
    assert (output / ".graphify_contributions.jsonl").read_bytes() == ledger_before


@pytest.mark.parametrize(
    "invalid_cache",
    [
        "[]",
        "{}",
        '{"nodes":[]}',
        '{"nodes":[42],"edges":[]}',
        '{"nodes":[{}],"edges":[]}',
        '{"nodes":[],"edges":[{}]}',
        '{"nodes":[],"edges":[],"hyperedges":[{"id":"flow"}]}',
    ],
)
def test_wrong_shape_cache_warns_and_is_treated_as_a_miss(
    tmp_path,
    invalid_cache,
) -> None:
    """Reject syntactically valid cache content that is not graph evidence."""
    from graphify.cache import check_semantic_cache, save_cached

    source = tmp_path / "document.md"
    source.write_text("# Evidence\n", encoding="utf-8")
    save_cached(
        source,
        {"nodes": [], "edges": []},
        root=tmp_path,
        cache_root=tmp_path,
        kind="semantic",
    )
    cache_entries = list(
        (tmp_path / "graphify-out" / "cache" / "semantic").rglob("*.json")
    )
    assert len(cache_entries) == 1
    cache_entries[0].write_text(invalid_cache, encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="corrupt cache entry"):
        cached_nodes, cached_edges, cached_hyperedges, uncached = (
            check_semantic_cache(
                [str(source)],
                root=tmp_path,
                cache_root=tmp_path,
            )
        )

    assert (cached_nodes, cached_edges, cached_hyperedges) == ([], [], [])
    assert uncached == [str(source)]


def test_cache_loss_warning_matches_lexical_symlink_identity(tmp_path) -> None:
    """Match cache misses to the same symlink spelling used by the ledger."""
    from graphify.cache import check_semantic_cache
    from graphify.generation import Corpus, CorpusGraph, FullExtractionRequest
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _Publication

    target = tmp_path / "sub" / "document.md"
    target.parent.mkdir()
    target.write_text("# Meaning\n", encoding="utf-8")
    alias = tmp_path / "document-link.md"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    CorpusGraph(
        Corpus(root=tmp_path, output=tmp_path / "graphify-out")
    ).full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=alias,
                    interpretation=_InterpretationKind.SEMANTIC,
                    nodes=(
                        {
                            "id": "meaning",
                            "source_file": str(alias),
                        },
                    ),
                ),
            ),
        ),
    )

    with pytest.warns(
        RuntimeWarning,
        match="expected from active Source contributions were unavailable",
    ):
        _, _, _, uncached = check_semantic_cache(
            [str(alias)],
            root=tmp_path,
            cache_root=tmp_path,
        )

    assert uncached == [str(alias)]


def test_cache_loss_warns_and_cannot_redefine_ledger_backed_graph(tmp_path) -> None:
    """Warn when prior semantic acceleration disappears, retaining the ledger."""
    from graphify.cache import check_semantic_cache, save_cached
    from graphify.generation import (
        Corpus,
        CorpusGraph,
        FullExtractionRequest,
        GenerationPublished,
        ReclusteringRequest,
    )
    from graphify.generation._contributions import (
        _InterpretationKind,
        _SourceContribution,
    )
    from graphify.generation._publication import _GraphData, _Publication

    source = tmp_path / "document.md"
    source.write_text("# Durable meaning\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    owner = CorpusGraph(Corpus(root=tmp_path, output=output))
    owner.full_extraction(
        FullExtractionRequest(),
        _publication=_Publication(
            contributions=(
                _SourceContribution(
                    source=source,
                    interpretation=_InterpretationKind.SEMANTIC,
                    nodes=(
                        {
                            "id": "durable_meaning",
                            "label": "Durable meaning",
                            "file_type": "document",
                            "source_file": str(source),
                        },
                    ),
                ),
            ),
        ),
    )
    graph_before = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    ledger_before = (output / ".graphify_contributions.jsonl").read_bytes()
    save_cached(
        source,
        {
            "nodes": [
                {
                    "id": "durable_meaning",
                    "source_file": str(source),
                }
            ],
            "edges": [],
        },
        root=tmp_path,
        cache_root=tmp_path,
        kind="semantic",
    )
    for cache_entry in (output / "cache" / "semantic").rglob("*.json"):
        cache_entry.unlink()

    with pytest.warns(
        RuntimeWarning,
        match="expected from active Source contributions were unavailable",
    ):
        _, _, _, uncached = check_semantic_cache(
            [str(source)],
            root=tmp_path,
            cache_root=tmp_path,
        )
    assert uncached == [str(source)]

    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            graph=_GraphData(
                {
                    "directed": False,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [{"id": "lost_cache"}],
                    "links": [],
                    "hyperedges": [],
                },
                force=True,
            )
        ),
    )

    assert isinstance(outcome, GenerationPublished)
    assert json.loads((output / "graph.json").read_text(encoding="utf-8")) == graph_before
    assert (output / ".graphify_contributions.jsonl").read_bytes() == ledger_before
