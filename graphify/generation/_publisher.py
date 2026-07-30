"""Private canonical artifact publisher used by ``CorpusGraph``."""

from __future__ import annotations

from pathlib import Path

from graphify.generation._contributions import (
    _adopt_graph_payload,
    _adopt_legacy_graph,
    _iter_contribution_ledger,
    _materialize_graph_data,
    _prepare_contributions,
    _write_contribution_ledger,
)
from graphify.generation._layout import _PublicationLayout
from graphify.generation._publication import (
    _CanonicalArtifact,
    _GraphData,
    _GraphModel,
    _Publication,
)
from graphify.generation._types import (
    AlreadyCurrent,
    Corpus,
    CorpusStateAdvanced,
    GenerationPublished,
    OperationFailed,
    PublicationRefused,
    TerminalOutcome,
)


class _Publisher:
    """Apply one accepted publication to a Corpus's canonical artifacts."""

    def __init__(self, corpus: Corpus) -> None:
        """Bind filesystem publication policy to one Corpus."""
        self._corpus = corpus

    def publish(self, publication: _Publication, *, operation: str) -> TerminalOutcome:
        """Write one prepared publication through production writers."""
        output = self._corpus.output
        layout = _PublicationLayout(
            root=self._corpus.root,
            output=output,
            overrides=publication.artifact_paths,
        )
        graph_path = layout.path_for(_CanonicalArtifact.GRAPH)
        contribution_path = layout.path_for(_CanonicalArtifact.CONTRIBUTIONS)
        active_contributions = None
        if contribution_path.exists():
            try:
                active_contributions = tuple(
                    _iter_contribution_ledger(contribution_path)
                )
            except (OSError, ValueError) as exc:
                return PublicationRefused(
                    f"cannot validate authoritative Source contributions: {exc}"
                )
        prepared_contributions = None
        contributions_to_write = None
        if publication.contributions is not None:
            prepared_contributions = _prepare_contributions(
                publication.contributions,
                self._corpus.root,
            )
            contributions_to_write = prepared_contributions
        graph = publication.graph
        if (
            prepared_contributions is None
            and active_contributions is None
            and graph is None
            and graph_path.exists()
        ):
            try:
                prepared_contributions = _adopt_legacy_graph(
                    graph_path,
                    self._corpus.root,
                )
            except (OSError, ValueError) as exc:
                return PublicationRefused(str(exc))
            contributions_to_write = prepared_contributions
        elif (
            prepared_contributions is None
            and graph is not None
            and not (
                active_contributions is not None
                and operation == "reclustering"
            )
        ):
            try:
                prepared_contributions = _adopt_graph_payload(
                    self._graph_payload(graph),
                    self._corpus.root,
                )
            except ValueError as exc:
                return PublicationRefused(str(exc))
            contributions_to_write = prepared_contributions
        if prepared_contributions is not None:
            if (
                publication.contributions is not None
                or graph is not None
                or not graph_path.exists()
            ):
                graph = self._graph_from_contributions(
                    prepared_contributions,
                    graph,
                )
        elif (
            active_contributions is not None
            and operation == "reclustering"
            and graph is not None
        ):
            graph = self._graph_from_contributions(active_contributions, graph)
        refusal = self._graph_refusal(graph, graph_path)
        if refusal is not None:
            return refusal

        changed: list[str] = []
        try:
            output.mkdir(parents=True, exist_ok=True)
            if publication.protect_previous:
                from graphify.export import backup_if_protected

                backup_if_protected(output)

            if contributions_to_write is not None:
                _write_contribution_ledger(
                    contribution_path,
                    contributions_to_write,
                )
                changed.append(_CanonicalArtifact.CONTRIBUTIONS.value)
            if graph is not None:
                if not self._write_graph(graph, graph_path):
                    return PublicationRefused(
                        f"the established graph safety guard refused {graph_path}"
                    )
                changed.append(_CanonicalArtifact.GRAPH.value)
            if publication.report is not None:
                self._write_text(
                    _CanonicalArtifact.REPORT,
                    publication.report,
                    changed,
                    layout,
                )
            self._write_json(
                _CanonicalArtifact.ANALYSIS,
                publication.analysis,
                changed,
                layout,
                indent=2,
            )
            self._write_json(
                _CanonicalArtifact.LABELS,
                publication.labels,
                changed,
                layout,
                indent=2,
            )
            self._write_json(
                _CanonicalArtifact.LABEL_SIGNATURES,
                publication.label_signatures,
                changed,
                layout,
            )
            self._write_json(
                _CanonicalArtifact.BUILD_CONFIG,
                publication.build_config,
                changed,
                layout,
            )
            self._write_json(
                _CanonicalArtifact.SEMANTIC_MARKER,
                publication.semantic_marker,
                changed,
                layout,
            )
            if publication.root_marker is not None:
                self._write_text(
                    _CanonicalArtifact.ROOT,
                    publication.root_marker,
                    changed,
                    layout,
                )
            if publication.manifest is not None:
                from graphify.detect import save_manifest

                update = publication.manifest
                save_manifest(
                    dict(update.files),
                    manifest_path=str(layout.path_for(_CanonicalArtifact.MANIFEST)),
                    kind=update.kind,
                    root=update.root,
                    scan_corpus=update.scan_corpus,
                    clear_semantic=update.clear_semantic,
                )
                changed.append(_CanonicalArtifact.MANIFEST.value)
            if publication.needs_update is not None:
                if publication.needs_update:
                    self._write_text(
                        _CanonicalArtifact.NEEDS_UPDATE,
                        "1",
                        changed,
                        layout,
                    )
                else:
                    self._retire(_CanonicalArtifact.NEEDS_UPDATE, changed, layout)
            for artifact in publication.retire:
                self._retire(artifact, changed, layout)
        except Exception as exc:
            return OperationFailed(str(exc))

        if graph is not None:
            return GenerationPublished(tuple(changed))
        if changed:
            return CorpusStateAdvanced(tuple(changed))
        return AlreadyCurrent()

    def _graph_payload(
        self,
        graph: _GraphData | _GraphModel,
    ) -> dict:
        """Return a node-link compatibility payload for provisional admission."""
        if isinstance(graph, _GraphData):
            return dict(graph.data)
        from networkx.readwrite import json_graph

        try:
            data = json_graph.node_link_data(graph.graph, edges="links")
        except TypeError:
            data = json_graph.node_link_data(graph.graph)
        data["hyperedges"] = list(
            getattr(graph.graph, "graph", {}).get("hyperedges", [])
        )
        return data

    def _graph_from_contributions(
        self,
        contributions,
        prepared_graph: _GraphData | _GraphModel | None,
    ) -> _GraphData | _GraphModel:
        """Materialize topology from the ledger while retaining view metadata."""
        multigraph = (
            prepared_graph.graph.is_multigraph()
            if isinstance(prepared_graph, _GraphModel)
            else (
                bool(prepared_graph.data.get("multigraph", False))
                if isinstance(prepared_graph, _GraphData)
                else False
            )
        )
        data = _materialize_graph_data(
            contributions,
            multigraph=multigraph,
        )
        if isinstance(prepared_graph, _GraphData):
            template = prepared_graph.data
            view = {
                key: value
                for key, value in template.items()
                if key not in {"nodes", "links", "edges", "hyperedges"}
            }
            view["nodes"] = data["nodes"]
            if "edges" in template and "links" not in template:
                view["edges"] = data["links"]
            else:
                view["links"] = data["links"]
            if data["hyperedges"] or "hyperedges" in template:
                view["hyperedges"] = data["hyperedges"]
            data = view
        if not isinstance(prepared_graph, _GraphModel):
            force = prepared_graph.force if isinstance(prepared_graph, _GraphData) else False
            return _GraphData(data, force=force)

        from graphify.paths import load_node_link_graph

        data["directed"] = prepared_graph.graph.is_directed()
        data["multigraph"] = prepared_graph.graph.is_multigraph()
        data["graph"] = {
            key: value
            for key, value in getattr(prepared_graph.graph, "graph", {}).items()
            if key != "hyperedges"
        }
        graph = load_node_link_graph(data)
        graph.graph["hyperedges"] = data.get("hyperedges", [])
        return _GraphModel(
            graph=graph,
            communities=prepared_graph.communities,
            force=prepared_graph.force,
            built_at_commit=prepared_graph.built_at_commit,
            community_labels=prepared_graph.community_labels,
        )

    def _graph_refusal(
        self,
        graph: _GraphData | _GraphModel | None,
        graph_path: Path,
    ) -> PublicationRefused | None:
        """Apply the existing fail-closed node-count guard before any write."""
        if graph is None or graph.force:
            return None
        from graphify.export import MALFORMED_GRAPH, existing_graph_node_count

        existing = existing_graph_node_count(graph_path)
        if existing is MALFORMED_GRAPH or (existing is not None and not isinstance(existing, int)):
            return PublicationRefused(f"existing {graph_path} is non-empty but cannot be validated")
        if existing is None:
            return None
        if isinstance(graph, _GraphModel):
            new_count = graph.graph.number_of_nodes()
        else:
            nodes = graph.data.get("nodes", [])
            new_count = len(nodes) if isinstance(nodes, list) else 0
        if new_count < existing:
            return PublicationRefused(
                f"new graph has {new_count} nodes but active graph has {existing}"
            )
        return None

    def _write_graph(
        self,
        graph: _GraphData | _GraphModel,
        graph_path: Path,
    ) -> bool:
        """Serialize a prepared graph with the established schema and atomic writer."""
        if isinstance(graph, _GraphData):
            from graphify.paths import write_json_atomic

            write_json_atomic(graph_path, dict(graph.data), indent=2)
            return True
        from graphify.export import to_json

        return to_json(
            graph.graph,
            dict(graph.communities),
            str(graph_path),
            force=graph.force,
            built_at_commit=graph.built_at_commit,
            community_labels=(
                dict(graph.community_labels) if graph.community_labels is not None else None
            ),
        )

    def _write_text(
        self,
        artifact: _CanonicalArtifact,
        content: str,
        changed: list[str],
        layout: _PublicationLayout,
    ) -> None:
        """Atomically write a UTF-8 canonical text artifact."""
        from graphify.paths import write_text_atomic

        write_text_atomic(layout.path_for(artifact), content)
        changed.append(artifact.value)

    def _write_json(
        self,
        artifact: _CanonicalArtifact,
        content,
        changed: list[str],
        layout: _PublicationLayout,
        *,
        indent: int | None = None,
    ) -> None:
        """Atomically write a canonical JSON artifact when content was supplied."""
        if content is None:
            return
        from graphify.paths import write_json_atomic

        write_json_atomic(
            layout.path_for(artifact),
            dict(content),
            indent=indent,
            ensure_ascii=False,
        )
        changed.append(artifact.value)

    def _retire(
        self,
        artifact: _CanonicalArtifact,
        changed: list[str],
        layout: _PublicationLayout,
    ) -> None:
        """Remove a stale canonical artifact after its replacement is accepted."""
        path = layout.path_for(artifact)
        if path.exists():
            path.unlink()
            changed.append(artifact.value)
