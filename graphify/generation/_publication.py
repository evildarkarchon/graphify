"""Private compatibility payloads for the publication-custody migration slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class _CanonicalArtifact(str, Enum):
    """Stable artifacts and compatibility projections owned by a Corpus graph."""

    GRAPH = "graph.json"
    REPORT = "GRAPH_REPORT.md"
    ANALYSIS = ".graphify_analysis.json"
    LABELS = ".graphify_labels.json"
    LABEL_SIGNATURES = ".graphify_labels.json.sig"
    MANIFEST = "manifest.json"
    ROOT = ".graphify_root"
    BUILD_CONFIG = ".graphify_build.json"
    SEMANTIC_MARKER = ".graphify_semantic_marker"
    NEEDS_UPDATE = "needs_update"


@dataclass(frozen=True)
class _GraphData:
    """A prepared raw or node-link graph payload."""

    data: Mapping[str, Any]
    force: bool = False


@dataclass(frozen=True)
class _GraphModel:
    """A prepared NetworkX graph requiring the established JSON serializer."""

    graph: Any
    communities: Mapping[int, list[str]]
    force: bool = False
    built_at_commit: str | None = None
    community_labels: Mapping[int, str] | None = None


@dataclass(frozen=True)
class _ManifestUpdate:
    """Inputs to the production manifest writer."""

    files: Mapping[str, list[str]]
    kind: str = "both"
    root: Path | None = None
    scan_corpus: set[str] | list[str] | None = None
    clear_semantic: set[str] | list[str] | None = None


@dataclass(frozen=True)
class _Publication:
    """Canonical artifacts prepared by an existing lifecycle algorithm.

    This payload is intentionally private. It lets current adapters cross one
    owner during the custody cutover without making prepared artifacts part of
    the long-term public operation contract.
    """

    graph: _GraphData | _GraphModel | None = None
    report: str | None = None
    analysis: Mapping[str, Any] | None = None
    labels: Mapping[str, Any] | None = None
    label_signatures: Mapping[str, Any] | None = None
    manifest: _ManifestUpdate | None = None
    root_marker: str | None = None
    build_config: Mapping[str, Any] | None = None
    semantic_marker: Mapping[str, Any] | None = None
    needs_update: bool | None = None
    retire: frozenset[_CanonicalArtifact] = field(default_factory=frozenset)
    artifact_paths: Mapping[_CanonicalArtifact, Path] = field(default_factory=dict)
    protect_previous: bool = False
