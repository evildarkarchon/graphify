"""Reconciliation primitives shared by the lifecycle operations.

Code update and Full extraction reconcile the same Corpus against the same
active Graph generation; they differ in which evidence they are entitled to
produce, not in how the Corpus is read, how policy is resolved, or how a
prospective manifest is derived. Those shared steps live here so the two
operations cannot answer the same question about one Corpus differently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from graphify.generation._contributions import (
    _InterpretationKind,
    _PreparedContribution,
    _SourceContribution,
    _adopt_legacy_graph,
    _is_external_source_identity,
    _iter_contribution_ledger,
    _relative_source_identity,
)
from graphify.generation._layout import _PublicationLayout
from graphify.generation._publication import _CanonicalArtifact, _ManifestUpdate
from graphify.generation._types import PublicationRefused

# Community identity, its analysis, and the report describe a clustered
# generation. Both operations here publish a Raw one, so these stop describing
# the active generation and are retired instead of being presented as current.
_RETIRED_BY_RAW_PUBLICATION = frozenset(
    {
        _CanonicalArtifact.REPORT,
        _CanonicalArtifact.ANALYSIS,
        _CanonicalArtifact.LABELS,
        _CanonicalArtifact.LABEL_SIGNATURES,
    }
)


def _active_build_policy(
    layout: _PublicationLayout,
) -> tuple[list[str], bool] | PublicationRefused:
    """Return the persisted corpus-shaping policy, or the documented defaults.

    The documented defaults apply only when no policy has ever been recorded: no
    extra excludes, and VCS ignore files honored. A recorded policy that cannot
    be read fails closed instead, because falling back to the defaults would
    silently re-include the very paths the Corpus was told to exclude and grow
    the graph past every downstream guard.
    """
    excludes: list[str] = []
    gitignore = True
    path = layout.path_for(_CanonicalArtifact.BUILD_CONFIG)
    if not path.is_file():
        return excludes, gitignore
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return PublicationRefused(f"cannot read the active Corpus build policy: {exc}")
    if not isinstance(config, dict):
        return PublicationRefused(
            f"the active Corpus build policy is not an object: {path}"
        )
    persisted = config.get("excludes")
    if isinstance(persisted, list):
        excludes = [value for value in persisted if isinstance(value, str) and value]
    if isinstance(config.get("gitignore"), bool):
        gitignore = config["gitignore"]
    return excludes, gitignore


def _active_evidence(
    layout: _PublicationLayout,
    graph_path: Path,
    root: Path,
) -> tuple[_PreparedContribution, ...]:
    """Return the active evidence an operation must reconcile against.

    An authoritative ledger is used as-is. A Corpus that predates the ledger is
    adopted from its valid graph instead of being treated as having no evidence,
    so a first operation over legacy output stays non-destructive: attributed
    legacy evidence is replaced per source as real extraction covers it, and
    unattributed legacy evidence survives.

    Raises when present evidence cannot be validated. Reconciling against an
    empty set in that case would silently retire evidence the operation may not
    be allowed to reinterpret, so the caller refuses publication instead.
    """
    ledger = layout.path_for(_CanonicalArtifact.CONTRIBUTIONS)
    if ledger.is_file():
        return tuple(_iter_contribution_ledger(ledger))
    if graph_path.is_file():
        return _adopt_legacy_graph(graph_path, root)
    return ()


def _prospective_manifest(
    layout: _PublicationLayout,
    update: _ManifestUpdate,
) -> tuple[dict[str, Any], bool]:
    """Return the manifest this discovery would publish, and whether it advances.

    The manifest is the one canonical artifact the publisher recomputes from the
    live filesystem rather than receiving from a caller, so an identical result
    genuinely means Corpus state did not advance. The prospective manifest is
    produced by the production writer against a copy of the active one, so no
    second encoding of manifest state — including per-source semantic freshness
    — exists to drift from what publication will record.
    """
    import shutil
    import tempfile

    from graphify.detect import save_manifest

    active = layout.path_for(_CanonicalArtifact.MANIFEST)
    active_bytes = active.read_bytes() if active.is_file() else b""
    with tempfile.TemporaryDirectory() as scratch:
        candidate = Path(scratch) / _CanonicalArtifact.MANIFEST.value
        if active_bytes:
            shutil.copy2(active, candidate)
        save_manifest(
            dict(update.files),
            manifest_path=str(candidate),
            kind=update.kind,
            root=update.root,
            scan_corpus=update.scan_corpus,
            clear_semantic=update.clear_semantic,
        )
        candidate_bytes = candidate.read_bytes()
    try:
        prospective = json.loads(candidate_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        prospective = None
    return (
        prospective if isinstance(prospective, dict) else {},
        candidate_bytes != active_bytes,
    )


def _pending_marker_is_raised(path: Path) -> bool:
    """Return whether the compatibility pending marker already reads as raised."""
    try:
        return path.read_text(encoding="utf-8") == "1"
    except (OSError, ValueError):
        return False


def _root_marker_advances(layout: _PublicationLayout, marker: str) -> bool:
    """Return whether the active Corpus root marker still records ``marker``."""
    path = layout.path_for(_CanonicalArtifact.ROOT)
    try:
        return path.read_text(encoding="utf-8") != marker
    except (OSError, ValueError):
        return True


def _live_source_identities(detection: Mapping[str, Any], root: Path) -> set[str]:
    """Return the portable identity of every source the live Corpus contains.

    Every discovered file counts, not only the structural corpus: a live
    document still owns its Semantic evidence even when the running operation
    cannot re-derive it. Sources outside the Corpus root cannot be keyed in the
    ledger and are therefore not part of the live set.
    """
    identities: set[str] = set()
    for group in detection["files"].values():
        for path in group:
            identity = _identity_or_none(path, root)
            if identity is not None:
                identities.add(identity)
    return identities


def _structural_corpus(
    detection: Mapping[str, Any],
    unreproducible: set[str],
    root: Path,
) -> dict[str, Path]:
    """Map every source an operation owns structurally to its scanned path.

    Documents with structural extractors (Markdown, MDX, Quarto) belong to the
    code-only corpus: their headings are structural evidence any operation can
    derive without a Semantic provider. A document in ``unreproducible`` is
    excluded — either it already carries evidence the operation cannot reproduce,
    or a provider is interpreting it in this very run — because that
    interpretation is the source's whole representation, and quick-scanning it
    too would both represent the document twice and replace knowledge with
    headings.

    Code sources are never excluded: structural evidence is their primary
    representation, so re-deriving one legitimately supersedes provisional
    evidence adopted for it.
    """
    from graphify.extract import _get_extractor

    files = detection["files"]
    owned: dict[str, Path] = {}
    candidates = [(path, False) for path in files.get("code", [])]
    candidates.extend((path, True) for path in files.get("document", []))
    for path, is_document in candidates:
        source = Path(path)
        if is_document and _get_extractor(source) is None:
            continue
        identity = _identity_or_none(source, root)
        if identity is None or (is_document and identity in unreproducible):
            continue
        owned[identity] = source
    return owned


def _as_source_contribution(
    contribution: _PreparedContribution,
    *,
    stale: bool | None = None,
) -> _SourceContribution:
    """Return one validated ledger record as a publishable contribution."""
    return _SourceContribution(
        source=contribution.source,
        interpretation=contribution.interpretation,
        nodes=contribution.nodes,
        edges=contribution.edges,
        hyperedges=contribution.hyperedges,
        provisional=contribution.provisional,
        stale=contribution.stale if stale is None else stale,
    )


def _shrink_is_accounted(
    graph_path: Path,
    candidate: Mapping[str, Any],
    accounted: frozenset[str] | set[str],
    root: Path,
) -> bool:
    """Return whether every node the candidate drops is explained by this run.

    The established node-count guard exists to catch silently partial
    extraction. An operation reconciled from authoritative discovery loses
    nodes only where a source was re-derived or retired, so this reproduces the
    per-source accounting rather than waiving the guard wholesale: an
    unexplained loss — for instance Semantic evidence disappearing from a source
    that is still live — still refuses publication.
    """
    try:
        active = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(active, dict) or not isinstance(active.get("nodes"), list):
        return False
    candidate_ids = {
        node.get("id")
        for node in candidate.get("nodes", [])
        if isinstance(node, Mapping)
    }
    for node in active["nodes"]:
        if not isinstance(node, Mapping) or node.get("id") in candidate_ids:
            continue
        source = node.get("source_file")
        if not source or not isinstance(source, str):
            # Unattributed evidence cannot belong to a source this run replaced,
            # and the established node-count guard treats it as accounted too.
            continue
        # A present but unattributable spelling proves nothing about which source
        # lost the node, so it stays unaccounted and the guard keeps applying.
        if _identity_or_none(source, root) not in accounted:
            return False
    return True


def _identity_or_none(source: Path | str, root: Path) -> str | None:
    """Return a portable Corpus-relative identity, or None when unattributable.

    A source system's address is already the identity its evidence is keyed by,
    so it is returned unchanged rather than being relativized against a Corpus
    root it was never inside — which would report it as unattributable and let
    its evidence disappear unaccounted for.
    """
    if _is_external_source_identity(source):
        return str(source)
    try:
        return _relative_source_identity(source, root)
    except (OSError, ValueError):
        return None


def _structural_contributions(
    result: Mapping[str, Any],
) -> tuple[_SourceContribution, ...]:
    """Group one structural extraction result into per-source contributions."""
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for bucket in ("nodes", "edges", "hyperedges"):
        for item in result.get(bucket) or ():
            source = item.get("source_file")
            if not isinstance(source, str) or not source:
                continue
            group = grouped.setdefault(
                source,
                {"nodes": [], "edges": [], "hyperedges": []},
            )
            group[bucket].append(item)
    return tuple(
        _SourceContribution(
            source=source,
            interpretation=_InterpretationKind.STRUCTURAL,
            nodes=tuple(group["nodes"]),
            edges=tuple(group["edges"]),
            hyperedges=tuple(group["hyperedges"]),
        )
        for source, group in sorted(grouped.items())
    )
