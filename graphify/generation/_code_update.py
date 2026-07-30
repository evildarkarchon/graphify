"""Private deterministic, LLM-free Code update execution.

Code update is the structural half of the Graph-generation lifecycle: it
performs authoritative Corpus discovery, re-derives structural Source
contributions, reconciles them against the active ledger, and publishes the
resulting code-only Graph generation. It never invokes a Semantic provider, so
a routine update stays free and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from graphify.generation._contributions import (
    _UNATTRIBUTED_LEGACY_SOURCE,
    _InterpretationKind,
    _PreparedContribution,
    _SourceContribution,
    _adopt_legacy_graph,
    _iter_contribution_ledger,
    _ledger_matches,
    _materialize_graph_data,
    _prepare_contributions,
    _relative_source_identity,
)
from graphify.generation._layout import _PublicationLayout
from graphify.generation._publication import (
    _CanonicalArtifact,
    _GraphData,
    _ManifestUpdate,
    _Publication,
)
from graphify.generation._publisher import _Publisher
from graphify.generation._transaction import _PublicationTransaction
from graphify.generation._types import (
    CodeUpdateRequest,
    Corpus,
    OperationFailed,
    PublicationRefused,
    TerminalOutcome,
)

# Community identity, its analysis, and the report describe a clustered
# generation. A Code update publishes a Raw one, so these stop describing the
# active generation and are retired instead of being presented as current.
_RETIRED_BY_RAW_PUBLICATION = frozenset(
    {
        _CanonicalArtifact.REPORT,
        _CanonicalArtifact.ANALYSIS,
        _CanonicalArtifact.LABELS,
        _CanonicalArtifact.LABEL_SIGNATURES,
    }
)


def _execute_code_update(
    corpus: Corpus,
    request: CodeUpdateRequest,
) -> TerminalOutcome:
    """Run one Code update end to end and publish through the owning module.

    Discovery is authoritative: the live Corpus decides which sources may
    contribute evidence, so additions, edits, renames, deletions, and newly
    excluded sources are all reconciled from the same scan. No step consults a
    Semantic provider, and evidence this operation cannot re-derive is carried
    forward rather than replaced.
    """
    # Recovery precedes every read of the active generation, not just the write:
    # reconciliation below decides what to retire from the active ledger and
    # graph, so it must never observe an interrupted publication's state.
    recovery = _PublicationTransaction.recover(corpus)
    if recovery is not None:
        return recovery
    try:
        planned = _plan_code_update(corpus, request)
    except Exception as exc:
        # Preparation runs discovery, third-party parsers, and evidence
        # validation. A failure there is a terminal outcome of an accepted
        # operation, not a crash for adapters to interpret, and it leaves the
        # active Graph generation untouched because nothing has been staged yet.
        return OperationFailed(f"could not prepare the Code update: {exc}")
    if isinstance(planned, PublicationRefused):
        return planned
    return _Publisher(corpus).publish(planned, operation="code-update")


def _plan_code_update(
    corpus: Corpus,
    request: CodeUpdateRequest,
) -> _Publication | PublicationRefused:
    """Reconcile the live Corpus into the publication one Code update should make."""
    root = corpus.root
    output = corpus.output
    # Resolve active artifact locations exactly as the next transaction will, so
    # reconciliation reads the same files publication is going to replace.
    layout = _PublicationTransaction.active_layout(corpus)
    policy = _active_build_policy(layout)
    if isinstance(policy, PublicationRefused):
        return policy
    excludes, gitignore = policy

    from graphify.detect import detect

    detection = detect(
        root,
        extra_excludes=excludes or None,
        gitignore=gitignore,
        cache_root=output.parent,
    )
    if detection.get("walk_errors"):
        # Discovery could not enumerate the whole Corpus, so absence is not
        # deletion evidence and the resulting generation would silently describe
        # a subset. Code update is always fail closed here: ``force`` authorizes
        # a smaller graph, never a graph built from an unknown Corpus.
        return PublicationRefused(
            "Corpus discovery was incomplete "
            f"({len(detection['walk_errors'])} unreadable location(s)): "
            "refusing to replace the active Graph generation"
        )

    graph_path = layout.path_for(_CanonicalArtifact.GRAPH)
    try:
        active = _active_evidence(layout, graph_path, root)
    except (OSError, ValueError) as exc:
        return PublicationRefused(
            f"cannot validate authoritative Source contributions: {exc}"
        )

    live = _live_source_identities(detection, root)
    # Evidence this operation cannot produce: interpreted meaning, and adopted
    # legacy evidence whose interpretation is unknown. A document carrying either
    # is represented by it, so re-deriving that document structurally would
    # replace knowledge a Code update has no way to recover.
    unreproducible = {
        contribution.source
        for contribution in active
        if contribution.interpretation is not _InterpretationKind.STRUCTURAL
    }
    owned = _structural_corpus(detection, unreproducible, root)

    from graphify.extract import extract

    # Every source this operation owns is re-derived, so ``request.changed_paths``
    # never restricts the work: the content-hash extraction cache is the
    # accelerator, and an omitted or unknown hint can therefore neither narrow
    # the published graph nor become deletion authority.
    result: dict[str, Any] = (
        extract(list(owned.values()), cache_root=output.parent, root=root)
        if owned
        else {"nodes": [], "edges": [], "hyperedges": []}
    )
    contributions = _structural_contributions(result) + _preserved_contributions(
        active,
        live=live,
        rederived=frozenset(owned),
    )
    prepared = _prepare_contributions(contributions, root)
    graph_data = _materialize_graph_data(prepared)

    # Sources whose evidence this run is entitled to replace or retire: every
    # source it re-derived from the live Corpus, plus every source the same
    # authoritative scan proved has left it. A graph that shrinks only because
    # of those is reconciled, not partial, so it must not need shrink authority.
    accounted = frozenset(owned) | {
        contribution.source
        for contribution in active
        if contribution.source not in live
    }

    # The ledger is the authoritative graph evidence, so an unchanged ledger
    # means the Graph generation itself did not change: leave graph.json and the
    # ledger exactly as published so a Corpus-state advance causes no output
    # churn. A missing or unreadable graph still has to be rematerialized.
    graph_changed = not graph_path.is_file() or not _ledger_matches(
        layout.path_for(_CanonicalArtifact.CONTRIBUTIONS),
        prepared,
    )
    # kind="ast" is what keeps per-source semantic freshness truthful: the
    # production manifest writer preserves a source's semantic_hash only while
    # its content is unchanged, so a live source this operation could not
    # reinterpret stays pending instead of being stamped as current.
    manifest = _ManifestUpdate(
        files=detection["files"],
        kind="ast",
        root=root,
        scan_corpus={
            path for group in detection["files"].values() for path in group
        },
    )

    return _Publication(
        contributions=contributions if graph_changed else None,
        graph=(
            _GraphData(
                graph_data,
                force=(
                    request.force
                    or _shrink_is_accounted(
                        graph_path,
                        graph_data,
                        accounted,
                        root,
                    )
                ),
            )
            if graph_changed
            else None
        ),
        manifest=manifest if _manifest_advances(layout, manifest) else None,
        root_marker=(
            str(root) if _root_marker_advances(layout, str(root)) else None
        ),
        # Publishing graph evidence publishes a Raw graph generation, so the
        # clustered artifacts of the previous generation stop describing it and
        # must be retired rather than presented as current. Reclustering
        # republishes them from this generation's Source contributions.
        retire=_RETIRED_BY_RAW_PUBLICATION if graph_changed else frozenset(),
        protect_previous=graph_changed,
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
    """Return the active evidence this Code update must reconcile against.

    An authoritative ledger is used as-is. A Corpus that predates the ledger is
    adopted from its valid graph instead of being treated as having no evidence,
    so a first Code update over legacy output stays non-destructive: attributed
    legacy evidence is replaced per source as real extraction covers it, and
    unattributed legacy evidence survives.

    Raises when present evidence cannot be validated. Reconciling against an
    empty set in that case would silently retire evidence this operation is not
    allowed to reinterpret, so the caller refuses publication instead.
    """
    ledger = layout.path_for(_CanonicalArtifact.CONTRIBUTIONS)
    if ledger.is_file():
        return tuple(_iter_contribution_ledger(ledger))
    if graph_path.is_file():
        return _adopt_legacy_graph(graph_path, root)
    return ()


def _manifest_advances(layout: _PublicationLayout, update: _ManifestUpdate) -> bool:
    """Return whether this discovery would change the active Corpus manifest.

    The manifest is the one canonical artifact the publisher recomputes from the
    live filesystem rather than receiving from a caller, so an identical result
    genuinely means Corpus state did not advance. The prospective manifest is
    produced by the production writer against a copy of the active one, so no
    second encoding of manifest state exists to drift.
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
        return candidate.read_bytes() != active_bytes


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
    document still owns its Semantic evidence even though a Code update cannot
    re-derive it. Sources outside the Corpus root cannot be keyed in the ledger
    and are therefore not part of the live set.
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
    """Map every source this Code update owns structurally to its scanned path.

    Documents with structural extractors (Markdown, MDX, Quarto) belong to the
    code-only corpus: their headings are structural evidence a Code update can
    derive without a Semantic provider. A document already carrying evidence this
    operation cannot reproduce is excluded — that evidence is the source's whole
    representation, and quick-scanning it too would both represent the document
    twice and replace knowledge with headings.

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


def _preserved_contributions(
    active: tuple[_PreparedContribution, ...],
    *,
    live: set[str],
    rederived: frozenset[str],
) -> tuple[_SourceContribution, ...]:
    """Carry forward active evidence this Code update must not replace.

    Structural evidence never survives: this run re-derives it for every source
    it owns, and a source it no longer owns — a departed path, or a document
    whose Semantic evidence now represents it — must not keep stale structural
    evidence either.

    Semantic evidence always survives for a live source, because Code update
    never reinterprets. That is custody, not a freshness claim: whether a live
    source's Semantic evidence is still current is recorded separately by the
    manifest this operation publishes, so a changed source stays pending.

    Provisional legacy evidence is replaced per source once real extraction
    covers it, while unattributed legacy evidence survives until a complete Full
    extraction or an explicit operator retirement retires it.
    """
    preserved: list[_SourceContribution] = []
    for contribution in active:
        if contribution.source == _UNATTRIBUTED_LEGACY_SOURCE:
            preserved.append(_as_source_contribution(contribution))
            continue
        if contribution.source not in live:
            continue
        if contribution.interpretation is _InterpretationKind.STRUCTURAL:
            continue
        if contribution.provisional and contribution.source in rederived:
            continue
        preserved.append(_as_source_contribution(contribution))
    return tuple(preserved)


def _as_source_contribution(
    contribution: _PreparedContribution,
) -> _SourceContribution:
    """Return one validated ledger record as a publishable contribution."""
    return _SourceContribution(
        source=contribution.source,
        interpretation=contribution.interpretation,
        nodes=contribution.nodes,
        edges=contribution.edges,
        hyperedges=contribution.hyperedges,
        provisional=contribution.provisional,
    )


def _shrink_is_accounted(
    graph_path: Path,
    candidate: Mapping[str, Any],
    accounted: frozenset[str] | set[str],
    root: Path,
) -> bool:
    """Return whether every node the candidate drops is explained by this run.

    The established node-count guard exists to catch silently partial
    extraction. A Code update reconciled from authoritative discovery loses
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
    """Return a portable Corpus-relative identity, or None when unattributable."""
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
