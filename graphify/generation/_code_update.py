"""Private deterministic, LLM-free Code update execution.

Code update is the structural half of the Graph-generation lifecycle: it
performs authoritative Corpus discovery, re-derives structural Source
contributions, reconciles them against the active ledger, and publishes the
resulting code-only Graph generation. It never invokes a Semantic provider, so
a routine update stays free and reproducible.
"""

from __future__ import annotations

from typing import Any

from graphify.generation._contributions import (
    _UNATTRIBUTED_LEGACY_SOURCE,
    _InterpretationKind,
    _PreparedContribution,
    _SourceContribution,
    _is_external_source_identity,
    _ledger_matches,
    _materialize_graph_data,
    _prepare_contributions,
)
from graphify.generation._publication import (
    _CanonicalArtifact,
    _GraphData,
    _ManifestUpdate,
    _Publication,
)
from graphify.generation._publisher import _Publisher
from graphify.generation._reconciliation import (
    _RETIRED_BY_RAW_PUBLICATION,
    _active_build_policy,
    _active_evidence,
    _as_source_contribution,
    _live_source_identities,
    _pending_marker_is_raised,
    _prospective_manifest,
    _root_marker_advances,
    _shrink_is_accounted,
    _structural_contributions,
    _structural_corpus,
)
from graphify.generation._transaction import _PublicationTransaction
from graphify.generation._types import (
    CodeUpdateRequest,
    Corpus,
    OperationFailed,
    PublicationRefused,
    TerminalOutcome,
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
    # Produced once, by the production manifest writer, and used for two
    # decisions: whether Corpus state advances at all, and which sources this
    # generation must disclose as awaiting reinterpretation. Deriving pending
    # state from the manifest this operation is about to publish is what keeps
    # the ledger and the manifest from encoding freshness differently.
    prospective, manifest_changed = _prospective_manifest(layout, manifest)

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
    # A live source whose interpretation the prospective manifest can no longer
    # vouch for is pending: its Semantic evidence still describes the source as
    # it was before the change, so it is retained and disclosed as stale rather
    # than dropped or silently presented as current.
    from graphify.detect import manifest_records_current_interpretation

    stale_semantic = {
        contribution.source
        for contribution in active
        if contribution.interpretation is _InterpretationKind.SEMANTIC
        and contribution.source in live
        and not manifest_records_current_interpretation(
            prospective,
            contribution.source,
        )
    }

    contributions = _structural_contributions(result) + _preserved_contributions(
        active,
        live=live,
        rederived=frozenset(owned),
        stale_semantic=stale_semantic,
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
        # A source system is never absent from a filesystem scan, so its
        # evidence disappearing would be a loss this run cannot explain.
        and not _is_external_source_identity(contribution.source)
    }

    # The ledger is the authoritative graph evidence, so an unchanged ledger
    # means the Graph generation itself did not change: leave graph.json and the
    # ledger exactly as published so a Corpus-state advance causes no output
    # churn. A missing or unreadable graph still has to be rematerialized.
    graph_changed = not graph_path.is_file() or not _ledger_matches(
        layout.path_for(_CanonicalArtifact.CONTRIBUTIONS),
        prepared,
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
        manifest=manifest if manifest_changed else None,
        root_marker=(
            str(root) if _root_marker_advances(layout, str(root)) else None
        ),
        # The compatibility pending marker is a projection of the state above,
        # never an independent claim. A Code update may raise it, but it never
        # lowers it: this operation performs no interpretation, so it can never
        # be the evidence that a source stopped being pending.
        needs_update=(
            True
            if stale_semantic
            and not _pending_marker_is_raised(
                layout.path_for(_CanonicalArtifact.NEEDS_UPDATE)
            )
            else None
        ),
        # Publishing graph evidence publishes a Raw graph generation, so the
        # clustered artifacts of the previous generation stop describing it and
        # must be retired rather than presented as current. Reclustering
        # republishes them from this generation's Source contributions.
        retire=_RETIRED_BY_RAW_PUBLICATION if graph_changed else frozenset(),
        protect_previous=graph_changed,
    )


def _preserved_contributions(
    active: tuple[_PreparedContribution, ...],
    *,
    live: set[str],
    rederived: frozenset[str],
    stale_semantic: set[str],
) -> tuple[_SourceContribution, ...]:
    """Carry forward active evidence this Code update must not replace.

    Structural evidence never survives: this run re-derives it for every source
    it owns, and a source it no longer owns — a departed path, or a document
    whose Semantic evidence now represents it — must not keep stale structural
    evidence either.

    Semantic evidence always survives for a live source, because Code update
    never reinterprets. That is custody, not a freshness claim: a source in
    ``stale_semantic`` keeps its evidence carrying an explicit stale marking, so
    the generation states plainly that the interpretation predates the file it
    describes. The marking is re-derived every run from the manifest this
    operation publishes, so it clears exactly when a Full extraction has proven
    the interpretation current again — never because a Code update ran.

    Provisional legacy evidence is replaced per source once real extraction
    covers it, while unattributed legacy evidence survives until a complete Full
    extraction or an explicit operator retirement retires it.

    Evidence a source *system* described — a database schema, say — survives
    too. Corpus discovery is authoritative about files, and a system that is not
    a file cannot be absent from a filesystem scan; only a Full extraction that
    asked that system again may replace what it said.
    """
    preserved: list[_SourceContribution] = []
    for contribution in active:
        if contribution.source == _UNATTRIBUTED_LEGACY_SOURCE or (
            _is_external_source_identity(contribution.source)
        ):
            preserved.append(_as_source_contribution(contribution))
            continue
        if contribution.source not in live:
            continue
        if contribution.interpretation is _InterpretationKind.STRUCTURAL:
            continue
        if contribution.provisional and contribution.source in rederived:
            continue
        preserved.append(
            _as_source_contribution(
                contribution,
                stale=(
                    contribution.source in stale_semantic
                    if contribution.interpretation is _InterpretationKind.SEMANTIC
                    else contribution.stale
                ),
            )
        )
    return tuple(preserved)
