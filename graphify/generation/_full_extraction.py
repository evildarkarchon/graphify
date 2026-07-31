"""Private Full extraction execution over every requested evidence source.

Full extraction is the interpreting half of the Graph-generation lifecycle. It
performs the same authoritative Corpus discovery a Code update does, then
gathers evidence from *every* source the request asked for — the Corpus
filesystem, an external Semantic provider, a PostgreSQL schema, a Cargo
workspace, Google Workspace exports — and commits once, when all of them have
finished. Nothing is staged while a source is still working, so a published
generation never describes some sources at one moment of the Corpus and the rest
at another.

Two custody rules govern what a run is allowed to replace:

* A source's contribution is replaced **atomically**. Either this run produced
  that source's complete evidence, in which case it supersedes what was there,
  or it did not, in which case the prior contribution stands untouched.
* Evidence a source failed to produce is never invented. A source whose
  interpretation did not complete keeps its last complete contribution — marked
  stale, because it now predates the file it describes — or keeps having none,
  and stays pending either way. Whatever fragment the attempt did produce belongs
  in the provider's own cache, never in the ledger.

Publishing the successful sources while failed ones stay stale and pending is
what makes partial progress *possible*; whether it is **permitted** is a
separate safety rule that gates this operation from outside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

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
from graphify.generation._layout import _PublicationLayout
from graphify.generation._manifest import _stamped_manifest_files
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
    _identity_or_none,
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
    BuildPolicy,
    BuildPolicyRequest,
    CargoSource,
    EvidenceSource,
    ClearBuildPolicy,
    Corpus,
    FullExtractionRequest,
    GoogleWorkspaceSource,
    OperationFailed,
    PostgresSource,
    PublicationRefused,
    ReplaceBuildPolicy,
    SemanticProvider,
    SemanticRequest,
    SemanticSource,
    SourceEvidence,
    TerminalOutcome,
)

_SourceT = TypeVar("_SourceT", bound=EvidenceSource)

# The file kinds an external Semantic provider interprets. Everything else in the
# Corpus is represented by deterministic structural extraction.
_SEMANTIC_FILE_TYPES = ("document", "paper", "image")

# How ``detect`` records a Google Workspace shortcut it could not turn into a
# readable document. Matched as a prefix on the reason it appends, because the
# shortcut path is what precedes it.
_GOOGLE_WORKSPACE_FAILURES = (
    "[Google Workspace export failed",
    "[Google Workspace export produced no readable text]",
)


class _SourceUnavailable(Exception):
    """A requested evidence source could not be collected.

    Raised rather than returned so no partial collection can be mistaken for a
    complete one: an unavailable source aborts the whole gathering step before
    anything is staged.
    """


def _execute_full_extraction(
    corpus: Corpus,
    request: FullExtractionRequest,
) -> TerminalOutcome:
    """Run one Full extraction end to end and publish through the owning module.

    Every requested evidence source is gathered before anything is staged, so a
    source that cannot answer leaves the active Graph generation exactly as it
    was rather than half-replaced.
    """
    # Recovery precedes every read of the active generation, not just the write:
    # reconciliation below decides what to retire from the active ledger and
    # graph, so it must never observe an interrupted publication's state.
    recovery = _PublicationTransaction.recover(corpus)
    if recovery is not None:
        return recovery
    try:
        planned = _plan_full_extraction(corpus, request)
    except _SourceUnavailable as exc:
        return OperationFailed(f"a requested evidence source could not complete: {exc}")
    except Exception as exc:
        # Preparation runs discovery, third-party parsers, external providers,
        # and evidence validation. A failure there is a terminal outcome of an
        # accepted operation, not a crash for adapters to interpret, and it
        # leaves the active Graph generation untouched because nothing has been
        # staged yet.
        return OperationFailed(f"could not prepare the Full extraction: {exc}")
    if isinstance(planned, PublicationRefused):
        return planned
    return _Publisher(corpus).publish(planned, operation="full-extraction")


def _plan_full_extraction(
    corpus: Corpus,
    request: FullExtractionRequest,
) -> _Publication | PublicationRefused:
    """Reconcile every requested source into the publication one run should make."""
    root = corpus.root
    output = corpus.output
    # Resolve active artifact locations exactly as the next transaction will, so
    # reconciliation reads the same files publication is going to replace.
    layout = _PublicationTransaction.active_layout(corpus)
    resolved = _resolve_build_policy(layout, request.build_policy)
    if isinstance(resolved, PublicationRefused):
        return resolved
    policy, recorded_policy, clear_policy = resolved

    from graphify.detect import detect

    google_workspace = _requested(request, GoogleWorkspaceSource) is not None
    detection = detect(
        root,
        extra_excludes=list(policy.excludes) or None,
        gitignore=policy.gitignore,
        # None defers to the ambient setting, which is what an unrequested source
        # means; True is only ever the request's own doing.
        google_workspace=True if google_workspace else None,
        cache_root=output.parent,
    )
    if detection.get("walk_errors"):
        # Discovery could not enumerate the whole Corpus, so absence is not
        # deletion evidence and the resulting generation would silently describe
        # a subset.
        return PublicationRefused(
            "Corpus discovery was incomplete "
            f"({len(detection['walk_errors'])} unreadable location(s)): "
            "refusing to replace the active Graph generation"
        )
    failed_exports = (
        _failed_google_workspace_exports(detection) if google_workspace else ()
    )

    graph_path = layout.path_for(_CanonicalArtifact.GRAPH)
    try:
        active = _active_evidence(layout, graph_path, root)
    except (OSError, ValueError) as exc:
        return PublicationRefused(
            f"cannot validate authoritative Source contributions: {exc}"
        )

    live = _live_source_identities(detection, root)
    semantic_corpus = _semantic_corpus(detection, root)
    provider = _semantic_provider(request)

    # Every requested source is gathered here, before a single artifact is
    # staged. A source that cannot answer raises out of this block, so the
    # publication below is only ever built from a complete gathering.
    interpreted = (
        _interpret(provider, semantic_corpus, corpus)
        if provider is not None and semantic_corpus
        else {}
    )
    uninterpreted = (
        frozenset(semantic_corpus) - frozenset(interpreted)
        if provider is not None
        else frozenset()
    )

    # A document this run is interpreting — or already carries evidence that
    # cannot be re-derived structurally — is represented by that interpretation.
    # Quick-scanning its headings as well would both represent it twice and
    # replace knowledge with structure.
    unreproducible = {
        contribution.source
        for contribution in active
        if contribution.interpretation is not _InterpretationKind.STRUCTURAL
    } | (set(semantic_corpus) if provider is not None else set())
    owned = _structural_corpus(detection, unreproducible, root)

    from graphify.extract import extract

    structural_result: dict[str, Any] = (
        extract(list(owned.values()), cache_root=output.parent, root=root)
        if owned
        else {"nodes": [], "edges": [], "hyperedges": []}
    )
    # Source systems answer in the same node/edge shape as structural
    # extraction and attribute their evidence to the manifest or address that
    # produced it, so one grouping pass covers all of them.
    system_result = _collect_source_systems(request, corpus)
    structural = _structural_contributions(
        _combined(structural_result, system_result)
    )
    semantic = _semantic_contributions(interpreted)
    # Keyed by source *and* interpretation kind, which is how the ledger itself
    # is keyed: producing a source's structural evidence replaces its structural
    # contribution, and says nothing about what its meaning was interpreted to be.
    replaced = {
        (str(contribution.source), contribution.interpretation)
        for contribution in structural + semantic
    }
    # Sources this run produced any evidence for. Provisional evidence adopted
    # for one of them described a source real extraction has now covered, so it
    # has been superseded whichever kind it was recorded under.
    covered = {source for source, _kind in replaced}

    manifest = _ManifestUpdate(
        files=_stamped_manifest_files(
            detection["files"],
            _interpreted_evidence(interpreted),
            root,
        ),
        kind="both",
        root=root,
        scan_corpus={
            path for group in detection["files"].values() for path in group
        },
        # Dispatched but not completely interpreted: without this the seed loop
        # would copy the prior semantic hash forward and mask the omission.
        clear_semantic={
            str(semantic_corpus[identity]) for identity in uninterpreted
        },
    )
    # Produced once, by the production manifest writer, and used for three
    # decisions: whether Corpus state advances at all, which retained evidence is
    # stale, and which sources this generation must disclose as awaiting
    # reinterpretation. Deriving all three from the manifest this operation is
    # about to publish is what keeps the ledger and the manifest from encoding
    # freshness differently.
    prospective, manifest_changed = _prospective_manifest(layout, manifest)

    from graphify.detect import manifest_records_current_interpretation

    stale_semantic = {
        contribution.source
        for contribution in active
        if contribution.interpretation is _InterpretationKind.SEMANTIC
        and contribution.source in live
        and (contribution.source, _InterpretationKind.SEMANTIC) not in replaced
        and not manifest_records_current_interpretation(
            prospective,
            contribution.source,
        )
    }
    # Pending is exactly the interpretation this run owed and did not deliver:
    # a source it was asked to interpret and could not, or one whose retained
    # evidence no longer describes the file. A document nobody asked to
    # interpret is not pending — it is structurally represented on purpose.
    pending = stale_semantic | set(uninterpreted)
    # "Complete" is a claim about interpretation, not about finishing without an
    # error: a run that never asked a provider anything has not interpreted the
    # Corpus's documents, however cleanly it ran, so it may neither clear pending
    # state nor retire the legacy evidence a complete interpretation accounts for.
    complete = (
        not uninterpreted
        and not failed_exports
        and (provider is not None or not semantic_corpus)
    )

    contributions = (
        structural
        + semantic
        + _preserved_contributions(
            active,
            live=live,
            replaced=replaced,
            covered=covered,
            stale_semantic=stale_semantic,
            # A failed export makes absence ambiguous: a shortcut that could not
            # be exported has no sidecar in this scan, and reconciling that as a
            # deletion would destroy evidence the Corpus still owns.
            deletion_is_authoritative=not failed_exports,
            retire_unattributed_legacy=complete,
        )
    )
    prepared = _prepare_contributions(contributions, root)
    graph_data = _materialize_graph_data(prepared)

    # Sources whose evidence this run is entitled to replace or retire: every
    # source it produced complete evidence for, plus every source the same
    # authoritative scan proved has left the Corpus. A graph that shrinks only
    # because of those is reconciled, not partial.
    accounted = frozenset(covered) | {
        contribution.source
        for contribution in active
        if contribution.source not in live
        and not failed_exports
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
    retire = set(_RETIRED_BY_RAW_PUBLICATION if graph_changed else ())
    if clear_policy:
        retire.add(_CanonicalArtifact.BUILD_CONFIG)
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
        build_config=recorded_policy,
        root_marker=(
            str(root) if _root_marker_advances(layout, str(root)) else None
        ),
        needs_update=_pending_marker(
            layout,
            # A shortcut that could not be exported is outstanding work too, but
            # it has no ledger identity to disclose per source — the sidecar it
            # would have produced does not exist — so it reaches the marker as
            # unnamed outstanding work rather than as a named pending source.
            pending=bool(pending) or bool(failed_exports),
            complete=complete,
        ),
        # Publishing graph evidence publishes a Raw graph generation, so the
        # clustered artifacts of the previous generation stop describing it and
        # must be retired rather than presented as current. Reclustering
        # republishes them from this generation's Source contributions.
        retire=frozenset(retire),
        protect_previous=graph_changed,
    )


def _resolve_build_policy(
    layout: _PublicationLayout,
    requested: BuildPolicyRequest | None,
) -> tuple[BuildPolicy, dict[str, Any] | None, bool] | PublicationRefused:
    """Return the policy to build under, what to record, and whether to clear it.

    Full extraction preserves the active Corpus policy by default: an ordinary
    run must not quietly re-shape the Corpus, because that would silently
    re-include paths an operator excluded or drop paths they kept. Only an
    explicit replacement or clearing changes it, and the presence of the request
    — not its contents — is what makes it one, so replacing with the documented
    defaults still counts as a replacement.
    """
    if isinstance(requested, ReplaceBuildPolicy):
        policy = requested.policy
        return (
            policy,
            {"excludes": list(policy.excludes), "gitignore": policy.gitignore},
            False,
        )
    if isinstance(requested, ClearBuildPolicy):
        # Clearing retires the record rather than writing the defaults into it,
        # so a later reader cannot tell a cleared Corpus from one that never had
        # a policy — which is exactly what "cleared" means.
        return BuildPolicy(), None, True
    active = _active_build_policy(layout)
    if isinstance(active, PublicationRefused):
        return active
    excludes, gitignore = active
    return BuildPolicy(excludes=tuple(excludes), gitignore=gitignore), None, False


def _validate_sources(request: FullExtractionRequest) -> None:
    """Reject a request naming one kind of evidence source more than once.

    Two Semantic providers or two DSNs are a contradiction rather than something
    to resolve by picking one. Raised as ``ValueError`` so a caller can refuse
    the request before it is durably accepted, rather than stranding a queue
    record nothing can carry out.
    """
    for kind in {type(source) for source in request.sources}:
        _requested(request, kind)


def _requested(
    request: FullExtractionRequest,
    kind: type[_SourceT],
) -> _SourceT | None:
    """Return the one requested source of ``kind``, or None when none was asked for.

    Raises ``ValueError`` when the request names ``kind`` more than once; see
    :func:`_validate_sources`, which is how a caller checks every kind at once.
    """
    matches = [source for source in request.sources if isinstance(source, kind)]
    if len(matches) > 1:
        raise ValueError(
            f"a Full extraction may request {kind.__name__} at most once"
        )
    return matches[0] if matches else None


def _semantic_provider(request: FullExtractionRequest) -> SemanticProvider | None:
    """Return the external Semantic provider this request asked to interpret with."""
    source = _requested(request, SemanticSource)
    return source.provider if source is not None else None


def _semantic_corpus(detection: Mapping[str, Any], root: Path) -> dict[str, Path]:
    """Map every source a Semantic provider would be asked about to its path."""
    corpus: dict[str, Path] = {}
    for file_type in _SEMANTIC_FILE_TYPES:
        for path in detection["files"].get(file_type, []):
            identity = _identity_or_none(path, root)
            if identity is not None:
                corpus[identity] = Path(path)
    return corpus


def _interpret(
    provider: SemanticProvider,
    semantic_corpus: Mapping[str, Path],
    corpus: Corpus,
) -> dict[str, SourceEvidence]:
    """Ask one provider to interpret the Corpus and keep only complete answers.

    An answer naming a source that was not requested, or one outside the Corpus,
    is dropped rather than admitted: a provider may only speak for the sources it
    was given, and evidence keyed to anything else could not be reconciled
    against discovery later.
    """
    try:
        interpretation = provider.interpret(
            SemanticRequest(
                root=corpus.root,
                output=corpus.output,
                sources=tuple(semantic_corpus[identity] for identity in sorted(semantic_corpus)),
            )
        )
    except Exception as exc:
        raise _SourceUnavailable(f"the Semantic provider failed: {exc}") from exc
    accepted: dict[str, SourceEvidence] = {}
    for source, evidence in interpretation.interpreted.items():
        identity = _identity_or_none(source, corpus.root)
        if identity in semantic_corpus:
            accepted[str(identity)] = evidence
    return accepted


def _semantic_contributions(
    interpreted: Mapping[str, SourceEvidence],
) -> tuple[_SourceContribution, ...]:
    """Return one complete Semantic contribution per interpreted source.

    Only sources the provider reported as completely interpreted reach this, so
    every contribution it returns is a whole replacement for that source's prior
    evidence rather than an addition to it.
    """
    return tuple(
        _SourceContribution(
            source=identity,
            interpretation=_InterpretationKind.SEMANTIC,
            nodes=_attributed(interpreted[identity].nodes, identity),
            edges=_attributed(interpreted[identity].edges, identity),
            hyperedges=_attributed(interpreted[identity].hyperedges, identity),
        )
        for identity in sorted(interpreted)
    )


def _attributed(
    items: Iterable[Mapping[str, Any]],
    identity: str,
) -> tuple[dict[str, Any], ...]:
    """Return evidence carrying the source attribution it was interpreted from.

    A provider that already attributes its evidence is left alone; one that does
    not gets the identity it was asked about, so every admitted item can be
    reconciled against discovery and accounted for when the graph shrinks.
    """
    attributed = []
    for item in items:
        copied = dict(item)
        if not copied.get("source_file"):
            copied["source_file"] = identity
        attributed.append(copied)
    return tuple(attributed)


def _interpreted_evidence(
    interpreted: Mapping[str, SourceEvidence],
) -> dict[str, list[dict[str, Any]]]:
    """Return the interpretation flattened into the shape the manifest reads.

    The manifest writer stamps a source as interpreted only when this run
    produced output attributed to it, which is the same "complete or nothing"
    rule the ledger applies — so both are derived from one set of evidence
    rather than from two independent judgements.
    """
    flattened: dict[str, list[dict[str, Any]]] = {
        "nodes": [],
        "edges": [],
        "hyperedges": [],
    }
    for identity in sorted(interpreted):
        evidence = interpreted[identity]
        flattened["nodes"].extend(_attributed(evidence.nodes, identity))
        flattened["edges"].extend(_attributed(evidence.edges, identity))
        flattened["hyperedges"].extend(_attributed(evidence.hyperedges, identity))
    return flattened


def _combined(*results: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Concatenate several extraction results into one node/edge/hyperedge set."""
    combined: dict[str, list[Any]] = {"nodes": [], "edges": [], "hyperedges": []}
    for result in results:
        for bucket in combined:
            combined[bucket].extend(result.get(bucket) or ())
    return combined


def _collect_source_systems(
    request: FullExtractionRequest,
    corpus: Corpus,
) -> dict[str, list[Any]]:
    """Collect evidence from every requested source system, or raise.

    Each system is asked in turn and all of them must answer, because a
    generation missing one requested system's evidence would be indistinguishable
    from one where that system genuinely has nothing to say.
    """
    results: list[Mapping[str, Any]] = []
    postgres = _requested(request, PostgresSource)
    if postgres is not None:
        from graphify.pg_introspect import introspect_postgres

        try:
            results.append(introspect_postgres(postgres.dsn))
        except Exception as exc:
            raise _SourceUnavailable(f"PostgreSQL: {exc}") from exc
    cargo = _requested(request, CargoSource)
    if cargo is not None:
        from graphify.cargo_introspect import introspect_cargo

        try:
            results.append(introspect_cargo(corpus.root))
        except Exception as exc:
            raise _SourceUnavailable(f"Cargo: {exc}") from exc
    return _combined(*results)


def _failed_google_workspace_exports(detection: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the Google Workspace shortcuts discovery could not export."""
    return tuple(
        entry
        for entry in detection.get("skipped_sensitive", [])
        if isinstance(entry, str)
        and any(failure in entry for failure in _GOOGLE_WORKSPACE_FAILURES)
    )


def _preserved_contributions(
    active: tuple[_PreparedContribution, ...],
    *,
    live: set[str],
    replaced: set[tuple[str, _InterpretationKind]],
    covered: set[str],
    stale_semantic: set[str],
    deletion_is_authoritative: bool,
    retire_unattributed_legacy: bool,
) -> tuple[_SourceContribution, ...]:
    """Carry forward active evidence this Full extraction did not replace.

    A ``(source, kind)`` in ``replaced`` had its complete evidence produced by
    this run, so the prior contribution is superseded rather than merged with —
    that is what makes replacement atomic, and keying it by kind as the ledger
    does is what keeps re-deriving a source's structure from claiming anything
    about what its meaning was interpreted to be. Everything else is judged on
    whether this run had the authority to retire it: a source the authoritative
    scan proved has left the Corpus is dropped, and a source that is still live
    keeps the evidence this run could not reproduce.

    Retained Semantic evidence carries an explicit stale marking whenever the
    manifest this run publishes can no longer vouch for the interpretation, so
    the generation states plainly that the evidence predates the file it
    describes rather than quietly presenting it as current.

    Unattributed legacy evidence is the one contribution a complete run retires:
    it exists only because a pre-ledger graph could not say which source produced
    it, and a run that reconciled every requested source has now accounted for
    all of it. An incomplete run keeps it, because it has not.
    """
    preserved: list[_SourceContribution] = []
    for contribution in active:
        source = contribution.source
        if source == _UNATTRIBUTED_LEGACY_SOURCE:
            if not retire_unattributed_legacy:
                preserved.append(_as_source_contribution(contribution))
            continue
        if (source, contribution.interpretation) in replaced:
            continue
        if _is_external_source_identity(source):
            # A source system is not a file, so a filesystem scan can neither
            # confirm nor deny it. Only asking that system again may replace what
            # it said, which the check above already did.
            preserved.append(_as_source_contribution(contribution))
            continue
        if deletion_is_authoritative and source not in live:
            continue
        if contribution.interpretation is _InterpretationKind.STRUCTURAL:
            # Structural evidence for a source this run still owns was re-derived
            # above; for one it no longer owns, keeping it would describe the
            # Corpus as it used to be.
            continue
        if contribution.provisional and source in covered:
            # Evidence adopted from a pre-ledger graph, for a source real
            # extraction has now covered under some other kind. It was always a
            # guess about attribution, so the real evidence supersedes it.
            continue
        preserved.append(
            _as_source_contribution(
                contribution,
                stale=(
                    source in stale_semantic
                    if contribution.interpretation is _InterpretationKind.SEMANTIC
                    else contribution.stale
                ),
            )
        )
    return tuple(preserved)


def _pending_marker(
    layout: _PublicationLayout,
    *,
    pending: bool,
    complete: bool,
) -> bool | None:
    """Return the compatibility pending marker this run should publish.

    The marker is a projection of authoritative generation state, never an
    independent claim, so it is raised whenever interpretation is outstanding.
    Lowering it needs more: only a run that completed every requested source and
    left nothing pending has the evidence that the work is actually done. ``None``
    leaves whatever the previous generation recorded alone.
    """
    raised = _pending_marker_is_raised(layout.path_for(_CanonicalArtifact.NEEDS_UPDATE))
    if pending:
        return True if not raised else None
    if complete and raised:
        return False
    return None
