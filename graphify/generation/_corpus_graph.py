"""Corpus-bound operation facade for canonical publication."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from graphify.generation._code_update import _execute_code_update
from graphify.generation._coordination import (
    _LATE_ARRIVAL_PASSES,
    _AcceptedRequest,
    _CoalescedRequest,
    _LeaseOutcome,
    _PolicyReplacement,
    _RequestCoordinator,
    _RequestedAuthority,
    _coalesce,
)
from graphify.generation._full_extraction import (
    _execute_full_extraction,
    _validate_sources,
)
from graphify.generation._publication import _Publication
from graphify.generation._publisher import _Publisher
from graphify.generation._types import (
    AlreadyCurrent,
    BuildPolicy,
    BuildPolicyRequest,
    ClearBuildPolicy,
    CodeUpdateRequest,
    Completion,
    Corpus,
    CorpusStateAdvanced,
    FullExtractionRequest,
    GenerationPublished,
    OperationFailed,
    Queued,
    ReclusteringRequest,
    ReplaceBuildPolicy,
    ReturnWhenQueued,
    TerminalOutcome,
    WaitUntilCovered,
)

_LEASE_HELD_ELSEWHERE = (
    "another process holds the Corpus executor lease; this request remains "
    "durably queued for the executor that covers it"
)

# What a Full-extraction executor genuinely covers today. It reconciles every
# Corpus source, so it does the work an ordinary Code update would — but it
# publishes a Raw generation, so it has not done a Reclustering's work and must
# not retire a queued request asking for one. The entry widens when Reclustering
# moves behind this module.
_FULL_EXTRACTION_COVERS = frozenset({"full-extraction", "code-update"})


class CorpusGraph:
    """Own canonical Graph-generation publication for one Corpus."""

    def __init__(self, corpus: Corpus) -> None:
        """Bind the owner to one source root and output directory."""
        # Freeze lexical identity now so a later chdir cannot retarget this
        # Corpus; absolute() deliberately preserves symlink spelling.
        self._corpus = Corpus(
            root=Path(corpus.root).absolute(),
            output=Path(corpus.output).absolute(),
        )

    @property
    def corpus(self) -> Corpus:
        """Return the immutable Corpus identity bound to this owner."""
        return self._corpus

    def full_extraction(
        self,
        request: FullExtractionRequest,
        *,
        completion: Completion = WaitUntilCovered(),
        _publication: _Publication | None = None,
    ) -> TerminalOutcome:
        """Reconcile every requested evidence source into one Graph generation.

        Called without ``_publication`` this owns the whole operation:
        authoritative discovery under the active or explicitly replaced Corpus
        policy, structural extraction, interpretation by the requested Semantic
        provider, collection from every requested source system, and publication
        — all under the one executor lease for the Corpus, and all completed
        before anything is committed.

        ``ReturnWhenQueued`` is refused: a request's evidence sources are live
        in-process objects, so a durable record of one could never be carried out
        by the executor that eventually drained it. The ``_publication`` handoff
        remains only for compatibility adapters that have not yet been rerouted.
        """
        if not isinstance(request, FullExtractionRequest):
            raise TypeError("full_extraction requires FullExtractionRequest")
        if _publication is not None:
            return self._complete(
                completion,
                _publication,
                operation="full-extraction",
            )
        self._validate_completion(completion)
        if isinstance(completion, ReturnWhenQueued):
            raise ValueError(
                "full-extraction cannot be durably queued: its evidence sources "
                "are in-process adapters that no other executor could carry out"
            )
        # Rejected before acceptance rather than after: an impossible request
        # must not become durable state something would later have to strand.
        _validate_sources(request)
        coordinator = _RequestCoordinator(self._corpus)
        accepted = coordinator.accept(
            "full-extraction",
            changed_paths=request.changed_paths,
            authority=_RequestedAuthority(
                force=request.force,
                policy_replacement=_policy_replacement(request.build_policy),
            ),
        )
        return self._execute(
            coordinator,
            accepted,
            operation="full-extraction",
            covers=_FULL_EXTRACTION_COVERS,
            run=lambda unit: _execute_full_extraction(
                self._corpus,
                _full_extraction_for(unit, request),
            ),
        )

    def code_update(
        self,
        request: CodeUpdateRequest,
        *,
        completion: Completion = WaitUntilCovered(),
        _publication: _Publication | None = None,
    ) -> TerminalOutcome:
        """Run one deterministic, LLM-free Code update for this Corpus.

        Called without ``_publication`` this owns the whole operation:
        authoritative discovery, structural extraction, reconciliation against
        the active Source-contribution ledger, and publication — all under the
        one executor lease for the Corpus. The ``_publication`` handoff remains
        only for compatibility adapters that have not yet been rerouted.
        """
        if not isinstance(request, CodeUpdateRequest):
            raise TypeError("code_update requires CodeUpdateRequest")
        if _publication is not None:
            return self._complete(completion, _publication, operation="code-update")
        self._validate_completion(completion)
        coordinator = _RequestCoordinator(self._corpus)
        accepted = coordinator.accept(
            "code-update",
            changed_paths=request.changed_paths,
            authority=_RequestedAuthority(force=request.force),
        )
        if isinstance(completion, ReturnWhenQueued):
            return Queued(request_id=accepted.request_id)
        return self._execute(
            coordinator,
            accepted,
            operation="code-update",
            covers=coordinator.subsumed_by("code-update"),
            run=lambda unit: _execute_code_update(
                self._corpus,
                _code_update_for(unit),
            ),
        )

    def reclustering(
        self,
        request: ReclusteringRequest,
        *,
        completion: Completion = WaitUntilCovered(),
        _publication: _Publication | None = None,
    ) -> TerminalOutcome:
        """Publish the candidate prepared by Reclustering or label compatibility.

        Like Full extraction, this is still adapter-prepared, so it refuses
        ``ReturnWhenQueued`` rather than durably accepting work no executor owns.
        """
        if not isinstance(request, ReclusteringRequest):
            raise TypeError("reclustering requires ReclusteringRequest")
        return self._complete(completion, _publication, operation="reclustering")

    def _execute(
        self,
        coordinator: _RequestCoordinator,
        accepted: _AcceptedRequest,
        *,
        operation: str,
        covers: frozenset[str],
        run: Callable[[_CoalescedRequest], TerminalOutcome],
    ) -> TerminalOutcome:
        """Execute one accepted request as this Corpus's single executor.

        The caller's own request is executed *through the queue* rather than
        beside it: it was made durable before the lease was contended for, so it
        is one of the accepted requests coalesced here. That is what lets this
        executor honor a ``force`` another process asked for, and it is why the
        returned outcome describes the work that covered this caller.

        ``covers`` is the request kinds this executor's finished work genuinely
        does, which is not always everything the operation nominally subsumes —
        an operation that publishes only part of a generation must not retire a
        queued request for the part it did not publish.
        """
        with coordinator.executor_lease(until_covered=accepted) as lease:
            if lease is _LeaseOutcome.COVERED:
                # Another executor published a generation covering this request
                # while we waited, which is exactly what waiting asked for.
                return AlreadyCurrent()
            if lease is _LeaseOutcome.UNAVAILABLE:
                return OperationFailed(_LEASE_HELD_ELSEWHERE)
            # Snapshot before executing: these operations re-derive every source
            # they own from authoritative discovery, so they cover every request
            # accepted before they started regardless of their path hints.
            # Requests that arrive mid-run are deliberately left for the passes
            # below.
            primary = self._cover_pending(
                coordinator,
                coordinator.pending(operations=covers),
                operation=operation,
                run=run,
            )
            if not _advances_corpus_state(primary):
                # A refusal or failure changed nothing, so the queued work stays
                # accepted for whichever executor manages to cover it next.
                return primary
            for _ in range(_LATE_ARRIVAL_PASSES):
                # Only this operation's own late arrivals. A lesser request that
                # turned up mid-run *could* be covered by re-running, but the
                # coalesced unit would then name that lesser operation, and
                # running this one for it would do — and charge for — work nobody
                # asked for. It stays durably accepted for its own executor.
                late = coordinator.pending(operations=frozenset({operation}))
                if not late:
                    break
                if not _advances_corpus_state(
                    self._cover_pending(
                        coordinator,
                        late,
                        operation=operation,
                        run=run,
                    )
                ):
                    # Whatever is still queued stays durably accepted rather than
                    # being retried until this process gives up on it.
                    break
            return primary

    def _cover_pending(
        self,
        coordinator: _RequestCoordinator,
        pending: tuple[_AcceptedRequest, ...],
        *,
        operation: str,
        run: Callable[[_CoalescedRequest], TerminalOutcome],
    ) -> TerminalOutcome:
        """Run the coalesced work for ``pending`` and retire what it covers.

        ``pending`` is filtered to the request kinds this executor covers, and
        coalescing folds them into one unit whenever their exclusive policies
        agree. A second unit means two queued requests disagreed about something
        that cannot be averaged — two different Corpus policies, say — and this
        executor cannot perform both. That is reported rather than worked around:
        performing one and retiring the other would claim coverage nothing
        earned, so both stay durably accepted.

        The unit is retired only after the operation advanced Corpus state, so a
        failure leaves every request it would have covered durably accepted. An
        empty queue means another executor covered this caller between acceptance
        and the lease.
        """
        units = _coalesce(pending)
        if not units:
            return AlreadyCurrent()
        if len(units) != 1 or units[0].operation != operation:
            return OperationFailed(
                f"the {operation} executor was given queued work it cannot "
                f"perform: {sorted({unit.operation for unit in units})}"
            )
        unit = units[0]
        outcome = run(unit)
        if _advances_corpus_state(outcome):
            coordinator.cover(unit.covers)
        return outcome

    def _complete(
        self,
        completion: Completion,
        publication: _Publication | None,
        *,
        operation: str,
    ) -> TerminalOutcome:
        """Validate caller completion policy and publish a prepared candidate."""
        self._validate_completion(completion)
        if isinstance(completion, ReturnWhenQueued):
            # Neither shape of this call can honestly be acknowledged as queued.
            # A prepared candidate lives in this process's memory, so no other
            # executor could carry it out; and no executor owns Full extraction
            # or Reclustering end to end yet, so a bare request would be recorded
            # durably and then covered by nobody. Both are caught before anything
            # is written rather than after.
            raise ValueError(
                f"{operation} cannot be durably queued: it is still prepared by a "
                "compatibility adapter rather than executed by CorpusGraph"
            )
        if publication is None:
            raise ValueError("the compatibility adapter did not prepare a publication")
        coordinator = _RequestCoordinator(self._corpus)
        with coordinator.executor_lease() as lease:
            if lease is not _LeaseOutcome.HELD:
                return OperationFailed(_LEASE_HELD_ELSEWHERE)
            # Deliberately covers nothing. A prepared candidate was built before
            # this call, so only the adapter that prepared it knows which moment
            # of the Corpus it describes — and therefore which accepted requests
            # it genuinely covers. An adapter that owns that window snapshots the
            # queue itself; guessing here would retire work nobody had done.
            return _Publisher(self._corpus).publish(publication, operation=operation)

    @staticmethod
    def _validate_completion(completion: Completion) -> None:
        """Reject a completion policy outside the agreed closed union."""
        if not isinstance(completion, (WaitUntilCovered, ReturnWhenQueued)):
            raise TypeError("completion must be WaitUntilCovered or ReturnWhenQueued")


def _policy_replacement(
    requested: BuildPolicyRequest | None,
) -> _PolicyReplacement | None:
    """Return the durable record of an explicitly requested Corpus policy.

    Clearing and replacing-with-the-defaults record identically, because they
    ask discovery for the same Corpus. The distinction that survives — whether
    the recorded policy is rewritten or retired — matters only to the process
    that asked, and that process executes its own request.
    """
    if isinstance(requested, ReplaceBuildPolicy):
        return _PolicyReplacement(
            excludes=tuple(requested.policy.excludes),
            gitignore=requested.policy.gitignore,
        )
    if isinstance(requested, ClearBuildPolicy):
        return _PolicyReplacement()
    return None


def _full_extraction_for(
    unit: _CoalescedRequest,
    request: FullExtractionRequest,
) -> FullExtractionRequest:
    """Return the Full extraction one coalesced unit asks this executor to run.

    Hints, ``force``, and the requested Corpus policy come from the unit, so
    authority another process asked for is honored rather than dropped. The
    evidence sources come from this process's own request: they are live
    adapters — an open provider, a DSN a caller resolved — that a durable record
    could not carry, which is also why a Full extraction is never acknowledged as
    merely queued.
    """
    replacement = unit.authority.policy_replacement
    if replacement is None:
        build_policy = None
    elif isinstance(request.build_policy, ClearBuildPolicy) and replacement == (
        _PolicyReplacement()
    ):
        build_policy = request.build_policy
    else:
        build_policy = ReplaceBuildPolicy(
            BuildPolicy(
                excludes=replacement.excludes,
                gitignore=replacement.gitignore,
            )
        )
    return FullExtractionRequest(
        sources=request.sources,
        build_policy=build_policy,
        changed_paths=tuple(Path(hint) for hint in unit.changed_paths),
        force=unit.authority.force,
    )


def _code_update_for(unit: _CoalescedRequest) -> CodeUpdateRequest:
    """Return the Code update one coalesced unit asks this executor to run.

    Only the hints and ``force`` survive the trip: those are the whole of what a
    Code-update request may carry, and coalescing has already refused to accept
    any other authority for this operation.
    """
    return CodeUpdateRequest(
        changed_paths=tuple(Path(hint) for hint in unit.changed_paths),
        force=unit.authority.force,
    )


def _advances_corpus_state(outcome: TerminalOutcome) -> bool:
    """Return whether an outcome means the Corpus is current with the request.

    Only these three outcomes prove a generation now describes the Corpus the
    queued requests were asking about; a refusal, failure, or cancellation must
    leave them accepted so the work is not silently dropped.
    """
    return isinstance(
        outcome,
        (GenerationPublished, CorpusStateAdvanced, AlreadyCurrent),
    )
