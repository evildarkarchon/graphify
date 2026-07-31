"""Corpus-bound operation facade for canonical publication."""

from __future__ import annotations

from pathlib import Path

from graphify.generation._code_update import _execute_code_update
from graphify.generation._coordination import (
    _LATE_ARRIVAL_PASSES,
    _AcceptedRequest,
    _CoalescedRequest,
    _LeaseOutcome,
    _RequestCoordinator,
    _RequestedAuthority,
    _coalesce,
)
from graphify.generation._publication import _Publication
from graphify.generation._publisher import _Publisher
from graphify.generation._types import (
    AlreadyCurrent,
    CodeUpdateRequest,
    Completion,
    Corpus,
    CorpusStateAdvanced,
    FullExtractionRequest,
    GenerationPublished,
    OperationFailed,
    Queued,
    ReclusteringRequest,
    ReturnWhenQueued,
    TerminalOutcome,
    WaitUntilCovered,
)

_LEASE_HELD_ELSEWHERE = (
    "another process holds the Corpus executor lease; this request remains "
    "durably queued for the executor that covers it"
)


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
        """Publish the candidate prepared by the current Full extraction adapter.

        ``_publication`` is the private migration handoff. Invalid calls fail
        before acceptance; lifecycle preparation moves behind this method in a
        later slice without changing the public request or outcome types. Until
        it does, ``ReturnWhenQueued`` is refused rather than acknowledged: no
        executor owns this operation, so a durable record would never be covered.
        """
        if not isinstance(request, FullExtractionRequest):
            raise TypeError("full_extraction requires FullExtractionRequest")
        return self._complete(completion, _publication, operation="full-extraction")

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
        return self._execute_code_update(coordinator, accepted)

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

    def _execute_code_update(
        self,
        coordinator: _RequestCoordinator,
        accepted: _AcceptedRequest,
    ) -> TerminalOutcome:
        """Execute one accepted Code update as this Corpus's single executor.

        The caller's own request is executed *through the queue* rather than
        beside it: it was made durable before the lease was contended for, so it
        is one of the accepted requests coalesced here. That is what lets this
        executor honor a ``force`` another process asked for, and it is why the
        returned outcome describes the work that covered this caller.
        """
        kinds = coordinator.subsumed_by("code-update")
        with coordinator.executor_lease(until_covered=accepted) as lease:
            if lease is _LeaseOutcome.COVERED:
                # Another executor published a generation covering this request
                # while we waited, which is exactly what waiting asked for.
                return AlreadyCurrent()
            if lease is _LeaseOutcome.UNAVAILABLE:
                return OperationFailed(_LEASE_HELD_ELSEWHERE)
            # Snapshot before executing: a Code update re-derives every source it
            # owns from authoritative discovery, so it covers every request
            # accepted before it started regardless of their path hints. Requests
            # that arrive mid-run are deliberately left for the passes below.
            primary = self._cover_pending(
                coordinator,
                coordinator.pending(operations=kinds),
            )
            if not _advances_corpus_state(primary):
                # A refusal or failure changed nothing, so the queued work stays
                # accepted for whichever executor manages to cover it next.
                return primary
            for _ in range(_LATE_ARRIVAL_PASSES):
                late = coordinator.pending(operations=kinds)
                if not late:
                    break
                if not _advances_corpus_state(
                    self._cover_pending(coordinator, late)
                ):
                    # Whatever is still queued stays durably accepted rather than
                    # being retried until this process gives up on it.
                    break
            return primary

    def _cover_pending(
        self,
        coordinator: _RequestCoordinator,
        pending: tuple[_AcceptedRequest, ...],
    ) -> TerminalOutcome:
        """Run the coalesced Code-update work for ``pending`` and retire what it covers.

        ``pending`` is filtered to the request kinds a Code update covers, and
        every such request is mutually compatible — hints and ``force`` both
        union, and no other authority may be accepted for this operation — so
        coalescing yields exactly one unit. That invariant is asserted rather
        than worked around: a second unit would mean this executor had been
        handed work it cannot perform, and retiring it would claim coverage
        nothing earned.

        The unit is retired only after the operation advanced Corpus state, so a
        failure leaves every request it would have covered durably accepted. An
        empty queue means another executor covered this caller between acceptance
        and the lease.
        """
        units = _coalesce(pending)
        if not units:
            return AlreadyCurrent()
        if len(units) != 1 or units[0].operation != "code-update":
            return OperationFailed(
                "the Code-update executor was given queued work it cannot "
                f"perform: {sorted({unit.operation for unit in units})}"
            )
        unit = units[0]
        outcome = _execute_code_update(self._corpus, _code_update_for(unit))
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
