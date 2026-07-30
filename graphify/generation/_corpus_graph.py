"""Corpus-bound operation facade for canonical publication."""

from __future__ import annotations

from pathlib import Path

from graphify.generation._code_update import _execute_code_update
from graphify.generation._publication import _Publication
from graphify.generation._publisher import _Publisher
from graphify.generation._types import (
    CodeUpdateRequest,
    Completion,
    Corpus,
    FullExtractionRequest,
    ReclusteringRequest,
    ReturnWhenQueued,
    TerminalOutcome,
    WaitUntilCovered,
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
        later slice without changing the public request or outcome types.
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
        the active Source-contribution ledger, and publication. The
        ``_publication`` handoff remains only for compatibility adapters that
        have not yet been rerouted.
        """
        if not isinstance(request, CodeUpdateRequest):
            raise TypeError("code_update requires CodeUpdateRequest")
        if _publication is not None:
            return self._complete(completion, _publication, operation="code-update")
        self._validate_completion(completion)
        return _execute_code_update(self._corpus, request)

    def reclustering(
        self,
        request: ReclusteringRequest,
        *,
        completion: Completion = WaitUntilCovered(),
        _publication: _Publication | None = None,
    ) -> TerminalOutcome:
        """Publish the candidate prepared by Reclustering or label compatibility."""
        if not isinstance(request, ReclusteringRequest):
            raise TypeError("reclustering requires ReclusteringRequest")
        return self._complete(completion, _publication, operation="reclustering")

    def _complete(
        self,
        completion: Completion,
        publication: _Publication | None,
        *,
        operation: str,
    ) -> TerminalOutcome:
        """Validate caller completion policy and synchronously publish a candidate."""
        self._validate_completion(completion)
        if publication is None:
            raise ValueError("the compatibility adapter did not prepare a publication")
        return _Publisher(self._corpus).publish(publication, operation=operation)

    @staticmethod
    def _validate_completion(completion: Completion) -> None:
        """Reject a completion policy outside the agreed closed union."""
        if not isinstance(completion, (WaitUntilCovered, ReturnWhenQueued)):
            raise TypeError("completion must be WaitUntilCovered or ReturnWhenQueued")
