"""Corpus-bound ownership of canonical Graph-generation publication.

This migration slice keeps the existing extraction, clustering, and labeling
algorithms in place while making this package their sole canonical publisher.
"""

from pathlib import Path
from typing import Sequence

from graphify.generation._contributions import _stale_semantic_sources
from graphify.generation._corpus_graph import CorpusGraph
from graphify.generation._publication import _CanonicalArtifact
from graphify.generation._transaction import _PublicationTransaction
from graphify.generation._types import (
    AlreadyCurrent,
    BuildPolicy,
    BuildPolicyRequest,
    Cancelled,
    CargoSource,
    ClearBuildPolicy,
    CodeUpdateRequest,
    Completion,
    Corpus,
    CorpusDiscovered,
    CorpusStateAdvanced,
    EvidenceCollected,
    EvidenceOrigin,
    EvidenceSource,
    FullExtractionRequest,
    GenerationPublished,
    GoogleWorkspaceSource,
    Observation,
    ObservationAdapter,
    OperationFailed,
    PostgresSource,
    PublicationRefused,
    PublicationStarted,
    Queued,
    ReclusteringRequest,
    ReplaceBuildPolicy,
    ReturnWhenQueued,
    SemanticInterpretation,
    SemanticProvider,
    SemanticRequest,
    SemanticSource,
    SourceEvidence,
    SourcesInterpreted,
    TerminalOutcome,
    WaitUntilCovered,
    covers_the_request,
)

def stale_semantic_sources(output: Path | str) -> tuple[str, ...]:
    """Return the Corpus sources whose Semantic evidence is stale, sorted.

    Query, report, and agent-guidance readers use this to disclose evidence that
    is still authoritative but describes a source as it was before its last
    change. The answer comes from the active Graph generation's own
    Source-contribution ledger, so no reader re-derives freshness for itself.

    Best effort by design: a Corpus with no published generation, or one whose
    ledger cannot be validated, discloses nothing rather than failing a read.
    Publication and the operations themselves refuse an unreadable ledger.
    """
    try:
        ledger = _PublicationTransaction.active_artifact_path(
            Path(output),
            _CanonicalArtifact.CONTRIBUTIONS,
        )
        if not ledger.is_file():
            return ()
        return _stale_semantic_sources(ledger)
    except (OSError, ValueError):
        return ()


def stale_semantic_disclosure(
    sources: Sequence[str],
    *,
    limit: int = 5,
) -> str:
    """Return the sentence every reader uses to disclose Stale semantic evidence.

    One wording and one truncation rule, owned here beside the state it
    describes, so a query banner and a report section cannot drift into
    describing the same Graph generation differently. ``limit`` caps how many
    sources are named before the remainder is counted. Empty when nothing is
    stale, so a caller can use the result as its own "disclose anything?" test.
    """
    if not sources:
        return ""
    named = ", ".join(sources[:limit])
    more = f" (+{len(sources) - limit} more)" if len(sources) > limit else ""
    return (
        f"{len(sources)} source(s) changed after their last interpretation — "
        f"{named}{more}. Evidence from them is the last complete "
        "interpretation, not a description of the file as it is now. Run "
        "`graphify extract` to reinterpret them; `graphify update .` cannot "
        "clear this state."
    )


__all__ = [
    "AlreadyCurrent",
    "BuildPolicy",
    "BuildPolicyRequest",
    "Cancelled",
    "CargoSource",
    "ClearBuildPolicy",
    "CodeUpdateRequest",
    "Completion",
    "Corpus",
    "CorpusDiscovered",
    "CorpusGraph",
    "CorpusStateAdvanced",
    "EvidenceCollected",
    "EvidenceOrigin",
    "EvidenceSource",
    "FullExtractionRequest",
    "GenerationPublished",
    "GoogleWorkspaceSource",
    "Observation",
    "ObservationAdapter",
    "OperationFailed",
    "PostgresSource",
    "PublicationRefused",
    "PublicationStarted",
    "Queued",
    "ReclusteringRequest",
    "ReplaceBuildPolicy",
    "ReturnWhenQueued",
    "SemanticInterpretation",
    "SemanticProvider",
    "SemanticRequest",
    "SemanticSource",
    "SourceEvidence",
    "SourcesInterpreted",
    "TerminalOutcome",
    "WaitUntilCovered",
    "covers_the_request",
    "stale_semantic_disclosure",
    "stale_semantic_sources",
]
