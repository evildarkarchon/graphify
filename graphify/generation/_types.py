"""Closed request, completion, and terminal outcome types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable


@dataclass(frozen=True)
class Corpus:
    """Identify one requested source root and canonical output location."""

    root: Path
    output: Path


@dataclass(frozen=True)
class WaitUntilCovered:
    """Complete the call only after the accepted request has a terminal outcome."""


@dataclass(frozen=True)
class ReturnWhenQueued:
    """Return after a background request has been durably accepted."""


Completion: TypeAlias = WaitUntilCovered | ReturnWhenQueued


@dataclass(frozen=True)
class SourceEvidence:
    """The graph evidence one source contributed, before deduplication.

    This is the shape every evidence source answers in, whatever it is: a
    document an external Semantic provider interpreted, a schema a database
    described, a manifest a package manager reported. Keeping one shape is what
    lets ``CorpusGraph`` replace a source's contribution atomically without
    knowing which kind of source produced it.
    """

    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    hyperedges: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SemanticRequest:
    """The Corpus sources one Full extraction asks a provider to interpret.

    ``root`` is the Corpus root every returned ``source_file`` is relativized
    against, and ``output`` is where a provider may keep its own accelerator
    cache. ``sources`` is the complete set the provider was asked about; a
    provider is free to serve any of them from its cache.
    """

    root: Path
    output: Path
    sources: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SemanticInterpretation:
    """A Semantic provider's answer for the sources it was asked about.

    ``interpreted`` carries **only** sources whose interpretation completed. A
    requested source absent from it was not completely interpreted — whether the
    provider failed, was cut short, or produced a truncated fragment — and
    ``CorpusGraph`` therefore keeps that source's prior evidence, leaves it
    pending, and lets the provider keep any fragment in its own cache. Reporting
    partial work by omission rather than by a flag is what makes "complete
    interpretation" impossible to claim by accident.
    """

    interpreted: Mapping[str, SourceEvidence] = field(default_factory=dict)


@runtime_checkable
class SemanticProvider(Protocol):
    """Interpret Corpus documents into graph evidence.

    This is one of the two seams the owning module accepts from outside itself,
    because a Semantic provider is a genuinely external system. Everything else
    a Full extraction touches — discovery, caches, structural extraction,
    publication — stays an implementation detail.
    """

    def interpret(self, request: SemanticRequest) -> SemanticInterpretation:
        """Return the complete interpretations of ``request.sources``."""
        ...


@dataclass(frozen=True)
class SemanticSource:
    """Request interpretation of the Corpus's documents, papers, and images."""

    provider: SemanticProvider


@dataclass(frozen=True)
class PostgresSource:
    """Request PostgreSQL schema evidence.

    ``dsn`` is ``None`` when the connection is described by the standard ``PG*``
    environment variables rather than by the request.
    """

    dsn: str | None = None


@dataclass(frozen=True)
class CargoSource:
    """Request Cargo workspace evidence for the Corpus root."""


@dataclass(frozen=True)
class GoogleWorkspaceSource:
    """Admit Google Workspace shortcuts as Corpus documents.

    Unlike the other sources this one contributes through discovery: a shortcut
    is exported to a Markdown sidecar that then takes part in the Corpus like any
    other document. A shortcut whose export fails is a source that did not
    complete, so its prior evidence is kept and left pending rather than being
    reconciled away as a deletion.
    """


EvidenceSource: TypeAlias = (
    SemanticSource | PostgresSource | CargoSource | GoogleWorkspaceSource
)


@dataclass(frozen=True)
class BuildPolicy:
    """The Corpus-shaping policy a Graph generation is built under."""

    excludes: tuple[str, ...] = ()
    gitignore: bool = True


@dataclass(frozen=True)
class ReplaceBuildPolicy:
    """Explicit authority to record a different Corpus build policy.

    The presence of this request, not its contents, is what distinguishes
    "preserve the recorded policy" from "the Corpus is shaped differently now",
    so replacing with the documented defaults is still a replacement.
    """

    policy: BuildPolicy = BuildPolicy()


@dataclass(frozen=True)
class ClearBuildPolicy:
    """Explicit authority to return the Corpus to the documented defaults."""


BuildPolicyRequest: TypeAlias = ReplaceBuildPolicy | ClearBuildPolicy


@dataclass(frozen=True)
class FullExtractionRequest:
    """Identify a Full extraction operation and the evidence it requests.

    ``sources`` names the canonical evidence sources beyond the Corpus
    filesystem, which is always reconciled. Every requested source must complete
    before anything is committed, so a generation never describes some sources at
    one moment of the Corpus and the rest at another.

    ``build_policy`` is absent for the ordinary case: Full extraction preserves
    the active Corpus policy unless replacement or clearing is asked for
    explicitly. ``changed_paths`` is an optimization hint only — authoritative
    discovery decides which sources are live — and ``force`` authorizes replacing
    the active graph with a smaller one without weakening any other rule.
    """

    sources: tuple[EvidenceSource, ...] = ()
    build_policy: BuildPolicyRequest | None = None
    changed_paths: tuple[Path, ...] = ()
    force: bool = False


@dataclass(frozen=True)
class CodeUpdateRequest:
    """Identify an LLM-free Code update operation.

    ``changed_paths`` is an optimization hint only: authoritative discovery, not
    the hint, decides which sources are live. ``force`` authorizes replacing the
    active graph with a smaller one; it never weakens a fail-closed publication
    rule.
    """

    changed_paths: tuple[Path, ...] = ()
    force: bool = False


@dataclass(frozen=True)
class ReclusteringRequest:
    """Identify a Reclustering operation, including label compatibility calls."""


@dataclass(frozen=True)
class GenerationPublished:
    """A request published a changed materialized graph."""

    changed_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusStateAdvanced:
    """Corpus state advanced without changing the materialized graph."""

    changed_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlreadyCurrent:
    """The active generation already covers the request."""


@dataclass(frozen=True)
class Queued:
    """A background request was durably accepted for later execution.

    ``request_id`` identifies the durable record the acknowledgment refers to, so
    an integrating caller can correlate a queued submission with the executor
    that eventually covers it without reading printed output.
    """

    request_id: str


@dataclass(frozen=True)
class PublicationRefused:
    """Safety validation refused to replace the active generation."""

    reason: str


@dataclass(frozen=True)
class OperationFailed:
    """An accepted operation failed before it could publish."""

    reason: str


@dataclass(frozen=True)
class Cancelled:
    """A foreground request was cancelled before canonical commit."""


TerminalOutcome: TypeAlias = (
    GenerationPublished
    | CorpusStateAdvanced
    | AlreadyCurrent
    | Queued
    | PublicationRefused
    | OperationFailed
    | Cancelled
)
