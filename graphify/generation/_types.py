"""Closed request, completion, and terminal outcome types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


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
class FullExtractionRequest:
    """Identify a Full extraction operation."""


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
