"""Canonical and compatibility artifact location policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from graphify.generation._publication import _CanonicalArtifact


@dataclass(frozen=True)
class _PublicationLayout:
    """Resolve owner-controlled artifact paths for one publication.

    Legacy generated runbooks placed a few sidecars beside ``graphify-out``.
    Private overrides preserve those invocation-root paths during the custody
    cutover while keeping validation and all filesystem writes inside
    ``CorpusGraph``.
    """

    root: Path
    output: Path
    overrides: Mapping[_CanonicalArtifact, Path]

    def path_for(self, artifact: _CanonicalArtifact) -> Path:
        """Return a validated path for an artifact in this Corpus."""
        if artifact not in self.overrides:
            return self.output / artifact.value

        compatibility_root = self.output.parent
        candidate = Path(self.overrides[artifact])
        if not candidate.is_absolute():
            candidate = compatibility_root / candidate
        resolved_root = compatibility_root.resolve()
        resolved_candidate = candidate.resolve()
        if (
            resolved_candidate.name != artifact.value
            or resolved_candidate.parent != resolved_root
        ):
            raise ValueError(f"invalid compatibility path for {artifact.value}: {candidate}")
        return candidate
