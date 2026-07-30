"""Manifest inputs prepared for owner-controlled publication."""

from __future__ import annotations

from pathlib import Path


def _stamped_manifest_files(
    files_by_type: dict[str, list[str]],
    sem_result: dict,
    root: Path,
    partial_source_files: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return files safe to stamp in a generation manifest.

    Only stamp semantic files that actually produced output (cache hit or fresh
    extraction). Files whose chunk failed have no ``source_file`` entry in
    ``sem_result``; leaving their semantic hash empty makes incremental
    detection re-queue them (#933).

    A file in ``partial_source_files`` did produce output this run, but only a
    truncated fragment. It remains unstamped so the warm incremental path
    retries it instead of treating an incomplete node set as final.

    Both sides of the membership test resolve against the scan ``root``
    (#1897). Hyperedges count as output because they carry their own
    ``source_file`` and are persisted by the semantic cache (#1920).
    """
    root = Path(root)

    def _resolve(value: str) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        try:
            return p.resolve()
        except (OSError, RuntimeError):
            return p

    sem_extracted: set[Path] = set()
    for coll in ("nodes", "edges", "hyperedges"):
        for item in sem_result.get(coll, []):
            sf = item.get("source_file", "")
            if sf:
                sem_extracted.add(_resolve(sf))
    partial_resolved = {_resolve(p) for p in (partial_source_files or set())}
    sem_types = {"document", "paper", "image"}
    return {
        ftype: [
            f
            for f in flist
            if ftype not in sem_types
            or (_resolve(f) in sem_extracted and _resolve(f) not in partial_resolved)
        ]
        for ftype, flist in files_by_type.items()
    }
