"""Deterministic Source-contribution ledger encoding and materialization."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping


# Ledger key for legacy evidence no source attribution could be recovered for.
# It is deliberately not a Corpus-relative path so it can never collide with a
# real source, and it survives until a complete Full extraction replaces it.
_UNATTRIBUTED_LEGACY_SOURCE = "@legacy/unattributed"


class _InterpretationKind(str, Enum):
    """Identify how one Corpus source produced graph evidence."""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    LEGACY_ATTRIBUTED = "legacy-attributed"
    LEGACY_UNATTRIBUTED = "legacy-unattributed"


@dataclass(frozen=True)
class _SourceContribution:
    """Carry one source's evidence before graph-wide deduplication.

    ``stale`` records that the live source has changed since this evidence was
    interpreted. It is custody metadata, not a quality judgement: the evidence
    remains authoritative and queryable until a successful reinterpretation
    replaces it.
    """

    source: Path | str
    interpretation: _InterpretationKind
    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    hyperedges: tuple[Mapping[str, Any], ...] = ()
    provisional: bool = False
    stale: bool = False


@dataclass(frozen=True)
class _PreparedContribution:
    """Hold one portable, serialization-ready Source contribution."""

    source: str
    interpretation: _InterpretationKind
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    hyperedges: tuple[dict[str, Any], ...]
    provisional: bool
    stale: bool = False


def _validate_graph_evidence(
    nodes,
    edges,
    hyperedges,
    *,
    description: str,
) -> None:
    """Reject graph evidence that cannot form the canonical node-link schema."""
    if not all(
        isinstance(node, Mapping) and isinstance(node.get("id"), str)
        for node in nodes
    ):
        raise ValueError(f"invalid node evidence in {description}")
    if not all(
        isinstance(edge, Mapping)
        and isinstance(edge.get("source"), str)
        and isinstance(edge.get("target"), str)
        for edge in edges
    ):
        raise ValueError(f"invalid edge evidence in {description}")
    if not all(
        isinstance(hyperedge, Mapping)
        and isinstance(hyperedge.get("id"), str)
        and isinstance(hyperedge.get("nodes"), list)
        and all(isinstance(member, str) for member in hyperedge["nodes"])
        for hyperedge in hyperedges
    ):
        raise ValueError(f"invalid hyperedge evidence in {description}")


def _prepare_contributions(
    contributions: tuple[_SourceContribution, ...],
    root: Path,
) -> tuple[_PreparedContribution, ...]:
    """Normalize contribution keys and evidence to portable root-relative paths."""
    prepared: list[_PreparedContribution] = []
    seen: set[tuple[str, _InterpretationKind]] = set()
    for contribution in contributions:
        # Reconciliation carries an existing unattributed legacy record forward
        # verbatim; its key is intentionally not a Corpus-relative path.
        source = (
            _UNATTRIBUTED_LEGACY_SOURCE
            if contribution.source == _UNATTRIBUTED_LEGACY_SOURCE
            else _relative_source_identity(contribution.source, root)
        )
        key = (source, contribution.interpretation)
        if key in seen:
            raise ValueError(
                "duplicate Source contribution key "
                f"({source!r}, {contribution.interpretation.value!r})"
            )
        seen.add(key)
        _validate_graph_evidence(
            contribution.nodes,
            contribution.edges,
            contribution.hyperedges,
            description=(
                f"Source contribution ({source!r}, "
                f"{contribution.interpretation.value!r})"
            ),
        )
        prepared.append(
            _PreparedContribution(
                source=source,
                interpretation=contribution.interpretation,
                nodes=_portable_items(contribution.nodes, root),
                edges=_portable_items(contribution.edges, root),
                hyperedges=_portable_items(contribution.hyperedges, root),
                provisional=contribution.provisional,
                stale=contribution.stale,
            )
        )
    return tuple(
        sorted(
            prepared,
            key=lambda contribution: (
                contribution.source,
                contribution.interpretation.value,
            ),
        )
    )


def _adopt_legacy_graph(path: Path, root: Path) -> tuple[_PreparedContribution, ...]:
    """Decompose a valid pre-ledger graph into provisional Source contributions."""
    from graphify.security import check_graph_file_size_cap

    check_graph_file_size_cap(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot validate legacy graph {path}: {exc}") from exc
    return _adopt_graph_payload(payload, root, description=str(path))


def _canonicalize_legacy_hyperedges(hyperedges: list[Any]) -> list[Any]:
    """Fold historical member aliases onto the canonical ``nodes`` key."""
    canonical: list[Any] = []
    for value in hyperedges:
        if not isinstance(value, Mapping):
            canonical.append(value)
            continue
        hyperedge = dict(value)
        if not isinstance(hyperedge.get("nodes"), list):
            for alias in ("members", "node_ids"):
                members = hyperedge.get(alias)
                if isinstance(members, list):
                    hyperedge["nodes"] = members
                    break
        hyperedge.pop("members", None)
        hyperedge.pop("node_ids", None)
        canonical.append(hyperedge)
    return canonical


def _adopt_graph_payload(
    payload: Mapping[str, Any],
    root: Path,
    *,
    description: str = "prepared graph",
) -> tuple[_PreparedContribution, ...]:
    """Decompose a validated compatibility graph into provisional contributions."""
    if not isinstance(payload, dict):
        raise ValueError(f"legacy graph must be a JSON object: {description}")
    nodes = payload.get("nodes")
    links_key = "links" if "links" in payload else "edges"
    edges = payload.get(links_key)
    hyperedges = payload.get("hyperedges", [])
    if (
        not isinstance(nodes, list)
        or not isinstance(edges, list)
        or not isinstance(hyperedges, list)
    ):
        raise ValueError(
            f"legacy graph has an invalid node-link shape: {description}"
        )
    hyperedges = _canonicalize_legacy_hyperedges(hyperedges)
    _validate_graph_evidence(
        nodes,
        edges,
        hyperedges,
        description=f"legacy graph {description}",
    )

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}

    def _append(bucket: str, item: Mapping[str, Any]) -> None:
        source = _legacy_source_identity(item.get("source_file"), root)
        group = grouped.setdefault(
            source,
            {"nodes": [], "edges": [], "hyperedges": []},
        )
        group[bucket].append(item)

    for node in nodes:
        _append("nodes", node)
    for edge in edges:
        _append("edges", edge)
    for hyperedge in hyperedges:
        _append("hyperedges", hyperedge)

    contributions = tuple(
        _SourceContribution(
            source=source,
            interpretation=(
                _InterpretationKind.LEGACY_UNATTRIBUTED
                if source == _UNATTRIBUTED_LEGACY_SOURCE
                else _InterpretationKind.LEGACY_ATTRIBUTED
            ),
            nodes=tuple(group["nodes"]),
            edges=tuple(group["edges"]),
            hyperedges=tuple(group["hyperedges"]),
            provisional=True,
        )
        for source, group in grouped.items()
    )
    return _prepare_legacy_contributions(contributions, root)


def _prepare_legacy_contributions(
    contributions: tuple[_SourceContribution, ...],
    root: Path,
) -> tuple[_PreparedContribution, ...]:
    """Prepare legacy records while preserving the explicit unattributed key."""
    prepared: list[_PreparedContribution] = []
    for contribution in contributions:
        source = str(contribution.source)
        if source != _UNATTRIBUTED_LEGACY_SOURCE:
            source = _relative_source_identity(source, root)
        prepared.append(
            _PreparedContribution(
                source=source,
                interpretation=contribution.interpretation,
                nodes=_portable_items(
                    contribution.nodes,
                    root,
                    allow_unattributed=True,
                ),
                edges=_portable_items(
                    contribution.edges,
                    root,
                    allow_unattributed=True,
                ),
                hyperedges=_portable_items(
                    contribution.hyperedges,
                    root,
                    allow_unattributed=True,
                ),
                provisional=True,
            )
        )
    return tuple(
        sorted(
            prepared,
            key=lambda contribution: (
                contribution.source,
                contribution.interpretation.value,
            ),
        )
    )


def _legacy_source_identity(source: Any, root: Path) -> str:
    """Return a portable legacy attribution or the explicit unattributed key."""
    if not isinstance(source, str) or not source.strip():
        return _UNATTRIBUTED_LEGACY_SOURCE
    try:
        return _relative_source_identity(source, root)
    except (OSError, ValueError):
        return _UNATTRIBUTED_LEGACY_SOURCE


def _portable_path(value: Path | str) -> Path:
    """Interpret either platform's separator spelling as one filesystem path."""
    return Path(str(value).replace("\\", "/"))


def _uses_foreign_absolute_syntax(value: Path | str, native_path: Path) -> bool:
    """Detect rooted path syntax the current host cannot safely relativize."""
    spelling = str(value).replace("\\", "/")
    posix_path = PurePosixPath(spelling)
    windows_path = PureWindowsPath(spelling)
    return not native_path.is_absolute() and (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    )


def _relative_source_identity(source: Path | str, root: Path) -> str:
    """Return one normalized source identity contained by the Corpus root."""
    lexical_root = _portable_path(root).absolute()
    candidate = _portable_path(source)
    if _uses_foreign_absolute_syntax(source, candidate):
        raise ValueError(f"Source contribution is outside the Corpus root: {source}")
    if not candidate.is_absolute():
        candidate = lexical_root / candidate
    lexical_candidate = candidate.absolute()
    try:
        relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"Source contribution is outside the Corpus root: {source}") from exc
    try:
        lexical_candidate.resolve().relative_to(lexical_root.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Source contribution is outside the Corpus root: {source}") from exc
    identity = relative.as_posix()
    if identity in {"", "."}:
        raise ValueError("Source contribution identity must name a Corpus source")
    return identity


def _portable_items(
    items: tuple[Mapping[str, Any], ...],
    root: Path,
    *,
    allow_unattributed: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Copy evidence and enforce portable ``source_file`` attributes."""
    portable: list[dict[str, Any]] = []
    for item in items:
        copied = copy.deepcopy(dict(item))
        source_file = copied.get("source_file")
        if source_file:
            try:
                copied["source_file"] = _relative_source_identity(
                    str(source_file),
                    root,
                )
            except (OSError, ValueError) as exc:
                if not allow_unattributed:
                    raise ValueError(
                        f"evidence source_file is outside the Corpus root: "
                        f"{source_file}"
                    ) from exc
                copied.pop("source_file", None)
        portable.append(copied)
    return tuple(sorted(portable, key=_canonical_item_key))


def _canonical_item_key(item: Mapping[str, Any]) -> str:
    """Return the deterministic ordering key used inside one source record."""
    return _canonical_line(item)


def _canonical_line(payload: Mapping[str, Any]) -> str:
    """Encode one ledger line so identical evidence always yields identical bytes."""
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _ledger_header_line() -> str:
    """Return the encoded schema header every contribution ledger opens with."""
    return _canonical_line({"schema": "graphify-source-contributions", "version": 1})


def _ledger_record_line(contribution: _PreparedContribution) -> str:
    """Return the encoded ledger line for one prepared Source contribution.

    ``stale`` is written only when true. The encoding stays a pure function of
    the record, so comparison remains deterministic, and a generation with no
    stale evidence keeps the byte-for-byte ledger earlier versions produced —
    which is what lets an existing Corpus adopt this field without a rewrite.
    """
    record: dict[str, Any] = {
        "source": contribution.source,
        "interpretation": contribution.interpretation.value,
        "provisional": contribution.provisional,
        "nodes": list(contribution.nodes),
        "edges": list(contribution.edges),
        "hyperedges": list(contribution.hyperedges),
    }
    if contribution.stale:
        record["stale"] = True
    return _canonical_line(record)


def _write_contribution_ledger(
    path: Path,
    contributions: tuple[_PreparedContribution, ...],
) -> None:
    """Atomically stream a deterministic JSON-lines contribution ledger."""
    from graphify.paths import _atomic_replace

    def _write(handle) -> None:
        handle.write(_ledger_header_line())
        handle.write("\n")
        for contribution in contributions:
            handle.write(_ledger_record_line(contribution))
            handle.write("\n")

    # Required generation state must compare byte-for-byte across Windows and
    # POSIX, so disable the platform newline translation used by ordinary text
    # artifacts.
    _atomic_replace(path, _write, newline="\n")


def _ledger_matches(
    path: Path,
    contributions: tuple[_PreparedContribution, ...],
) -> bool:
    """Return whether writing ``contributions`` would leave the ledger unchanged.

    Compares line by line against the same encoder the writer uses, so the
    ledger is never loaded whole just to answer whether it would change.
    """
    try:
        with path.open("r", encoding="utf-8", newline="\n") as handle:
            if handle.readline().rstrip("\n") != _ledger_header_line():
                return False
            for contribution in contributions:
                if handle.readline().rstrip("\n") != _ledger_record_line(contribution):
                    return False
            return handle.readline() == ""
    except OSError:
        return False


def _iter_contribution_ledger(path: Path) -> Iterator[_PreparedContribution]:
    """Yield validated contribution records without loading the ledger at once."""
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"empty Source-contribution ledger: {path}")
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Source-contribution ledger header: {path}") from exc
        if header != {"schema": "graphify-source-contributions", "version": 1}:
            raise ValueError(f"unsupported Source-contribution ledger header: {path}")

        seen: set[tuple[str, _InterpretationKind]] = set()
        for line_number, line in enumerate(handle, start=2):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid Source-contribution record at {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Source-contribution record must be an object at "
                    f"{path}:{line_number}"
                )
            try:
                source = record["source"]
                interpretation = _InterpretationKind(record["interpretation"])
                provisional = record["provisional"]
                nodes = record["nodes"]
                edges = record["edges"]
                hyperedges = record["hyperedges"]
                # Absent in ledgers written before per-source semantic freshness
                # existed. Those generations recorded no staleness at all, so
                # reading the omission as "not stale" is the truthful default
                # and keeps a pre-existing Corpus readable.
                stale = record.get("stale", False)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid Source-contribution record at {path}:{line_number}"
                ) from exc
            if (
                not isinstance(source, str)
                or not _source_identity_is_portable(source)
                or not isinstance(provisional, bool)
                or not isinstance(stale, bool)
                or not all(isinstance(bucket, list) for bucket in (nodes, edges, hyperedges))
            ):
                raise ValueError(
                    f"invalid Source-contribution record at {path}:{line_number}"
                )
            try:
                _validate_graph_evidence(
                    nodes,
                    edges,
                    hyperedges,
                    description=f"{path}:{line_number}",
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid Source-contribution record at {path}:{line_number}"
                ) from exc
            if not all(
                _evidence_item_is_portable(item)
                for bucket in (nodes, edges, hyperedges)
                for item in bucket
            ):
                raise ValueError(
                    f"invalid Source-contribution record at {path}:{line_number}"
                )
            key = (source, interpretation)
            if key in seen:
                raise ValueError(
                    f"duplicate Source-contribution key at {path}:{line_number}"
                )
            seen.add(key)
            yield _PreparedContribution(
                source=source,
                interpretation=interpretation,
                nodes=tuple(nodes),
                edges=tuple(edges),
                hyperedges=tuple(hyperedges),
                provisional=provisional,
                stale=stale,
            )


def _source_identity_is_portable(source: str) -> bool:
    """Return whether a ledger key is portable and root-relative."""
    if not source or source == "." or "\\" in source:
        return False
    posix_path = PurePosixPath(source)
    windows_path = PureWindowsPath(source)
    return (
        source == posix_path.as_posix()
        and not posix_path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and ".." not in posix_path.parts
    )


def _evidence_item_is_portable(item: Mapping[str, Any]) -> bool:
    """Return whether an evidence record avoids machine-specific source paths."""
    source_file = item.get("source_file")
    if not source_file:
        return True
    return isinstance(source_file, str) and _source_identity_is_portable(source_file)


def _semantic_contribution_sources(path: Path) -> set[str]:
    """Return source identities whose active ledger evidence is semantic."""
    return {
        contribution.source
        for contribution in _iter_contribution_ledger(path)
        if contribution.interpretation is _InterpretationKind.SEMANTIC
    }


def _stale_semantic_sources(path: Path) -> tuple[str, ...]:
    """Return the sorted sources whose Semantic evidence is disclosed as stale.

    Streams the ledger rather than materializing it: disclosure runs on every
    read path, and only the source identities are needed.
    """
    return tuple(
        sorted(
            contribution.source
            for contribution in _iter_contribution_ledger(path)
            if contribution.interpretation is _InterpretationKind.SEMANTIC
            and contribution.stale
        )
    )


def _dedupe_hyperedges(hyperedges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse shared IDs with the latest interpretation's evidence winning."""
    by_id: dict[str, dict[str, Any]] = {}
    for hyperedge in hyperedges:
        by_id[hyperedge["id"]] = hyperedge
    return list(by_id.values())


def _materialize_graph_data(
    contributions: tuple[_PreparedContribution, ...],
    *,
    multigraph: bool = False,
) -> dict[str, Any]:
    """Derive the deduplicated raw graph view from authoritative contributions."""
    from graphify.build import dedupe_edges, dedupe_nodes

    precedence = {
        _InterpretationKind.LEGACY_UNATTRIBUTED: 0,
        _InterpretationKind.LEGACY_ATTRIBUTED: 0,
        _InterpretationKind.STRUCTURAL: 1,
        _InterpretationKind.SEMANTIC: 2,
    }
    materialization_order = sorted(
        contributions,
        key=lambda contribution: (
            precedence[contribution.interpretation],
            contribution.source,
        ),
    )
    nodes = [
        copy.deepcopy(node)
        for contribution in materialization_order
        for node in contribution.nodes
    ]
    edges = [
        copy.deepcopy(edge)
        for contribution in materialization_order
        for edge in contribution.edges
    ]
    hyperedges = [
        copy.deepcopy(hyperedge)
        for contribution in materialization_order
        for hyperedge in contribution.hyperedges
    ]
    if multigraph:
        materialized_edges = edges
    else:
        # Interpretation priority is last-writer-wins, while the established
        # helper is first-writer-wins. Reversing around it preserves stable
        # output order and lets semantic evidence replace provisional/structural
        # attributes for the same connectivity identity.
        materialized_edges = list(reversed(dedupe_edges(list(reversed(edges)))))
    return {
        "directed": False,
        "multigraph": multigraph,
        "graph": {},
        "nodes": dedupe_nodes(nodes),
        "links": materialized_edges,
        "hyperedges": _dedupe_hyperedges(hyperedges),
    }
