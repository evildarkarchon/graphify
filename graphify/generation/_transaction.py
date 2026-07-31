"""Durable staging, promotion, and recovery for one Graph generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from graphify.generation._contributions import (
    _adopt_legacy_graph,
    _iter_contribution_ledger,
    _materialize_graph_data,
    _write_contribution_ledger,
)
from graphify.generation._layout import _PublicationLayout
from graphify.generation._publication import _CanonicalArtifact
from graphify.generation._types import Corpus, PublicationRefused

_COMPLETION_SCHEMA = "graphify-generation-completion"
_JOURNAL_SCHEMA = "graphify-publication-journal"
_SCHEMA_VERSION = 1
_JOURNAL_NAME = ".graphify_publication.json"
_WORKSPACE_NAME = ".graphify_publication"
_PHASES = frozenset({"preparing", "prepared", "promoting", "committed"})
_JSON_ARTIFACTS = frozenset(
    {
        _CanonicalArtifact.GRAPH,
        _CanonicalArtifact.ANALYSIS,
        _CanonicalArtifact.LABELS,
        _CanonicalArtifact.LABEL_SIGNATURES,
        _CanonicalArtifact.MANIFEST,
        _CanonicalArtifact.BUILD_CONFIG,
        _CanonicalArtifact.SEMANTIC_MARKER,
    }
)
_PROMOTION_ORDER = (
    _CanonicalArtifact.CONTRIBUTIONS,
    _CanonicalArtifact.GRAPH,
    _CanonicalArtifact.REPORT,
    _CanonicalArtifact.ANALYSIS,
    _CanonicalArtifact.LABELS,
    _CanonicalArtifact.LABEL_SIGNATURES,
    _CanonicalArtifact.MANIFEST,
    _CanonicalArtifact.ROOT,
    _CanonicalArtifact.BUILD_CONFIG,
    _CanonicalArtifact.SEMANTIC_MARKER,
    _CanonicalArtifact.NEEDS_UPDATE,
)


class _PublicationTransaction:
    """Publish one complete candidate through a recoverable filesystem journal."""

    def __init__(
        self,
        corpus: Corpus,
        layout: _PublicationLayout,
        *,
        operation: str,
        protect_previous: bool,
    ) -> None:
        """Bind one new transaction to its Corpus and canonical placements."""
        self._corpus = corpus
        self._requested_layout = layout
        self._operation = operation
        self._protect_previous = protect_previous
        self._replaces_graph = False
        self._transaction_id = uuid.uuid4().hex
        self._output = corpus.output
        self._journal_path = self._output / _JOURNAL_NAME
        self._workspace = self._output / _WORKSPACE_NAME
        self._candidate = self._workspace / "candidate"
        self._prior = self._workspace / "prior"
        self._placements = self._resolve_placements()
        self._layout = self._layout_for(
            corpus.root,
            corpus.output,
            self._placements,
        )

    @property
    def layout(self) -> _PublicationLayout:
        """Return stable placements resolved from active and requested metadata."""
        return self._layout

    @classmethod
    def active_layout(cls, corpus: Corpus) -> _PublicationLayout:
        """Return where the active generation's artifacts actually live.

        Readers that inspect active state before staging anything — reconciling
        an operation, for instance — must resolve the same placements the next
        transaction will promote to. Reading ``output/<name>`` directly would
        silently inspect the wrong file for any artifact a legacy runbook placed
        beside ``graphify-out``.
        """
        return cls._layout_for(
            corpus.root,
            corpus.output,
            cls._active_marker_placements(corpus.output),
        )

    @classmethod
    def active_artifact_path(
        cls,
        output: Path,
        artifact: _CanonicalArtifact,
    ) -> Path:
        """Resolve one active artifact's location from the output alone.

        A read-only disclosure reader knows where a Corpus publishes but has no
        business inventing its source root, and placement resolution never needs
        one. Naming the output as the layout root keeps that honest: the value is
        unused for path resolution and no caller can mistake it for the Corpus.
        """
        return cls._layout_for(
            output,
            output,
            cls._active_marker_placements(output),
        ).path_for(artifact)

    @staticmethod
    def _layout_for(
        root: Path,
        output: Path,
        placements: Mapping[str, str],
    ) -> _PublicationLayout:
        """Turn closed placement metadata into a resolvable artifact layout."""
        return _PublicationLayout(
            root=root,
            output=output,
            overrides={
                artifact: output.parent / artifact.value
                for artifact in _PROMOTION_ORDER
                if placements.get(artifact.value) == "compatibility-root"
            },
        )

    def replacing_graph(self) -> None:
        """Authorize legacy-prior reconciliation before a graph replacement."""
        self._replaces_graph = True

    def active_generation_is_valid(self) -> bool:
        """Validate the stable generation against its completion marker."""
        records: dict[str, dict[str, Any]] = {}
        try:
            for artifact in _PROMOTION_ORDER:
                location = self._placements[artifact.value]
                path = self._target_path(
                    self._output,
                    artifact.value,
                    location,
                )
                records[artifact.value] = (
                    self._record(path, location=location)
                    if path.exists()
                    else {"location": location, "present": False}
                )
            marker = self._output / _CanonicalArtifact.COMPLETION.value
            records[_CanonicalArtifact.COMPLETION.value] = (
                self._record(marker, location="output")
                if marker.exists()
                else {"location": "output", "present": False}
            )
        except OSError:
            return False
        return self._records_are_valid(
            records,
            self._output,
            output=self._output,
            transaction_id=None,
            require_completion=True,
        )

    def active_graph_matches_authoritative_contributions(self) -> bool:
        """Return whether stable Graph topology still matches its Source ledger."""
        graph = _CanonicalArtifact.GRAPH
        contributions = _CanonicalArtifact.CONTRIBUTIONS
        try:
            records = {
                artifact.value: self._record(
                    self._target_path(
                        self._output,
                        artifact.value,
                        self._placements[artifact.value],
                    ),
                    location=self._placements[artifact.value],
                )
                for artifact in (graph, contributions)
            }
            return self._topology_is_equivalent(
                records,
                self._output,
                output=self._output,
                flat=False,
                require_pair=True,
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return False

    @classmethod
    def recover(cls, corpus: Corpus) -> PublicationRefused | None:
        """Finish or roll back the one durable transaction for ``corpus``."""
        output = corpus.output
        journal_path = output / _JOURNAL_NAME
        workspace = output / _WORKSPACE_NAME
        if not journal_path.exists():
            return None
        try:
            journal = cls._read_journal(journal_path)
            transaction_id = journal["transaction_id"]
            phase = journal["phase"]
            candidate_records = journal.get("candidate", {})
            prior_records = journal.get("prior", {})
            candidate = workspace / "candidate"
            prior = workspace / "prior"

            if phase == "preparing":
                # Stable paths are untouched until "promoting" is durable, so
                # an unauthorized/incomplete candidate must simply be discarded.
                cls._cleanup_paths(workspace, journal_path)
                return None

            if phase == "committed" and cls._records_are_valid(
                candidate_records,
                output,
                output=output,
                transaction_id=transaction_id,
                require_completion=True,
            ):
                cls._cleanup_temporary_targets(
                    output,
                    candidate_records,
                    transaction_id,
                )
                cls._cleanup_paths(workspace, journal_path)
                return None

            if cls._records_are_valid(
                candidate_records,
                candidate,
                output=output,
                transaction_id=transaction_id,
                require_completion=True,
                flat=True,
            ):
                cls._promote_records(
                    candidate_records,
                    candidate,
                    output,
                    transaction_id=transaction_id,
                )
                cls._cleanup_temporary_targets(
                    output,
                    candidate_records,
                    transaction_id,
                )
                cls._cleanup_paths(workspace, journal_path)
                return None

            if cls._records_are_valid(
                prior_records,
                prior,
                output=output,
                transaction_id=None,
                require_completion=False,
                flat=True,
            ):
                cls._promote_records(
                    prior_records,
                    prior,
                    output,
                    transaction_id=None,
                )
                cls._cleanup_temporary_targets(
                    output,
                    candidate_records,
                    transaction_id,
                )
                cls._cleanup_paths(workspace, journal_path)
                return None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return PublicationRefused(f"cannot recover interrupted publication: {exc}")
        return PublicationRefused(
            "cannot recover interrupted publication: candidate and prior generation "
            "are both invalid"
        )

    def begin(self) -> _PublicationLayout:
        """Journal intent and copy the active generation into candidate staging."""
        self._output.mkdir(parents=True, exist_ok=True)
        if self._journal_path.exists():
            raise RuntimeError("publication recovery did not clear the active journal")
        self._write_journal("preparing")
        self._candidate.mkdir(parents=True)
        self._prior.mkdir(parents=True)
        for artifact in _PROMOTION_ORDER:
            source = self._target_path(
                self._output,
                artifact.value,
                self._placements[artifact.value],
            )
            if source.exists():
                shutil.copy2(source, self._candidate / artifact.value)
        return _PublicationLayout(
            root=self._corpus.root,
            output=self._candidate,
            overrides={},
        )

    def prepare(self) -> PublicationRefused | None:
        """Validate candidate/prior bytes and durably authorize their promotion."""
        try:
            candidate_records = self._record_staged_generation(
                self._candidate,
                include_completion=True,
            )
            prior_records = self._snapshot_prior()
            if self._protect_previous:
                from graphify.export import backup_if_protected

                backup_if_protected(self._output, strict=True)
            self._write_journal(
                "prepared",
                candidate=candidate_records,
                prior=prior_records,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.abort()
            return PublicationRefused(f"cannot prepare protected publication: {exc}")
        return None

    def promote(self) -> PublicationRefused | None:
        """Promote every candidate artifact and its completion marker last."""
        try:
            journal = self._read_journal(self._journal_path)
            candidate_records = journal["candidate"]
            prior_records = journal["prior"]
            self._write_journal(
                "promoting",
                candidate=candidate_records,
                prior=prior_records,
            )
            self._promote_records(
                candidate_records,
                self._candidate,
                self._output,
                transaction_id=self._transaction_id,
            )
            self._write_journal(
                "committed",
                candidate=candidate_records,
                prior=prior_records,
            )
            self._cleanup_paths(self._workspace, self._journal_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            candidate_was_valid = self._staged_candidate_is_valid()
            recovered = self.recover(self._corpus)
            if recovered is not None:
                return recovered
            if not candidate_was_valid:
                return PublicationRefused(f"publication failed before a safe commit: {exc}")
        return None

    def abort(self) -> None:
        """Discard this transaction only when its journal proves ownership."""
        try:
            if not self._journal_path.exists():
                return
            journal = self._read_journal(self._journal_path)
            if journal["transaction_id"] != self._transaction_id:
                return
            self._cleanup_paths(self._workspace, self._journal_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # An unreadable journal cannot prove ownership; recovery must make
            # the fail-closed cleanup decision on a later operation.
            pass

    def _placement_for(self, artifact: _CanonicalArtifact) -> str:
        """Encode the closed canonical placement for one artifact."""
        target = self._requested_layout.path_for(artifact).absolute()
        if target == (self._output / artifact.value).absolute():
            return "output"
        if target == (self._output.parent / artifact.value).absolute():
            return "compatibility-root"
        raise ValueError(f"unsupported canonical placement for {artifact.value}: {target}")

    def _resolve_placements(self) -> dict[str, str]:
        """Preserve active placements unless this publication overrides them."""
        placements = {
            artifact.value: self._placement_for(artifact)
            for artifact in _PROMOTION_ORDER
        }
        active = self._active_marker_placements(self._output)
        for artifact in _PROMOTION_ORDER:
            if artifact not in self._requested_layout.overrides:
                placements[artifact.value] = active.get(
                    artifact.value,
                    placements[artifact.value],
                )
        return placements

    @staticmethod
    def _active_marker_placements(output: Path) -> dict[str, str]:
        """Read only closed placement metadata from the active completion marker."""
        marker_path = output / _CanonicalArtifact.COMPLETION.value
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(marker, dict)
            or marker.get("schema") != _COMPLETION_SCHEMA
            or marker.get("version") != _SCHEMA_VERSION
            or not isinstance(marker.get("artifacts"), dict)
        ):
            return {}
        placements: dict[str, str] = {}
        canonical_names = {artifact.value for artifact in _PROMOTION_ORDER}
        for name, metadata in marker["artifacts"].items():
            if (
                name in canonical_names
                and isinstance(metadata, dict)
                and metadata.get("location") in {"output", "compatibility-root"}
            ):
                placements[name] = metadata["location"]
        return placements

    def _snapshot_prior(self) -> dict[str, dict[str, Any]]:
        """Copy and validate the byte-exact rollback generation."""
        records: dict[str, dict[str, Any]] = {}
        for artifact in _PROMOTION_ORDER:
            location = self._placements[artifact.value]
            source = self._target_path(self._output, artifact.value, location)
            destination = self._prior / artifact.value
            if source.exists():
                shutil.copy2(source, destination)
                records[artifact.value] = self._record(
                    destination,
                    location=location,
                )
            else:
                records[artifact.value] = {
                    "location": location,
                    "present": False,
                }
        completion = _CanonicalArtifact.COMPLETION
        stable_marker = self._output / completion.value
        prior_marker = self._prior / completion.value
        if stable_marker.exists():
            shutil.copy2(stable_marker, prior_marker)
            records[completion.value] = self._record(
                prior_marker,
                location="output",
            )
        else:
            records[completion.value] = {
                "location": "output",
                "present": False,
            }
        valid = self._records_are_valid(
            records,
            self._prior,
            output=self._output,
            transaction_id=None,
            require_completion=False,
            flat=True,
        )
        if (
            not valid
            and records[completion.value]["present"]
            and self._marker_diff_is_admissible_label_curation(
                records,
                prior_marker,
            )
        ):
            # Existing adapters still admit valid external label edits and
            # markerless legacy state. A stale marker is not rollback authority:
            # retain the readable artifact bytes but restore them as legacy so
            # recovery never blesses stale digest metadata.
            prior_marker.unlink(missing_ok=True)
            records[completion.value] = {
                "location": "output",
                "present": False,
            }
            valid = self._records_are_valid(
                records,
                self._prior,
                output=self._output,
                transaction_id=None,
                require_completion=False,
                flat=True,
            )
        if (
            not valid
            and records[completion.value]["present"]
            and self._replaces_graph
        ):
            # A compatibility adapter replacing graph.json may encounter prior
            # bytes written outside CorpusGraph. Re-adopt that Graph as
            # provisional Source custody for rollback; sidecar-only operations
            # never receive this exception and therefore cannot bless tampering.
            prior_marker.unlink(missing_ok=True)
            records[completion.value] = {
                "location": "output",
                "present": False,
            }
            valid = self._adopt_markerless_legacy_prior(records)
        if not valid and not records[completion.value]["present"]:
            valid = self._adopt_markerless_legacy_prior(records)
        if not valid:
            raise ValueError("the protected prior generation did not validate")
        return records

    def _adopt_markerless_legacy_prior(
        self,
        records: dict[str, dict[str, Any]],
    ) -> bool:
        """Reconcile an explicitly markerless legacy graph into prior custody."""
        graph_record = records[_CanonicalArtifact.GRAPH.value]
        if not graph_record["present"]:
            return False
        graph_path = self._prior / _CanonicalArtifact.GRAPH.value
        ledger_path = self._prior / _CanonicalArtifact.CONTRIBUTIONS.value
        try:
            contributions = _adopt_legacy_graph(graph_path, self._corpus.root)
            _write_contribution_ledger(ledger_path, contributions)
            records[_CanonicalArtifact.CONTRIBUTIONS.value] = self._record(
                ledger_path,
                location=self._placements[_CanonicalArtifact.CONTRIBUTIONS.value],
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return self._records_are_valid(
            records,
            self._prior,
            output=self._output,
            transaction_id=None,
            require_completion=False,
            flat=True,
        )

    @staticmethod
    def _marker_diff_is_admissible_label_curation(
        records: Mapping[str, Mapping[str, Any]],
        marker_path: Path,
    ) -> bool:
        """Allow only a labels-file edit to invalidate an otherwise exact marker."""
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            not isinstance(marker, dict)
            or marker.get("schema") != _COMPLETION_SCHEMA
            or marker.get("version") != _SCHEMA_VERSION
            or not isinstance(marker.get("transaction_id"), str)
            or not marker["transaction_id"]
            or not isinstance(marker.get("artifacts"), dict)
        ):
            return False
        current = {
            name: {
                "location": record["location"],
                "sha256": record["sha256"],
                "size": record["size"],
            }
            for name, record in records.items()
            if name != _CanonicalArtifact.COMPLETION.value
            and record["present"]
        }
        marked = marker["artifacts"]
        differing = {
            name
            for name in set(current) | set(marked)
            if current.get(name) != marked.get(name)
        }
        return bool(differing) and differing <= {_CanonicalArtifact.LABELS.value}

    def _record_staged_generation(
        self,
        directory: Path,
        *,
        include_completion: bool,
    ) -> dict[str, dict[str, Any]]:
        """Read back staged artifacts and build their digest manifest."""
        records: dict[str, dict[str, Any]] = {}
        for artifact in _PROMOTION_ORDER:
            path = directory / artifact.value
            location = self._placements[artifact.value]
            if path.exists():
                self._validate_artifact(artifact, path)
                records[artifact.value] = self._record(path, location=location)
            else:
                records[artifact.value] = {
                    "location": location,
                    "present": False,
                }
        if include_completion:
            marker_path = directory / _CanonicalArtifact.COMPLETION.value
            marker = self._completion_payload(records)
            from graphify.paths import write_json_atomic

            # The candidate marker is serialized only after every artifact was
            # read back and validated; stable promotion repeats this ordering.
            write_json_atomic(marker_path, marker, indent=2, ensure_ascii=False)
            records[_CanonicalArtifact.COMPLETION.value] = self._record(
                marker_path,
                location="output",
            )
            if not self._records_are_valid(
                records,
                directory,
                output=self._output,
                transaction_id=self._transaction_id,
                require_completion=True,
                flat=True,
            ):
                raise ValueError("the complete candidate did not validate")
        return records

    def _completion_payload(
        self,
        records: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return the stable completion marker for the staged candidate."""
        return {
            "schema": _COMPLETION_SCHEMA,
            "version": _SCHEMA_VERSION,
            "transaction_id": self._transaction_id,
            "artifacts": {
                name: {
                    "location": record["location"],
                    "sha256": record["sha256"],
                    "size": record["size"],
                }
                for name, record in records.items()
                if record["present"]
            },
        }

    def _write_journal(
        self,
        phase: str,
        *,
        candidate: Mapping[str, Mapping[str, Any]] | None = None,
        prior: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Atomically persist one closed publication phase."""
        if phase not in _PHASES:
            raise ValueError(f"invalid publication phase: {phase}")
        from graphify.paths import write_json_atomic

        write_json_atomic(
            self._journal_path,
            {
                "schema": _JOURNAL_SCHEMA,
                "version": _SCHEMA_VERSION,
                "transaction_id": self._transaction_id,
                "operation": self._operation,
                "phase": phase,
                "candidate": dict(candidate or {}),
                "prior": dict(prior or {}),
            },
            indent=2,
            ensure_ascii=False,
        )

    def _staged_candidate_is_valid(self) -> bool:
        """Return whether recovery can still roll this staged candidate forward."""
        try:
            journal = self._read_journal(self._journal_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return self._records_are_valid(
            journal.get("candidate", {}),
            self._candidate,
            output=self._output,
            transaction_id=self._transaction_id,
            require_completion=True,
            flat=True,
        )

    @classmethod
    def _promote_records(
        cls,
        records: Mapping[str, Mapping[str, Any]],
        source_directory: Path,
        output: Path,
        *,
        transaction_id: str | None,
    ) -> None:
        """Promote or restore one validated record set with its marker last."""
        marker_target = output / _CanonicalArtifact.COMPLETION.value
        marker_target.unlink(missing_ok=True)
        for artifact in _PROMOTION_ORDER:
            record = records[artifact.value]
            target = cls._target_path(output, artifact.value, record["location"])
            if record["present"]:
                cls._copy_atomic(
                    source_directory / artifact.value,
                    target,
                    transaction_id=transaction_id or "restore",
                )
            else:
                target.unlink(missing_ok=True)

        marker_record = records[_CanonicalArtifact.COMPLETION.value]
        if marker_record["present"]:
            cls._copy_atomic(
                source_directory / _CanonicalArtifact.COMPLETION.value,
                marker_target,
                transaction_id=transaction_id or "restore",
            )
        else:
            marker_target.unlink(missing_ok=True)
        if not cls._records_are_valid(
            records,
            output,
            output=output,
            transaction_id=transaction_id,
            require_completion=marker_record["present"],
        ):
            raise ValueError("promoted Graph generation did not validate")

    @staticmethod
    def _copy_atomic(source: Path, target: Path, *, transaction_id: str) -> None:
        """Copy one staged byte stream into a canonical path atomically."""
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{transaction_id}.tmp")
        try:
            shutil.copy2(source, temporary)
            try:
                os.replace(temporary, target)
            except PermissionError:
                # Windows may briefly deny replacement of an open destination;
                # copy-over retains the established atomic-writer fallback.
                shutil.copy2(temporary, target)
                temporary.unlink()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def _records_are_valid(
        cls,
        records: Mapping[str, Mapping[str, Any]],
        base: Path,
        *,
        output: Path,
        transaction_id: str | None,
        require_completion: bool,
        flat: bool = False,
    ) -> bool:
        """Validate record schema, digests, artifact formats, and completion."""
        expected = {artifact.value for artifact in (*_PROMOTION_ORDER, _CanonicalArtifact.COMPLETION)}
        if set(records) != expected:
            return False
        try:
            for name, record in records.items():
                artifact = _CanonicalArtifact(name)
                cls._validate_record(record)
                path = (
                    base / name
                    if flat
                    else cls._target_path(output, name, record["location"])
                )
                if not record["present"]:
                    if path.exists():
                        return False
                    continue
                if not path.is_file():
                    return False
                if cls._record(path, location=record["location"]) != dict(record):
                    return False
                cls._validate_artifact(artifact, path)

            marker_record = records[_CanonicalArtifact.COMPLETION.value]
            if require_completion and not marker_record["present"]:
                return False
            if not cls._topology_is_equivalent(
                records,
                base,
                output=output,
                flat=flat,
                require_pair=require_completion or marker_record["present"],
            ):
                return False
            if marker_record["present"]:
                marker_path = (
                    base / _CanonicalArtifact.COMPLETION.value
                    if flat
                    else output / _CanonicalArtifact.COMPLETION.value
                )
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                expected_artifacts = {
                    name: {
                        "location": record["location"],
                        "sha256": record["sha256"],
                        "size": record["size"],
                    }
                    for name, record in records.items()
                    if name != _CanonicalArtifact.COMPLETION.value
                    and record["present"]
                }
                if (
                    marker.get("schema") != _COMPLETION_SCHEMA
                    or marker.get("version") != _SCHEMA_VERSION
                    or not isinstance(marker.get("transaction_id"), str)
                    or not marker["transaction_id"]
                    or (
                        transaction_id is not None
                        and marker["transaction_id"] != transaction_id
                    )
                    or marker.get("artifacts") != expected_artifacts
                ):
                    return False
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return False
        return True

    @classmethod
    def _topology_is_equivalent(
        cls,
        records: Mapping[str, Mapping[str, Any]],
        base: Path,
        *,
        output: Path,
        flat: bool,
        require_pair: bool,
    ) -> bool:
        """Compare the graph view with its authoritative contribution topology."""
        graph_record = records[_CanonicalArtifact.GRAPH.value]
        ledger_record = records[_CanonicalArtifact.CONTRIBUTIONS.value]
        graph_present = graph_record["present"]
        ledger_present = ledger_record["present"]
        if graph_present != ledger_present:
            return not require_pair
        if not graph_present:
            return True
        graph_path = cls._path_for_record(
            base,
            output,
            _CanonicalArtifact.GRAPH.value,
            graph_record,
            flat=flat,
        )
        ledger_path = cls._path_for_record(
            base,
            output,
            _CanonicalArtifact.CONTRIBUTIONS.value,
            ledger_record,
            flat=flat,
        )
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        contributions = tuple(_iter_contribution_ledger(ledger_path))
        materialized = _materialize_graph_data(
            contributions,
            multigraph=bool(graph.get("multigraph", False)),
        )
        graph_edges = graph.get("links", graph.get("edges", []))
        if not isinstance(graph_edges, list):
            return False
        return (
            cls._node_id_counts(graph["nodes"])
            == cls._node_id_counts(materialized["nodes"])
            and cls._edge_topology_counts(graph_edges)
            == cls._edge_topology_counts(materialized["links"])
            and cls._canonical_items(graph.get("hyperedges", []))
            == cls._canonical_items(materialized["hyperedges"])
        )

    @staticmethod
    def _path_for_record(
        base: Path,
        output: Path,
        name: str,
        record: Mapping[str, Any],
        *,
        flat: bool,
    ) -> Path:
        """Resolve a validated record to its staged or stable filesystem path."""
        if flat:
            return base / name
        return _PublicationTransaction._target_path(
            output,
            name,
            record["location"],
        )

    @staticmethod
    def _node_id_counts(nodes: list[Any]) -> Counter[str]:
        """Return order-independent multiplicities for materialized node identities."""
        if not all(isinstance(node, dict) for node in nodes):
            raise ValueError("graph nodes must be JSON objects")
        return Counter(
            json.dumps(
                node.get("id"),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for node in nodes
        )

    @staticmethod
    def _edge_topology_counts(edges: list[Any]) -> Counter[str]:
        """Return order-independent directed connectivity and relation identities."""
        if not all(isinstance(edge, dict) for edge in edges):
            raise ValueError("graph edges must be JSON objects")
        # The clustered writer restores these stashed endpoints after
        # NetworkX canonicalizes undirected storage, so validation must compare
        # the same directed evidence that reaches graph.json.
        return Counter(
            json.dumps(
                {
                    "source": edge.get("_src", edge.get("source")),
                    "target": edge.get("_tgt", edge.get("target")),
                    "relation": edge.get("relation"),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for edge in edges
        )

    @staticmethod
    def _canonical_items(items: Any) -> list[str]:
        """Return a stable comparison form for hyperedge topology records."""
        if not isinstance(items, list) or not all(
            isinstance(item, dict) for item in items
        ):
            raise ValueError("graph hyperedges must be JSON objects")
        return sorted(
            json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for item in items
        )

    @staticmethod
    def _validate_record(record: Mapping[str, Any]) -> None:
        """Reject malformed or escaping placement metadata."""
        if set(record) not in (
            {"location", "present"},
            {"location", "present", "sha256", "size"},
        ):
            raise ValueError("invalid publication artifact record")
        if record["location"] not in {"output", "compatibility-root"}:
            raise ValueError("invalid publication artifact placement")
        if not isinstance(record["present"], bool):
            raise ValueError("invalid publication artifact presence")
        if record["present"]:
            if (
                not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or not isinstance(record.get("size"), int)
                or record["size"] < 0
            ):
                raise ValueError("invalid publication artifact digest")

    @staticmethod
    def _record(path: Path, *, location: str) -> dict[str, Any]:
        """Return the digest record for one staged artifact."""
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return {
            "location": location,
            "present": True,
            "sha256": digest.hexdigest(),
            "size": size,
        }

    @staticmethod
    def _validate_artifact(artifact: _CanonicalArtifact, path: Path) -> None:
        """Read back one canonical artifact through its production format."""
        if artifact is _CanonicalArtifact.CONTRIBUTIONS:
            tuple(_iter_contribution_ledger(path))
            return
        if artifact in _JSON_ARTIFACTS or artifact is _CanonicalArtifact.COMPLETION:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{artifact.value} must contain a JSON object")
            if artifact is _CanonicalArtifact.GRAPH:
                nodes = payload.get("nodes")
                edges = payload.get("links", payload.get("edges"))
                if not isinstance(nodes, list) or not isinstance(edges, list):
                    raise ValueError("graph.json must contain node and edge lists")
            return
        content = path.read_text(encoding="utf-8")
        if artifact is _CanonicalArtifact.NEEDS_UPDATE and content != "1":
            raise ValueError("needs_update must contain the compatibility value '1'")

    @staticmethod
    def _target_path(output: Path, name: str, location: str) -> Path:
        """Resolve one closed placement without trusting a journal path."""
        if Path(name).name != name or name not in {artifact.value for artifact in _CanonicalArtifact}:
            raise ValueError(f"invalid canonical artifact name: {name}")
        if location == "output":
            return output / name
        if location == "compatibility-root":
            return output.parent / name
        raise ValueError(f"invalid canonical artifact placement: {location}")

    @staticmethod
    def _read_journal(path: Path) -> dict[str, Any]:
        """Parse and validate the durable publication journal header."""
        journal = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(journal, dict)
            or journal.get("schema") != _JOURNAL_SCHEMA
            or journal.get("version") != _SCHEMA_VERSION
            or journal.get("phase") not in _PHASES
            or not isinstance(journal.get("transaction_id"), str)
            or not journal["transaction_id"]
            or not isinstance(journal.get("operation"), str)
            or not isinstance(journal.get("candidate"), dict)
            or not isinstance(journal.get("prior"), dict)
        ):
            raise ValueError(f"invalid publication journal: {path}")
        return journal

    @staticmethod
    def _cleanup_paths(workspace: Path, journal_path: Path) -> None:
        """Remove only the fixed transaction workspace, then its journal."""
        if workspace.exists():
            shutil.rmtree(workspace)
        journal_path.unlink(missing_ok=True)

    @classmethod
    def _cleanup_temporary_targets(
        cls,
        output: Path,
        records: Mapping[str, Mapping[str, Any]],
        transaction_id: str,
    ) -> None:
        """Remove exact transaction-named copy files left by process termination."""
        for name, record in records.items():
            if name == _CanonicalArtifact.COMPLETION.value:
                target = output / name
            else:
                try:
                    target = cls._target_path(output, name, record["location"])
                except (KeyError, TypeError, ValueError):
                    continue
            temporary = target.with_name(f".{target.name}.{transaction_id}.tmp")
            temporary.unlink(missing_ok=True)
