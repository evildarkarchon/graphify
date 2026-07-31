"""Durable cross-process request acceptance and executor leasing for one Corpus.

Two pieces of coordination state live here, both owned by ``CorpusGraph``:

* a **durable request queue** — one fsynced record per accepted request, so a
  background caller can be told ``Queued`` and then exit without losing work; and
* one **executor lease** — the cross-platform claim that makes exactly one
  process responsible for recovery and publication at a time.

Both replace the previous POSIX-only ``flock`` and the path-only
``.pending_changes`` file. The queue records typed requests rather than bare
paths, so an accepted request survives the death of the process that submitted
it *and* the death of the process that was executing it; the lease is recovered
without manual cleanup when its holder can no longer be proven alive.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from graphify.generation._types import Corpus

_REQUEST_SCHEMA = "graphify-generation-request"
_LEASE_SCHEMA = "graphify-executor-lease"
_SCHEMA_VERSION = 1
_REQUEST_DIRNAME = ".graphify_requests"
_LEASE_FILENAME = ".graphify_executor.json"
_CLAIM_PREFIX = ".graphify_executor.claim-"

_OPERATIONS = frozenset({"full-extraction", "code-update", "reclustering"})

# What queued work a completed operation has already done. Code update and
# Reclustering cover only their own kind; a Full extraction reconciles every
# Corpus source and republishes community identity, so it subsumes both. Only an
# operation this package owns end to end may consult this: coverage is a claim
# about which moment of the Corpus the finished work describes, which a caller
# that merely handed over a prepared candidate cannot make. Code update is the
# only such operation today, so it is the only entry read until Full extraction
# and Reclustering stop being prepared by compatibility adapters.
_SUBSUMES: Mapping[str, frozenset[str]] = {
    "full-extraction": frozenset({"full-extraction", "code-update", "reclustering"}),
    "code-update": frozenset({"code-update"}),
    "reclustering": frozenset({"reclustering"}),
}

# A lease whose heartbeat is older than this can no longer prove it is owned.
# The holder renews well inside the window, so only a stopped executor expires.
_DEFAULT_LEASE_TTL = 30.0
# How long an executor waits for the lease before reporting that another process
# owns the Corpus. Generous, because the holder may be running a real extraction.
_DEFAULT_LEASE_TIMEOUT = 900.0
_LEASE_POLL_INTERVAL = 0.1

# How many extra passes an executor makes for requests accepted while it was
# already running. Bounded so a storm of commits quiesces instead of keeping one
# process working forever; anything still queued stays durably accepted for the
# next executor. Lives here so the operations and the adapters that still run
# their own rebuild loop share one budget.
_LATE_ARRIVAL_PASSES = 20

_HOST = platform.node() or "unknown-host"


class _LeaseOutcome(Enum):
    """Why an executor did or did not take ownership of a Corpus."""

    HELD = "held"
    COVERED = "covered"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class _AcceptedRequest:
    """One durably accepted request for a Corpus."""

    request_id: str
    sequence: int
    operation: str
    changed_paths: tuple[str, ...]
    force: bool
    pid: int
    host: str
    path: Path


def _lease_ttl() -> float:
    """Return the configured lease heartbeat window, in seconds."""
    return _positive_float("GRAPHIFY_EXECUTOR_LEASE_TTL", _DEFAULT_LEASE_TTL)


def _lease_timeout() -> float:
    """Return how long an executor waits for the lease, in seconds."""
    return _positive_float("GRAPHIFY_EXECUTOR_LEASE_TIMEOUT", _DEFAULT_LEASE_TIMEOUT)


def _positive_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to ``default``.

    Read per call rather than at import so a value set after import — by a test,
    a hook wrapper, or an operator debugging a wedged Corpus — is honored.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _process_is_alive(pid: int, host: str) -> bool:
    """Return whether ``pid`` on ``host`` can still be proven to be running.

    Liveness is only decidable on this host: a pid recorded by another machine is
    reported alive, so only the lease heartbeat can retire a foreign holder.

    Windows has no signal-0 probe — ``os.kill`` there calls ``TerminateProcess``
    and would kill the very process being asked about — so the process handle is
    opened and its exit state queried instead.
    """
    if host != _HOST:
        return True
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists; this user simply may not signal it.
        return True
    except OSError:
        return True
    return True


def _windows_process_is_alive(pid: int) -> bool:
    """Return whether a Windows process handle for ``pid`` is still unsignaled."""
    import ctypes

    process_query_limited_information = 0x1000
    synchronize = 0x00100000
    wait_object_0 = 0x0
    error_access_denied = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(
        process_query_limited_information | synchronize,
        0,
        pid,
    )
    if not handle:
        # Access denied proves the process exists; every other failure — most
        # commonly ERROR_INVALID_PARAMETER — means there is no such process.
        return ctypes.get_last_error() == error_access_denied
    try:
        # A terminated process object is signaled, so a zero-timeout wait that
        # returns WAIT_OBJECT_0 means the process has already exited.
        return kernel32.WaitForSingleObject(handle, 0) != wait_object_0
    finally:
        kernel32.CloseHandle(handle)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry where the platform exposes one.

    Windows has no directory descriptor to sync, so the rename itself is the
    durability boundary there; the failure is expected and deliberately ignored.
    """
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is unsupported on some filesystems; the rename still
        # made the record visible, which is what a later reader needs.
        pass
    finally:
        os.close(descriptor)


def _write_json_durably(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace ``path`` with ``payload`` so no reader can observe a partial record.

    The bytes are flushed and fsynced before the rename, and the containing
    directory is fsynced after it, so a record this function returns from is on
    disk rather than in a write-back cache. That ordering is what lets a caller
    be told its request was accepted and then exit immediately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    _fsync_directory(path.parent)


# In-process serialization for one Corpus output. The file lease alone cannot
# express reentrancy, and an owning operation legitimately re-enters it when its
# publisher takes custody of the same Corpus.
_REGISTRY_LOCK = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_LEASE_DEPTH: dict[str, int] = {}


def _process_lock(key: str) -> threading.RLock:
    """Return the process-wide reentrant lock guarding one Corpus output."""
    with _REGISTRY_LOCK:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


class _LeaseHolder:
    """Own one acquired executor lease and keep its heartbeat current."""

    def __init__(self, path: Path, lease_id: str) -> None:
        """Start renewing ``path`` until the lease is released or taken over."""
        self._path = path
        self._lease_id = lease_id
        self._stop = threading.Event()
        # Serializes a renewal against the release that clears the lease file.
        # Without it a renewal already in flight when release() unlinks would
        # recreate the file afterwards, leaving an orphan lease that blocks every
        # other executor until its heartbeat ages past the TTL.
        self._writing = threading.Lock()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="graphify-executor-lease",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat(self) -> None:
        """Renew the lease on a fixed interval until told to stop."""
        while not self._stop.wait(max(1.0, _lease_ttl() / 3)):
            with self._writing:
                if self._stop.is_set():
                    # Released while this renewal waited for the write lock.
                    return
                if not self._renew():
                    return

    def _renew(self) -> bool:
        """Stamp a fresh heartbeat, or report that this lease is no longer ours."""
        record = _read_lease_record(self._path)
        if record is None or record.get("lease_id") != self._lease_id:
            # Another executor already took this lease over. Renewing now would
            # forge a heartbeat for the new owner's record, so stop instead.
            return False
        try:
            _write_json_durably(self._path, {**record, "renewed_at": time.time()})
        except OSError:
            return False
        return True

    def release(self) -> None:
        """Stop the heartbeat and clear the lease if this holder still owns it.

        The write lock is taken before the unlink so no renewal can be in flight
        across it; a renewal blocked on the lock sees the stop flag and returns
        without writing. The thread is joined afterwards purely to reap it, so a
        heartbeat wedged on a slow filesystem cannot hold up the caller.
        """
        self._stop.set()
        with self._writing:
            record = _read_lease_record(self._path)
            if record is not None and record.get("lease_id") == self._lease_id:
                with contextlib.suppress(OSError):
                    self._path.unlink()
        self._thread.join(timeout=5)


def _read_lease_record(path: Path) -> dict[str, Any] | None:
    """Return a well-formed lease record, or ``None`` when there is not one."""
    record, _ = _read_lease(path)
    return record


def _read_lease(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Return the lease record and a fingerprint of the exact bytes observed.

    The fingerprint identifies one observed lease state even when the record
    itself is unreadable, which is what makes a takeover claim specific to the
    state that was judged stale rather than to the file name.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, ""
    except OSError:
        return None, "unreadable"
    fingerprint = hashlib.sha256(raw).hexdigest()[:32]
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, fingerprint
    if (
        not isinstance(record, dict)
        or record.get("schema") != _LEASE_SCHEMA
        or record.get("version") != _SCHEMA_VERSION
    ):
        return None, fingerprint
    return record, fingerprint


def _lease_is_stale(record: Mapping[str, Any] | None) -> bool:
    """Return whether a lease record can no longer prove it is owned."""
    if record is None:
        # An absent, truncated, or foreign-schema record proves nothing. Treating
        # it as owned would wedge the Corpus until someone deleted the file.
        return True
    host = record.get("host")
    pid = record.get("pid")
    renewed_at = record.get("renewed_at")
    if (
        not isinstance(host, str)
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(renewed_at, (int, float))
    ):
        return True
    if time.time() - float(renewed_at) > _lease_ttl():
        return True
    return not _process_is_alive(pid, host)


def _read_request(path: Path) -> _AcceptedRequest | None:
    """Return one accepted request, or ``None`` when the record is not usable.

    An unusable record is skipped and deliberately left on disk. Records are
    published by atomic rename, so a torn one should not exist; something that
    still cannot be read is either not ours or evidence of a problem, and neither
    is a reason to delete a file whose meaning we could not establish.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("schema") != _REQUEST_SCHEMA
        or record.get("version") != _SCHEMA_VERSION
        or record.get("operation") not in _OPERATIONS
        or not isinstance(record.get("request_id"), str)
        or not isinstance(record.get("sequence"), int)
    ):
        return None
    changed_paths = record.get("changed_paths")
    if not isinstance(changed_paths, list):
        changed_paths = []
    return _AcceptedRequest(
        request_id=record["request_id"],
        sequence=record["sequence"],
        operation=record["operation"],
        changed_paths=tuple(value for value in changed_paths if isinstance(value, str)),
        force=bool(record.get("force")),
        pid=record["pid"] if isinstance(record.get("pid"), int) else 0,
        host=record["host"] if isinstance(record.get("host"), str) else "",
        path=path,
    )


class _RequestCoordinator:
    """Durably accept requests for one Corpus and lease its single executor."""

    def __init__(self, corpus: Corpus) -> None:
        """Bind coordination state to one Corpus output directory."""
        self._corpus = corpus
        self._output = Path(corpus.output).absolute()
        self._requests = self._output / _REQUEST_DIRNAME
        self._lease_path = self._output / _LEASE_FILENAME
        # Normalized so two spellings of the same output share one process lock;
        # this is in-process bookkeeping only, never a canonical artifact path.
        self._key = os.path.normcase(str(self._output))

    def accept(
        self,
        operation: str,
        *,
        changed_paths: Iterable[str | Path] = (),
        force: bool = False,
    ) -> _AcceptedRequest:
        """Durably record one request and return it.

        The record is on disk before this returns, which is the guarantee a
        background caller needs: it may exit immediately afterwards and the work
        still belongs to the Corpus rather than to the process that asked for it.
        """
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown Graph-generation operation: {operation}")
        request_id = uuid.uuid4().hex
        # Wall-clock nanoseconds, zero padded into the file name, so a plain
        # directory listing sorts into acceptance order without a shared counter.
        sequence = time.time_ns()
        hints = tuple(os.fspath(path) for path in changed_paths)
        path = self._requests / f"{sequence:020d}-{request_id}.json"
        _write_json_durably(
            path,
            {
                "schema": _REQUEST_SCHEMA,
                "version": _SCHEMA_VERSION,
                "request_id": request_id,
                "sequence": sequence,
                "operation": operation,
                "changed_paths": list(hints),
                "force": bool(force),
                "pid": os.getpid(),
                "host": _HOST,
            },
        )
        return _AcceptedRequest(
            request_id=request_id,
            sequence=sequence,
            operation=operation,
            changed_paths=hints,
            force=bool(force),
            pid=os.getpid(),
            host=_HOST,
            path=path,
        )

    def pending(
        self,
        *,
        operations: frozenset[str] | None = None,
    ) -> tuple[_AcceptedRequest, ...]:
        """Return accepted requests still awaiting coverage, in acceptance order."""
        try:
            entries = sorted(self._requests.iterdir())
        except OSError:
            return ()
        accepted = []
        for entry in entries:
            if entry.suffix != ".json":
                continue
            request = _read_request(entry)
            if request is None:
                continue
            if operations is not None and request.operation not in operations:
                continue
            accepted.append(request)
        return tuple(accepted)

    def is_covered(self, request: _AcceptedRequest) -> bool:
        """Return whether a published generation has already covered ``request``."""
        return not request.path.exists()

    def cover(self, requests: Iterable[_AcceptedRequest]) -> None:
        """Retire requests a completed operation has already done the work for."""
        for request in requests:
            with contextlib.suppress(OSError):
                request.path.unlink()

    def subsumed_by(self, operation: str) -> frozenset[str]:
        """Return the request kinds a completed ``operation`` covers."""
        return _SUBSUMES[operation]

    @contextlib.contextmanager
    def executor_lease(
        self,
        *,
        timeout: float | None = None,
        until_covered: _AcceptedRequest | None = None,
    ) -> Iterator[_LeaseOutcome]:
        """Take sole executor ownership of this Corpus for the body of the block.

        Yields ``HELD`` when this process owns recovery and publication,
        ``COVERED`` when ``until_covered`` was satisfied by another executor
        while waiting, and ``UNAVAILABLE`` when the lease stayed owned elsewhere
        for the whole wait. Re-entering while already held is a no-op, so an
        operation and the publisher it delegates to share one ownership window.
        """
        deadline = time.monotonic() + (
            _lease_timeout() if timeout is None else float(timeout)
        )
        lock = _process_lock(self._key)
        entered = self._enter_process_lock(lock, deadline, until_covered)
        if entered is not _LeaseOutcome.HELD:
            yield entered
            return
        holder: _LeaseHolder | None = None
        try:
            if _LEASE_DEPTH.get(self._key, 0) == 0:
                outcome, holder = self._acquire_file_lease(deadline, until_covered)
                if outcome is not _LeaseOutcome.HELD:
                    yield outcome
                    return
            _LEASE_DEPTH[self._key] = _LEASE_DEPTH.get(self._key, 0) + 1
            try:
                yield _LeaseOutcome.HELD
            finally:
                _LEASE_DEPTH[self._key] -= 1
        finally:
            if holder is not None:
                holder.release()
            lock.release()

    def _enter_process_lock(
        self,
        lock: threading.RLock,
        deadline: float,
        until_covered: _AcceptedRequest | None,
    ) -> _LeaseOutcome:
        """Serialize this Corpus within the process before contending on disk."""
        while True:
            if until_covered is not None and self.is_covered(until_covered):
                return _LeaseOutcome.COVERED
            # Reentrant for the owning thread, so a nested acquire returns at once.
            if lock.acquire(timeout=_LEASE_POLL_INTERVAL):
                return _LeaseOutcome.HELD
            if time.monotonic() >= deadline:
                return _LeaseOutcome.UNAVAILABLE

    def _acquire_file_lease(
        self,
        deadline: float,
        until_covered: _AcceptedRequest | None,
    ) -> tuple[_LeaseOutcome, _LeaseHolder | None]:
        """Contend for the on-disk lease until it is won, covered, or timed out."""
        while True:
            if until_covered is not None and self.is_covered(until_covered):
                return _LeaseOutcome.COVERED, None
            holder = self._claim_lease()
            if holder is not None:
                return _LeaseOutcome.HELD, holder
            if time.monotonic() >= deadline:
                return _LeaseOutcome.UNAVAILABLE, None
            time.sleep(_LEASE_POLL_INTERVAL)

    def _claim_lease(self) -> _LeaseHolder | None:
        """Create the lease, or take over one whose holder cannot be proven alive."""
        lease_id = uuid.uuid4().hex
        payload = self._lease_payload(lease_id)
        try:
            self._output.mkdir(parents=True, exist_ok=True)
            # O_EXCL is the one primitive that is genuinely exclusive on both
            # Windows and POSIX, which is what makes this lease cross-platform.
            descriptor = os.open(
                str(self._lease_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            return self._take_over_stale_lease(lease_id, payload)
        except OSError:
            return None
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # An unwritable lease would be an unreadable one; drop the empty
            # placeholder rather than leaving a file only staleness can clear.
            with contextlib.suppress(OSError):
                self._lease_path.unlink()
            return None
        _fsync_directory(self._output)
        return _LeaseHolder(self._lease_path, lease_id)

    def _take_over_stale_lease(
        self,
        lease_id: str,
        payload: Mapping[str, Any],
    ) -> _LeaseHolder | None:
        """Replace a stale lease, letting exactly one contender win the takeover."""
        observed, fingerprint = _read_lease(self._lease_path)
        if not fingerprint:
            # The lease vanished between the failed create and this read; the
            # caller's retry will simply create it.
            return None
        if not _lease_is_stale(observed):
            return None
        claim = self._output / f"{_CLAIM_PREFIX}{fingerprint}"
        try:
            os.close(
                os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            )
        except FileExistsError:
            # Another contender is taking over this exact lease state, or one
            # died mid-takeover and left its claim behind.
            self._retire_abandoned_claim(claim)
            return None
        except OSError:
            return None
        try:
            # Re-read under the claim. Winning it is the right to replace this
            # exact observed state and nothing else, so a takeover that raced a
            # legitimate acquire is detected here instead of producing two owners.
            current, current_fingerprint = _read_lease(self._lease_path)
            if current_fingerprint != fingerprint or not _lease_is_stale(current):
                return None
            _write_json_durably(self._lease_path, payload)
        except OSError:
            return None
        finally:
            with contextlib.suppress(OSError):
                claim.unlink()
        return _LeaseHolder(self._lease_path, lease_id)

    def _retire_abandoned_claim(self, claim: Path) -> None:
        """Clear a takeover claim whose claimant died before completing it.

        Without this, a process killed between winning a claim and replacing the
        lease would block every future takeover of that lease state — exactly the
        manual cleanup a recoverable lease is supposed to remove.
        """
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            return
        if age > _lease_ttl():
            with contextlib.suppress(OSError):
                claim.unlink()

    def _lease_payload(self, lease_id: str) -> dict[str, Any]:
        """Return the record identifying this process as the Corpus executor."""
        now = time.time()
        return {
            "schema": _LEASE_SCHEMA,
            "version": _SCHEMA_VERSION,
            "lease_id": lease_id,
            "pid": os.getpid(),
            "host": _HOST,
            "acquired_at": now,
            "renewed_at": now,
        }
