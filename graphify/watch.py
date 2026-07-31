# watch a folder and submit the changes as Code updates
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

# Single source of truth in graphify.paths (#1423); re-exported as _GRAPHIFY_OUT.
from graphify.paths import GRAPHIFY_OUT as _GRAPHIFY_OUT

if TYPE_CHECKING:  # imported for annotations only; the runtime import is lazy
    from graphify.generation import Corpus, TerminalOutcome


def _watcher_honors_vcs_ignores(watch_path: Path) -> bool:
    """Return whether the watcher's own event filter should honor VCS ignores.

    Read from the Corpus build policy the active Graph generation was built
    under, so the watcher does not keep a second opinion about the shape of the
    Corpus (#1886/#1971). This only decides which filesystem events are
    forwarded as optimization hints — the operation rediscovers the Corpus
    authoritatively either way — so an unreadable policy falls back to the
    documented default rather than refusing to watch.
    """
    from graphify.generation import CorpusGraph

    try:
        return CorpusGraph(_corpus_for(watch_path)).active_build_policy().gitignore
    except (OSError, ValueError):
        return True


def _apply_resource_limits() -> None:
    """Best-effort nice + memory cap. Called from inline hook scripts.

    GRAPHIFY_REBUILD_MEMORY_LIMIT_MB caps RSS-ish memory. Uses RLIMIT_DATA on
    macOS (RLIMIT_AS is unreliable under Apple's libmalloc) and RLIMIT_AS on
    Linux. Silently skips if the platform doesn't support it.
    """
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass
    mb = os.environ.get("GRAPHIFY_REBUILD_MEMORY_LIMIT_MB", "").strip()
    if not mb:
        return
    try:
        limit = int(mb) * 1024 * 1024
    except ValueError:
        return
    try:
        import resource
        which = resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS
        soft, hard = resource.getrlimit(which)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < limit else limit
        resource.setrlimit(which, (limit, new_hard))
    except (ImportError, ValueError, OSError):
        pass


from graphify.detect import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    PAPER_EXTENSIONS,
    IMAGE_EXTENSIONS,
    _load_graphifyignore,
    _is_ignored,
)

_WATCHED_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | PAPER_EXTENSIONS | IMAGE_EXTENSIONS
_CODE_EXTENSIONS = CODE_EXTENSIONS


def _stabilize_rebuild_cwd(watch_path: Path) -> bool:
    """Ensure relative rebuild paths have a usable CWD before queue/lock setup.

    Detached git hooks can inherit a transient working directory that is deleted
    before the background rebuild starts. In that state Path.cwd(),
    Path('.').resolve(), and relative graphify-out mkdirs raise FileNotFoundError
    before the normal rebuild error handling can run. Hooks that know the repo
    root export GRAPHIFY_REPO_ROOT so the rebuild can recover by chdir'ing there.
    """
    if watch_path.is_absolute():
        return True

    repo_root = os.environ.get("GRAPHIFY_REPO_ROOT", "").strip()
    if repo_root and Path(repo_root).is_dir():
        try:
            os.chdir(repo_root)
            return True
        except OSError:
            pass

    try:
        Path.cwd()
        return True
    except FileNotFoundError:
        print(
            "[graphify watch] Rebuild failed: current working directory "
            "no longer exists and GRAPHIFY_REPO_ROOT is not set."
        )
        return False


# --- Code-update adapters ---------------------------------------------------
#
# Everything below only translates a caller's ask into a ``CorpusGraph`` Code
# update and renders the terminal outcome that comes back. Discovery,
# structural extraction, reconciliation against the active Source-contribution
# ledger, safety checks, and publication all live in ``graphify.generation``,
# which is the sole canonical writer for a Corpus. Nothing here may reach around
# it, and nothing here decides what the graph should contain.

_WATCH_PREFIX = "[graphify watch] "
_HOOK_PREFIX = "[graphify hook] "


def _corpus_for(watch_path: Path) -> "Corpus":
    """Return the Corpus one watched root and its canonical output describe."""
    from graphify.generation import Corpus

    return Corpus(root=watch_path, output=watch_path / _GRAPHIFY_OUT)


def _render_code_update(
    outcome: "TerminalOutcome",
    *,
    prefix: str = _WATCH_PREFIX,
) -> None:
    """Print one Code-update terminal outcome in the calling adapter's voice.

    Rendering is deliberately a closed match over the outcome union rather than
    a message the operation hands over: the owning module publishes facts, and
    each adapter decides how its users should read them. Refusals and failures
    go to stderr so a hook log and a CLI run keep the same stream contract they
    had before Code update moved behind ``CorpusGraph``.
    """
    from graphify.generation import (
        AlreadyCurrent,
        Cancelled,
        CorpusStateAdvanced,
        GenerationPublished,
        OperationFailed,
        PublicationRefused,
        Queued,
    )

    if isinstance(outcome, GenerationPublished):
        # "changed", not "written": publishing a Raw graph generation retires the
        # clustered artifacts of the previous one, and those are reported here
        # too. Only the operation knows which of the two happened to each.
        print(
            f"{prefix}Graph generation published. Canonical artifacts changed: "
            f"{_changed_artifacts(outcome.changed_artifacts)}."
        )
    elif isinstance(outcome, CorpusStateAdvanced):
        print(
            f"{prefix}Corpus state advanced. Canonical artifacts changed: "
            f"{_changed_artifacts(outcome.changed_artifacts)}. "
            "The graph itself is unchanged."
        )
    elif isinstance(outcome, AlreadyCurrent):
        print(f"{prefix}The active Graph generation already covers this request.")
    elif isinstance(outcome, Queued):
        print(
            f"{prefix}Queued as {outcome.request_id}. The Corpus executor covers "
            "it; this process may exit without losing the change set."
        )
    elif isinstance(outcome, Cancelled):
        print(f"{prefix}Cancelled before commit; the active Graph generation is unchanged.")
    elif isinstance(outcome, PublicationRefused):
        print(f"{prefix}Publication refused: {outcome.reason}", file=sys.stderr)
    elif isinstance(outcome, OperationFailed):
        print(f"{prefix}Code update failed: {outcome.reason}", file=sys.stderr)


def _changed_artifacts(changed: "tuple[str, ...]") -> str:
    """Return the canonical artifacts an outcome changed, for display."""
    return ", ".join(changed) if changed else "no canonical artifacts"


def _disclose_stale_evidence_under(
    watch_path: Path,
    *,
    prefix: str = _WATCH_PREFIX,
) -> None:
    """Print the Stale semantic evidence the Corpus under ``watch_path`` discloses.

    A Code update never interprets, so a source it re-derived structurally may
    still carry Semantic evidence describing the file as it was before the
    change. The disclosure wording is owned beside that state in
    ``graphify.generation`` precisely so a watcher line and a report section
    cannot describe one generation differently.

    Named for the Corpus root it takes, because ``cli._disclose_stale_semantic_evidence``
    is the sibling for readers and takes a graph path instead. Like that one it
    sanitizes what it prints and stays silent on any failure: a disclosure must
    never be the reason a completed operation looks like a broken one.
    """
    try:
        from graphify.generation import (
            stale_semantic_disclosure,
            stale_semantic_sources,
        )
        from graphify.security import sanitize_label

        # Ledger identities are validated as portable relative paths, but they
        # still reach a terminal as text, so they go through the same label
        # sanitizer every other rendered graph value does.
        disclosure = stale_semantic_disclosure(
            [
                sanitize_label(source)
                for source in stale_semantic_sources(watch_path / _GRAPHIFY_OUT)
            ]
        )
    except Exception:
        return
    if disclosure:
        print(f"{prefix}{disclosure}")


def _submit_code_update(
    watch_path: Path,
    *,
    changed_paths: "Sequence[Path] | None" = None,
    force: bool = False,
    prefix: str = _WATCH_PREFIX,
) -> bool:
    """Submit one foreground Code update and wait until the request is covered.

    ``WaitUntilCovered`` is what makes a returning call mean something: the
    requested state has been published, by this process or by whichever executor
    got there first. Interactive callers want exactly that, so this is the shape
    the watcher and the ``graphify update`` CLI share.

    Stabilizing the working directory first is not optional here: a relative
    ``watch_path`` is resolved against it, and a detached hook can inherit one
    that has been deleted.
    """
    from graphify.generation import (
        CodeUpdateRequest,
        CorpusGraph,
        WaitUntilCovered,
    )

    if not _stabilize_rebuild_cwd(watch_path):
        return False
    try:
        outcome = CorpusGraph(_corpus_for(watch_path)).code_update(
            CodeUpdateRequest(tuple(changed_paths or ()), force=force),
            completion=WaitUntilCovered(),
        )
    except OSError as exc:
        # An accepted operation reports failure as a terminal outcome, so this
        # only catches the environment refusing before acceptance — an output
        # directory that cannot be created or written, most often. The watcher
        # loop and the CLI both survived that before Code update moved behind
        # the owner, and neither should start showing a traceback for it. A
        # malformed request still raises: that is a caller bug, not an outcome.
        print(f"{prefix}Code update could not start: {exc}", file=sys.stderr)
        return False
    from graphify.generation import covers_the_request

    _render_code_update(outcome, prefix=prefix)
    if not covers_the_request(outcome):
        # Nothing was published, so the active generation is whatever it already
        # was. Its Stale semantic evidence is still true, but printing it under a
        # refusal reads as a finding about this run rather than a standing fact.
        return False
    _disclose_stale_evidence_under(watch_path, prefix=prefix)
    return True


def _background_code_update(
    watch_path: Path,
    *,
    changed_paths: "Sequence[Path] | None" = None,
    force: bool = False,
) -> bool:
    """Durably queue a hook's Code update, then cover it as the Corpus executor.

    The two steps are both real and deliberately ordered. ``ReturnWhenQueued``
    puts the change set — and the ``force`` this hook was told to use — on disk
    before any work starts, so the rebuild watchdog, a reboot, or a killed
    detached process can no longer lose it; the Corpus owns the request from
    that moment rather than this process. The executor pass that follows is the
    same request submitted again, so it coalesces with the queued record and
    covers both, and a failure leaves the record accepted for the next executor
    instead of silently dropping the commit that produced it.

    Returns whether the Corpus ended up covered, so an installed hook can report
    a rebuild the way it always has.
    """
    # Before the queue write, not just before the work: the durable record lives
    # under a ``watch_path`` that may be relative to a working directory a
    # detached hook has already lost.
    if not _stabilize_rebuild_cwd(watch_path):
        return False

    from graphify.generation import CodeUpdateRequest, CorpusGraph, ReturnWhenQueued

    owner = CorpusGraph(_corpus_for(watch_path))
    request = CodeUpdateRequest(tuple(changed_paths or ()), force=force)
    try:
        queued = owner.code_update(request, completion=ReturnWhenQueued())
    except OSError as exc:
        # Nothing is durable yet, so there is no queued work to report and
        # nothing to cover; see _submit_code_update for why this is narrow.
        print(f"{_HOOK_PREFIX}Code update could not be queued: {exc}", file=sys.stderr)
        return False
    _render_code_update(queued, prefix=_HOOK_PREFIX)
    return _submit_code_update(
        watch_path,
        changed_paths=changed_paths,
        force=force,
        prefix=_HOOK_PREFIX,
    )


def _rebuild_code(
    watch_path: Path,
    *,
    changed_paths: "list[Path] | None" = None,
    follow_symlinks: bool = False,
    force: bool = False,
    no_cluster: bool = False,
    acquire_lock: bool = True,
    block_on_lock: bool = False,
) -> bool:
    """Run one Code update for ``watch_path`` and report the legacy boolean.

    Compatibility adapter for hooks installed before Code update moved behind
    ``CorpusGraph``. It keeps this import path and this signature working for the
    documented transition window and does nothing else: the request is
    translated, the operation runs, and its terminal outcome is mapped to the
    True/False these callers were written against. New callers use
    ``CorpusGraph.code_update`` directly.

    Four parameters survive only so an installed hook's call still binds, and
    none of them can change what the operation does any more:

    * ``follow_symlinks`` — discovery policy belongs to the active Graph
      generation, which reads it from the Corpus build policy rather than from
      whichever process happened to trigger the rebuild.
    * ``no_cluster`` — a Code update publishes a Raw graph generation, so there
      is no clustering for this flag to decline. Reclustering completes such a
      generation as a separate operation.
    * ``acquire_lock`` / ``block_on_lock`` — cross-process coordination is owned
      by the operation. A foreground request always waits until it is covered,
      which is what ``block_on_lock=True`` used to ask for and what the callers
      that passed False were settling for a skip instead of.
    """
    return _submit_code_update(
        watch_path,
        changed_paths=changed_paths,
        force=force,
    )


def check_update(watch_path: Path) -> bool:
    """Check for pending semantic update flag and notify the user if set.

    Cron-safe: always returns True so cron jobs do not alarm.
    The marker is a compatibility projection of the active Graph generation's
    own Stale semantic evidence, raised by the Code update that observed it;
    this function only reports it, and only a successful Full extraction clears
    it.
    """
    flag = Path(watch_path) / _GRAPHIFY_OUT / "needs_update"
    if flag.exists():
        print(f"[graphify check-update] Pending non-code changes in {watch_path}.")
        print("[graphify check-update] Run `/graphify --update` to apply semantic re-extraction.")
    return True


def watch(watch_path: Path, debounce: float = 3.0) -> None:
    """
    Watch watch_path for new or modified files and submit the changes.

    Every relevant change is submitted as one Code update; the operation decides
    what its evidence means. The watcher deliberately no longer sorts changes
    into "code" and "needs an LLM" and no longer raises the semantic-pending
    marker itself: a file's extension is a poor proxy for whether the active
    Graph generation holds Semantic evidence for it, and two writers of that
    state could disagree. Stale semantic evidence is disclosed from the
    generation that recorded it instead.

    debounce: seconds to wait after the last change before triggering (avoids
    running on every keystroke when many files are saved at once).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.observers.polling import PollingObserver
        from watchdog.events import FileSystemEventHandler
    except ImportError as e:
        raise ImportError("watchdog not installed. Run: pip install watchdog") from e

    last_trigger: float = 0.0
    pending: bool = False
    changed: set[Path] = set()

    # Load .graphifyignore patterns ONCE at startup so the handler does not
    # re-parse the file on every filesystem event. Watchdog's handler runs on
    # the observer thread and is invoked for every event the OS delivers
    # (Time Machine writes, Docker/Colima VM I/O, Spotlight indexing, …) —
    # without this short-circuit a busy volume can saturate a CPU core
    # discarding events one extension at a time. (gh-928)
    watch_root_for_ignore = watch_path.resolve()
    ignore_patterns = _load_graphifyignore(
        watch_root_for_ignore,
        gitignore=_watcher_honors_vcs_ignores(watch_path),
    )

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal last_trigger, pending
            if event.is_directory:
                return
            path = Path(os.fsdecode(event.src_path))
            # Check .graphifyignore BEFORE the extension/dotfile/out filters so
            # the cheapest short-circuit for users with broad ignore patterns
            # (node_modules/, .venv/, build/, ...) fires first. _is_ignored
            # tolerates absolute paths outside watch_root via its internal
            # relative_to guard, so a stray symlinked event won't raise.
            if ignore_patterns and _is_ignored(path, watch_root_for_ignore, ignore_patterns):
                return
            if path.suffix.lower() not in _WATCHED_EXTENSIONS:
                return
            try:
                filter_parts = path.relative_to(watch_root_for_ignore).parts
            except ValueError:
                filter_parts = path.parts
            if any(part.startswith(".") for part in filter_parts):
                return
            if _GRAPHIFY_OUT in filter_parts:
                return
            last_trigger = time.monotonic()
            pending = True
            changed.add(path)

    handler = Handler()
    # Use polling observer on macOS - FSEvents can miss rapid saves in some editors
    observer = PollingObserver() if sys.platform == "darwin" else Observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    observer.start()

    print(f"[graphify watch] Watching {watch_path.resolve()} - press Ctrl+C to stop")
    print(f"[graphify watch] Changes are submitted as Code updates (no LLM). "
          f"Reinterpreting changed semantic sources still requires /graphify --update.")
    print(f"[graphify watch] Debounce: {debounce}s")

    try:
        while True:
            time.sleep(0.5)
            if pending and (time.monotonic() - last_trigger) >= debounce:
                pending = False
                batch = list(changed)
                changed.clear()
                print(f"\n{_WATCH_PREFIX}{len(batch)} file(s) changed")
                _submit_code_update(watch_path, changed_paths=batch)
    except KeyboardInterrupt:
        print("\n[graphify watch] Stopped.")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watch a folder and auto-update the graphify graph")
    parser.add_argument("path", nargs="?", default=".", help="Folder to watch (default: .)")
    parser.add_argument("--debounce", type=float, default=3.0,
                        help="Seconds to wait after last change before updating (default: 3)")
    args = parser.parse_args()
    watch(Path(args.path), debounce=args.debounce)
