"""graphify command dispatch — every non-install subcommand.

Extracted verbatim from __main__.main(); __main__ now calls dispatch_command(cmd)
after the install/platform dispatch. Kept out of __main__ to shrink the CLI entry
module. The path-redirect (`graphify <path>` -> extract) re-enters via a lazy
import of main to avoid a cli<->__main__ import cycle.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from graphify.generation._manifest import _stamped_manifest_files
from graphify.paths import GRAPHIFY_OUT as _GRAPHIFY_OUT
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported for annotations only; the runtime imports are lazy
    from graphify.generation import (
        CorpusDiscovered,
        CorpusGraph,
        Observation,
        TerminalOutcome,
    )


_SEARCH_NUDGE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'MANDATORY: graphify-out/graph.json exists. You MUST run '
            '`graphify query "<question>"` before grepping raw files. Only grep '
            'after graphify has oriented you, or to modify/debug specific lines.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_READ_NUDGE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'MANDATORY: graphify-out/graph.json exists. You MUST run graphify '
            'before reading source files. Use: `graphify query "<question>"` '
            '(scoped subgraph), `graphify explain "<concept>"`, or '
            '`graphify path "<A>" "<B>"`. Only read raw files after graphify has '
            'oriented you, or to modify/debug specific lines. This rule applies to '
            'subagents too — include it in every subagent prompt involving code '
            'exploration.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_READ_NUDGE_STALE = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            'graphify-out/graph.json exists but may be STALE for this file (the file '
            'changed after the last build, or a semantic source is awaiting '
            'reinterpretation). Prefer `graphify query "<question>"` for orientation — '
            'it names any source carrying Stale semantic evidence. Run `graphify '
            'update` to refresh code structure; docs, papers, and images need '
            '`/graphify --update`, which `graphify update` cannot do. Reading the file '
            'directly is fine.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
# Strict-mode block (opt-in). Claude Code PreToolUse honors
# hookSpecificOutput.permissionDecision == "deny" and shows permissionDecisionReason
# to the model. Fires at most once per session (see _mark_session_denied) so it can
# never strand an agent: the very next read proceeds with the soft nudge.
_READ_DENY = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            'graphify strict mode: this project has a fresh knowledge graph that covers '
            'this file. Run `graphify query "<your question>"` (or `graphify explain` / '
            '`graphify path`) FIRST to orient yourself, then re-issue this Read — it '
            'will be allowed. This block fires at most once per session; reading raw '
            'files to modify or debug specific lines is fine after one query. Apply the '
            'same rule in any subagent prompt that explores code.'
        ),
    }
}, ensure_ascii=False, separators=(",", ":")) + "\n"
_HOOK_SOURCE_EXTS = (
    '.py', '.js', '.cjs', '.ts', '.tsx', '.jsx', '.astro', '.vue', '.svelte', '.go',
    '.rs', '.java', '.rb', '.c', '.h', '.cpp', '.hpp', '.cc', '.cs', '.kt',
    '.swift', '.php', '.scala', '.lua', '.sh', '.md', '.rst', '.txt', '.mdx',
)
_GEMINI_NUDGE_TEXT = (
    'graphify: knowledge graph at graphify-out/. For focused questions, run '
    '`graphify query "<question>"` (scoped subgraph, usually much smaller than '
    'GRAPH_REPORT.md) instead of grepping raw files. Read GRAPH_REPORT.md only '
    'for broad architecture context.'
)

def _default_graph_path() -> str:
    return str(Path(_GRAPHIFY_OUT) / "graph.json")


def _disclose_stale_semantic_evidence(graph_path: Path) -> None:
    """Announce Stale semantic evidence before a reader renders graph content.

    A reader that shows interpreted evidence without saying it predates the file
    it describes lets an answer read as current. The disclosure is derived from
    the active generation's Source-contribution ledger through the owning
    module, and it heads the output so a truncated or piped answer still carries
    it. Silent when the generation is current, unreadable, or has no ledger, so
    no reader can fail because of a disclosure.
    """
    try:
        from graphify.generation import (
            stale_semantic_disclosure,
            stale_semantic_sources,
        )
        from graphify.security import sanitize_label

        sources = stale_semantic_sources(graph_path.parent)
        # Ledger identities are validated as portable relative paths, but they
        # still reach a terminal as text, so they go through the same label
        # sanitizer every other rendered graph value does.
        disclosure = stale_semantic_disclosure(
            [sanitize_label(source) for source in sources]
        )
    except Exception:
        return
    if disclosure:
        print(f"[!] STALE SEMANTIC EVIDENCE: {disclosure}\n")


class _StageTimer:
    """Print per-stage wall-clock timings to stderr when --timing is set (#1490).

    Monotonic (perf_counter), diagnostic-only: emits ``[graphify timing] <stage>:
    N.Ns`` after each stage and a final total. Off by default, so normal output is
    byte-identical and machine-read stdout is untouched.
    """

    def __init__(self, enabled: bool) -> None:
        import time as _time
        self._now = _time.perf_counter
        self.enabled = enabled
        self.start = self._now()
        self._last = self.start

    def mark(self, stage: str) -> None:
        now = self._now()
        if self.enabled:
            print(f"[graphify timing] {stage}: {now - self._last:.1f}s", file=sys.stderr)
        self._last = now

    def total(self) -> None:
        if self.enabled:
            print(f"[graphify timing] total: {self._now() - self.start:.1f}s", file=sys.stderr)


# --- Full-extraction adapters ----------------------------------------------
#
# `graphify extract` is a translation and presentation adapter over the Full
# extraction operation. It parses flags into one ``FullExtractionRequest``,
# renders the ordered observations and the terminal outcome the operation
# reports, and chooses an exit code. Discovery, extraction, interpretation,
# reconciliation, safety checks, and every canonical write live in
# ``graphify.generation``, which is the sole canonical writer for a Corpus.

_EXTRACT_PREFIX = "[graphify extract] "

# How many file names one discovery line shows before it counts the rest.
_NAMED_FILES = 6

# How this command names each evidence origin. The operation reports a closed
# identifier and leaves the wording to whoever is rendering it, which is why the
# phrasing lives here rather than in the observation.
_ORIGIN_WORDING = {
    "corpus": "corpus filesystem",
    "semantic": "semantic provider",
    "postgresql": "PostgreSQL",
    "cargo": "Cargo workspace",
}


def _named_files(entries: "list[str] | tuple[str, ...]") -> str:
    """Return the file names behind a discovery line, capped so it stays one line."""
    names = sorted({Path(entry).name for entry in entries})
    more = f" (+{len(names) - _NAMED_FILES} more)" if len(names) > _NAMED_FILES else ""
    return ", ".join(names[:_NAMED_FILES]) + more


class _ExtractionProgress:
    """Render one Full extraction's lifecycle observations for a terminal.

    A closed match over the observation union, exactly like the terminal-outcome
    rendering beside it: the operation reports facts, and this decides how a CLI
    user should read them. ``stages`` is marked from the same facts so ``--timing``
    measures the operation's real phases rather than the adapter's guesses about
    them, and ``code_only`` is what turns "documents were discovered" from a
    count into the note that they are being skipped on purpose.
    """

    def __init__(self, stages: "_StageTimer", *, code_only: bool = False) -> None:
        """Bind rendering to one run's stage timer and requested evidence."""
        self._stages = stages
        self._code_only = code_only

    def observe(self, observation: "Observation") -> None:
        """Print one lifecycle observation in the extraction CLI's voice."""
        from graphify.generation import (
            CorpusDiscovered,
            EvidenceCollected,
            PublicationStarted,
            SourcesInterpreted,
        )

        if isinstance(observation, CorpusDiscovered):
            self._discovered(observation)
            self._stages.mark("detect")
        elif isinstance(observation, SourcesInterpreted):
            print(
                f"{_EXTRACT_PREFIX}interpreted {observation.interpreted} of "
                f"{observation.requested} semantic source(s)"
            )
        elif isinstance(observation, EvidenceCollected):
            origin = _ORIGIN_WORDING.get(observation.origin.value, observation.origin.value)
            print(
                f"{_EXTRACT_PREFIX}{origin}: {observation.nodes} "
                f"nodes, {observation.edges} edges"
                + (
                    f", {observation.hyperedges} hyperedges"
                    if observation.hyperedges
                    else ""
                )
            )
            self._stages.mark(origin)
        elif isinstance(observation, PublicationStarted):
            print(f"{_EXTRACT_PREFIX}publishing the Graph generation...")

    def _discovered(self, observation: "CorpusDiscovered") -> None:
        """Report what authoritative discovery found in the Corpus."""
        print(
            f"{_EXTRACT_PREFIX}found {observation.sources} source(s), "
            f"{observation.semantic_sources} of them semantic"
        )
        if self._code_only and observation.semantic_sources:
            print(
                f"{_EXTRACT_PREFIX}--code-only: skipping LLM extraction for "
                f"{observation.semantic_sources} doc/paper/image source(s); they "
                "are represented by whatever structure can be derived locally"
            )
        if observation.unclassified:
            print(
                f"{_EXTRACT_PREFIX}{len(observation.unclassified)} file(s) not "
                "classified (no supported extension or shebang), skipped: "
                f"{_named_files(observation.unclassified)}"
            )
        # Operational skips (symlink/office/Workspace) carry a " [reason]"
        # suffix; this line reports only the security-heuristic drops, because
        # those are the ones a user may want to un-flag by renaming (#2106).
        flagged = [entry for entry in observation.skipped if " [" not in entry]
        if flagged:
            print(
                f"{_EXTRACT_PREFIX}{len(flagged)} file(s) skipped as potentially "
                f"sensitive (rename or move if wrongly flagged): "
                f"{_named_files(flagged)}"
            )
        if not observation.complete:
            print(
                f"{_EXTRACT_PREFIX}warning: discovery could not enumerate the "
                "whole Corpus; publication needs --allow-partial",
                file=sys.stderr,
            )


def _render_full_extraction(outcome: "TerminalOutcome") -> bool:
    """Print one Full extraction terminal outcome and say whether it published.

    Returns whether the Corpus is now current with the request, which is what
    the caller turns into an exit code. The wording is this adapter's own — the
    operation publishes facts and each adapter decides how its users read them —
    but the verdict is not: it comes from the owning module, so a CLI exit code
    and the queue's own accounting of covered work cannot disagree. Refusals and
    failures go to stderr so a scripted run keeps the stream contract `graphify
    extract` always had.
    """
    from graphify.generation import (
        AlreadyCurrent,
        Cancelled,
        CorpusStateAdvanced,
        GenerationPublished,
        OperationFailed,
        PublicationRefused,
        Queued,
        covers_the_request,
    )

    if isinstance(outcome, GenerationPublished):
        print(
            f"{_EXTRACT_PREFIX}Graph generation published. Canonical artifacts "
            f"changed: {', '.join(outcome.changed_artifacts) or 'none'}."
        )
    elif isinstance(outcome, CorpusStateAdvanced):
        print(
            f"{_EXTRACT_PREFIX}Corpus state advanced; the graph itself is unchanged."
        )
    elif isinstance(outcome, AlreadyCurrent):
        print(f"{_EXTRACT_PREFIX}The active Graph generation already covers this run.")
    elif isinstance(outcome, PublicationRefused):
        print(f"{_EXTRACT_PREFIX}error: publication refused: {outcome.reason}", file=sys.stderr)
    elif isinstance(outcome, OperationFailed):
        print(f"{_EXTRACT_PREFIX}error: {outcome.reason}", file=sys.stderr)
    elif isinstance(outcome, Cancelled):
        print(
            f"{_EXTRACT_PREFIX}Cancelled before commit; the active Graph "
            "generation is unchanged.",
            file=sys.stderr,
        )
    elif isinstance(outcome, Queued):  # pragma: no cover — extraction refuses this
        print(f"{_EXTRACT_PREFIX}Queued as {outcome.request_id}.")
    return covers_the_request(outcome)


def _published_graph_size(graph_path: Path) -> tuple[int, int]:
    """Return the node and edge counts of the published graph, or zeros.

    Read back from the stable public artifact rather than from anything the
    operation kept to itself: what the CLI reports is what a user (and every
    other tool) can see on disk.
    """
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, 0
    links = data.get("links")
    if links is None:
        links = data.get("edges") or []
    return len(data.get("nodes") or []), len(links)


def _publish_clustered_generation(
    owner: "CorpusGraph",
    graph_path: Path,
    *,
    resolution: float,
    exclude_hubs: "float | None",
    stages: "_StageTimer",
) -> bool:
    """Complete a published Raw generation with community identity and analysis.

    Full extraction publishes a Raw graph generation, because clustering is
    Reclustering's work rather than extraction's. Until that operation moves
    behind ``CorpusGraph`` (issue #13) this prepares the clustered candidate and
    hands it to the owner's documented compatibility handoff, so `graphify
    extract` keeps producing the communities and analysis it always has and the
    owning module remains the only canonical writer.

    Reads the Corpus's own published graph, so the clustered generation is
    derived from exactly the evidence the Raw one admitted rather than from a
    second in-memory build that could differ from it. Returns whether the
    clustered generation was published.
    """
    from graphify.analyze import god_nodes, surprising_connections
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.generation import ReclusteringRequest, covers_the_request
    from graphify.generation._publication import _GraphModel, _Publication

    try:
        raw = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            f"{_EXTRACT_PREFIX}error: cannot read the published graph: {exc}",
            file=sys.stderr,
        )
        return False
    graph = build_from_json(raw, directed=bool(raw.get("directed", False)))
    if graph.number_of_nodes() == 0:
        print(
            f"{_EXTRACT_PREFIX}error: the published graph is empty — extraction "
            "produced no nodes. Possible causes: all files skipped, a binary-only "
            "corpus, or an LLM that returned no edges.",
            file=sys.stderr,
        )
        return False
    communities = cluster(graph, resolution=resolution, exclude_hubs_percentile=exclude_hubs)
    stages.mark("cluster")
    cohesion = score_all(graph, communities)
    # Analysis describes the graph rather than being part of its evidence, so
    # neither of these may cost the caller the generation it just paid for: a
    # failed measure is reported as absent and the clustered generation stands.
    try:
        gods = god_nodes(graph)
    except Exception:
        gods = []
    try:
        surprises = surprising_connections(graph, communities)
    except Exception:
        surprises = []
    stages.mark("analyze")
    outcome = owner.reclustering(
        ReclusteringRequest(),
        _publication=_Publication(
            graph=_GraphModel(graph, communities, force=True),
            analysis={
                "communities": {str(cid): members for cid, members in communities.items()},
                "cohesion": {str(cid): score for cid, score in cohesion.items()},
                "gods": gods,
                "surprises": surprises,
            },
            protect_previous=True,
        ),
    )
    if not covers_the_request(outcome):
        # Every non-covering outcome is a failure for this step, including the
        # ones that carry no reason to quote: the caller asked for a clustered
        # generation and did not get one.
        print(
            f"{_EXTRACT_PREFIX}error: clustering could not be published: "
            f"{getattr(outcome, 'reason', type(outcome).__name__)}",
            file=sys.stderr,
        )
        return False
    print(
        f"{_EXTRACT_PREFIX}wrote {graph_path}: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {len(communities)} communities"
    )
    return True


def _enforce_graph_size_cap_or_exit(gp: Path) -> None:
    """Reject oversized graph files before parsing (CLI exit-on-fail flavor).

    Delegates to ``graphify.security.check_graph_file_size_cap`` and turns the
    raised ``ValueError`` into a CLI-style ``error: ...`` message + exit 1.
    Use this from ``__main__.py`` subcommands that already use the ``print +
    sys.exit(1)`` idiom. Library/MCP/loader callers (``serve._load_graph``,
    ``build``, ``benchmark``, ``tree_html``, ``callflow_html``, ``prs``,
    ``global_graph``, ``watch``, ``export``) call the security helper directly
    and let the ``ValueError`` propagate.
    """
    from graphify.security import check_graph_file_size_cap
    try:
        check_graph_file_size_cap(gp)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
def _hook_strict_enabled(flag: bool) -> bool:
    """Resolve strict mode: GRAPHIFY_HOOK_STRICT env overrides the baked-in flag
    (truthy forces on without a reinstall, falsy is the kill switch); unset defers
    to the flag the installed hook command carried."""
    v = os.environ.get("GRAPHIFY_HOOK_STRICT", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return flag


def _touch_query_stamp(graph_path: "Path") -> None:
    """Record that graphify oriented the agent recently, next to the queried graph.
    The strict guard suppresses its block while this stamp is fresh. Fail-silent."""
    try:
        from graphify.paths import write_text_atomic
        stamp = Path(graph_path).parent / "cache" / "last_query_stamp"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(stamp, str(time.time()))
    except Exception:
        pass


def _query_stamp_fresh() -> bool:
    """True if a query/explain/path ran within GRAPHIFY_HOOK_STRICT_TTL (default
    1800s) — recent orientation, so strict mode does not block this read."""
    from graphify.paths import out_path
    try:
        ttl = float(os.environ.get("GRAPHIFY_HOOK_STRICT_TTL", "1800"))
        return (time.time() - out_path("cache", "last_query_stamp").stat().st_mtime) < ttl
    except Exception:
        return False


def _mark_session_denied(session_id: str) -> bool:
    """Atomically claim a one-time strict block for this session. Returns True only
    on the FIRST call for a given session id (O_EXCL create wins once); every later
    call — or any error — returns False, so a session is blocked at most once and an
    agent can never be stranded. Best-effort GC of markers older than 24h."""
    from graphify.paths import out_path
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:64]
    if not sid:
        return False
    try:
        d = out_path("cache", "hook_sessions")
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(d / f"{sid}.denied"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            cutoff = time.time() - 86400
            for entry in os.scandir(d):
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                except OSError:
                    pass
        except OSError:
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _run_hook_guard(kind: str, strict: bool = False) -> None:
    """Shell-agnostic PreToolUse guard (#522).

    Reads the tool-call JSON from stdin and, when a fresh in-project knowledge graph
    exists, nudges the agent to use graphify instead of grepping/reading raw files.
    Replaces the old inline bash hooks that failed to parse on Windows.

    Fails open everywhere: any error, or a non-matching tool call, prints nothing
    and the caller exits 0, so a legitimate tool call is never blocked by a bug.

    In strict mode (opt-in, Claude Code Read only) the FIRST raw read of indexed,
    in-project, fresh code per session is DENIED with a redirect to `graphify query`
    (permissionDecision), then downgrades to the soft nudge — it fires at most once
    per session and can never strand the agent. Search (Bash) and Glob stay
    nudge-only: a compound shell command has no single parseable target and blocking
    file listing would strand navigation. #1840: reads of out-of-project files are
    ignored, and a graph that is stale for the target file softens to a non-mandatory
    nudge instead of blocking or demanding.
    """
    from graphify.paths import out_path, GRAPHIFY_OUT_NAME
    # Gemini's BeforeTool hook takes no stdin and must ALWAYS return a decision so
    # the tool is never blocked; the graph nudge is appended only when a graph
    # exists. Handled before the stdin read below (which the search/read guards need).
    if kind == "gemini":
        payload = {"decision": "allow"}
        try:
            if out_path("graph.json").is_file():
                payload["additionalContext"] = _GEMINI_NUDGE_TEXT
        except Exception:
            pass
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return
    try:
        d = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except Exception:
        return
    if not isinstance(d, dict):
        return
    t = d.get("tool_input", d)
    if not isinstance(t, dict):
        return
    try:
        if kind == "search":
            cmd_str = str(t.get("command", "") or "")
            # Two input shapes reach this guard (matcher "Bash|Grep", #1986):
            # the Bash tool carries `command`, while Claude Code's dedicated
            # Grep tool carries `pattern` (plus optional path/glob) and no
            # command — a Grep call IS a content search by definition, so it
            # nudges whenever a graph exists. For Bash, keep matching the same
            # set the old `case` matched: *grep*, *ripgrep*, and rg/find/fd/
            # ack/ag as a token (name followed by a space). Nudge-only, even in
            # strict mode — see the docstring.
            is_grep_tool = not cmd_str and bool(t.get("pattern"))
            is_bash_search = any(tok in cmd_str for tok in (
                "grep", "ripgrep", "rg ", "find ", "fd ", "ack ", "ag "))
            if (is_grep_tool or is_bash_search) and out_path("graph.json").is_file():
                sys.stdout.write(_SEARCH_NUDGE)
        elif kind == "read":
            vals = [str(t.get("file_path") or ""), str(t.get("pattern") or ""), str(t.get("path") or "")]
            j = " ".join(vals).lower().replace("\\", "/")
            tails = [
                "." + seg.rsplit(".", 1)[-1]
                for v in vals if v
                for seg in [v.lower().replace("\\", "/").rsplit("/", 1)[-1]]
                if "." in seg
            ]
            under_out = "graphify-out/" in j or (GRAPHIFY_OUT_NAME.lower() + "/") in j
            if under_out or not any(tl in _HOOK_SOURCE_EXTS for tl in tails):
                return
            # #1840 (a): skip files outside the graph's project. cwd (or
            # CLAUDE_PROJECT_DIR, which Claude Code sets) is the project root, since
            # the guard only triggers when graph.json exists relative to cwd. A path
            # candidate that resolves outside that root is out-of-project.
            root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
            try:
                root = root.resolve()
            except (OSError, RuntimeError):
                pass
            path_vals = [str(t.get("file_path") or ""), str(t.get("path") or "")]
            explicit = [v for v in path_vals if v]
            if explicit:
                in_project = False
                for v in explicit:
                    p = Path(v)
                    if not p.is_absolute():
                        in_project = True  # relative -> anchored at cwd == in project
                        break
                    try:
                        p.resolve().relative_to(root)
                        in_project = True
                        break
                    except (ValueError, OSError, RuntimeError):
                        continue
                if not in_project:
                    return
            # One stat for existence + mtime of the graph.
            try:
                gmtime = os.stat(str(out_path("graph.json"))).st_mtime
            except OSError:
                return
            # #1840 (b): stale-for-target -> soften, never block. The target file
            # changed after the last build, or watch flagged the tree.
            stale = False
            fp = str(t.get("file_path") or "")
            if fp:
                try:
                    stale = os.stat(fp).st_mtime > gmtime
                except OSError:
                    stale = False
            try:
                if out_path("needs_update").exists():
                    stale = True
            except Exception:
                pass
            if stale:
                sys.stdout.write(_READ_NUDGE_STALE)
                return
            # Strict block: Read tool only, first time per session, not recently
            # oriented, and the file is demonstrably indexed.
            tool_name = d.get("tool_name")
            if _hook_strict_enabled(strict) and tool_name in (None, "Read") \
                    and not _query_stamp_fresh() \
                    and _target_is_indexed(fp, root) \
                    and _mark_session_denied(str(d.get("session_id") or "")):
                sys.stdout.write(_READ_DENY)
                return
            sys.stdout.write(_READ_NUDGE)
    except Exception:
        pass


def _target_is_indexed(file_path: str, root: "Path") -> bool:
    """Guard the strict deny: only block a read of a file the graph actually indexes.
    Reads manifest.json (cheap, capped); on any doubt (missing/corrupt/oversized
    manifest, unresolvable path) returns True so the once-per-session deny still
    applies — that block is self-limiting, so erring toward it is safe."""
    from graphify.paths import out_path
    if not file_path:
        return True
    try:
        mp = out_path("manifest.json")
        st = mp.stat()
        if st.st_size > 2_000_000:
            return True
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not manifest:
            return True
        p = Path(file_path)
        rels = set()
        try:
            rels.add(p.resolve().relative_to(root).as_posix())
        except (ValueError, OSError, RuntimeError):
            pass
        rels.add(p.name)
        keys = {str(k).replace("\\", "/") for k in manifest}
        abskey = str(p).replace("\\", "/")
        return abskey in keys or any(r and (r in keys or any(k.endswith("/" + r) or k == r for k in keys)) for r in rels)
    except Exception:
        return True
def _clone_repo(
    url: str, branch: str | None = None, out_dir: Path | None = None
) -> Path:
    """Clone a GitHub repo to a local cache dir and return the path.

    Clones into ~/.graphify/repos/<owner>/<repo> by default so repeated
    runs on the same URL reuse the existing clone (git pull instead of clone).
    """
    import subprocess as _sp
    import re as _re

    # Normalise URL — strip trailing .git if present
    url = url.rstrip("/")
    if not url.endswith(".git"):
        git_url = url + ".git"
    else:
        git_url = url
        url = url[:-4]

    # Extract owner/repo from URL
    m = _re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
    if not m:
        print(f"error: not a recognised GitHub URL: {url}", file=sys.stderr)
        sys.exit(1)
    owner, repo = m.group(1), m.group(2)

    if out_dir:
        dest = out_dir
    else:
        dest = Path.home() / ".graphify" / "repos" / owner / repo

    if branch and branch.startswith("-"):
        print(f"error: invalid branch name: {branch!r}", file=sys.stderr)
        sys.exit(1)

    if dest.exists():
        print(f"Repo already cloned at {dest} - pulling latest...", flush=True)
        cmd = ["git", "-C", str(dest), "pull"]
        if branch:
            cmd += ["origin", "--", branch]
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"warning: git pull failed:\n{result.stderr}", file=sys.stderr)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {url} -> {dest} ...", flush=True)
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += ["--", git_url, str(dest)]
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"error: git clone failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    print(f"Ready at: {dest}", flush=True)
    return dest


def _reenter_main() -> None:
    from graphify.__main__ import main
    main()


def dispatch_command(cmd: str) -> None:
    if cmd == "provider":
        from graphify.llm import _custom_providers_path, BACKENDS
        import json as _json
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        global_path = _custom_providers_path(global_=True)

        if subcmd == "list":
            global_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if global_path.is_file():
                try:
                    existing = _json.loads(global_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if not existing:
                print("No custom providers registered.")
            else:
                for name in existing:
                    print(f"  {name}  ({existing[name].get('base_url', '')})")

        elif subcmd == "show":
            name = sys.argv[3] if len(sys.argv) > 3 else ""
            if not name:
                print("Usage: graphify provider show <name>", file=sys.stderr)
                sys.exit(1)
            existing = {}
            if global_path.is_file():
                try:
                    existing = _json.loads(global_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if name not in existing:
                print(f"Provider '{name}' not found.", file=sys.stderr)
                sys.exit(1)
            print(_json.dumps({name: existing[name]}, indent=2))

        elif subcmd == "add":
            args = sys.argv[3:]
            name = args[0] if args and not args[0].startswith("-") else ""
            if not name:
                print("Usage: graphify provider add <name> --base-url URL --default-model MODEL --env-key KEY", file=sys.stderr)
                sys.exit(1)
            if name in BACKENDS:
                print(f"Error: '{name}' is a built-in provider and cannot be overridden.", file=sys.stderr)
                sys.exit(1)
            base_url = ""
            default_model = ""
            env_key = ""
            pricing_input = 0.0
            pricing_output = 0.0
            i = 1
            while i < len(args):
                a = args[i]
                if a == "--base-url" and i + 1 < len(args):
                    base_url = args[i + 1]; i += 2
                elif a.startswith("--base-url="):
                    base_url = a.split("=", 1)[1]; i += 1
                elif a == "--default-model" and i + 1 < len(args):
                    default_model = args[i + 1]; i += 2
                elif a.startswith("--default-model="):
                    default_model = a.split("=", 1)[1]; i += 1
                elif a == "--env-key" and i + 1 < len(args):
                    env_key = args[i + 1]; i += 2
                elif a.startswith("--env-key="):
                    env_key = a.split("=", 1)[1]; i += 1
                elif a == "--pricing-input" and i + 1 < len(args):
                    pricing_input = float(args[i + 1]); i += 2
                elif a == "--pricing-output" and i + 1 < len(args):
                    pricing_output = float(args[i + 1]); i += 2
                else:
                    i += 1
            if not base_url or not default_model or not env_key:
                print("Error: --base-url, --default-model, and --env-key are required.", file=sys.stderr)
                sys.exit(1)
            from graphify.llm import provider_base_url_ok
            if not provider_base_url_ok(base_url, name):
                print(f"Error: refusing to add provider with unsafe base_url {base_url!r}.", file=sys.stderr)
                sys.exit(1)
            global_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if global_path.is_file():
                try:
                    existing = _json.loads(global_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            existing[name] = {
                "base_url": base_url,
                "default_model": default_model,
                "env_key": env_key,
                "pricing": {"input": pricing_input, "output": pricing_output},
                "temperature": 0,
            }
            global_path.write_text(_json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            print(f"Provider '{name}' added. Use with: graphify extract . --backend {name}")

        elif subcmd == "remove":
            name = sys.argv[3] if len(sys.argv) > 3 else ""
            if not name:
                print("Usage: graphify provider remove <name>", file=sys.stderr)
                sys.exit(1)
            existing = {}
            if global_path.is_file():
                try:
                    existing = _json.loads(global_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if name not in existing:
                print(f"Provider '{name}' not found.", file=sys.stderr)
                sys.exit(1)
            del existing[name]
            global_path.write_text(_json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            print(f"Provider '{name}' removed.")

        else:
            print("Usage: graphify provider [add|list|show|remove]", file=sys.stderr)
            if subcmd:
                sys.exit(1)
    elif cmd == "prs":
        from graphify.prs import cmd_prs
        cmd_prs(sys.argv[2:])
    elif cmd == "hook":
        from graphify.hooks import (
            install as hook_install,
            uninstall as hook_uninstall,
            status as hook_status,
        )

        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd == "install":
            print(hook_install(Path(".")))
        elif subcmd == "uninstall":
            print(hook_uninstall(Path(".")))
        elif subcmd == "status":
            print(hook_status(Path(".")))
        else:
            print("Usage: graphify hook [install|uninstall|status]", file=sys.stderr)
            sys.exit(1)
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: graphify query \"<question>\" [--dfs] [--context C] [--budget N] [--graph path]", file=sys.stderr)
            sys.exit(1)
        from graphify.serve import _query_graph_text
        from graphify.security import sanitize_label
        from networkx.readwrite import json_graph
        from graphify import querylog

        question = sys.argv[2]
        use_dfs = "--dfs" in sys.argv
        budget = 2000
        graph_path = _default_graph_path()
        context_filters: list[str] = []
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--budget" and i + 1 < len(args):
                try:
                    budget = int(args[i + 1])
                except ValueError:
                    print(f"error: --budget must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 2
            elif args[i].startswith("--budget="):
                try:
                    budget = int(args[i].split("=", 1)[1])
                except ValueError:
                    print(f"error: --budget must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 1
            elif args[i] == "--context" and i + 1 < len(args):
                context_filters.append(args[i + 1])
                i += 2
            elif args[i].startswith("--context="):
                context_filters.append(args[i].split("=", 1)[1])
                i += 1
            elif args[i] == "--graph" and i + 1 < len(args):
                graph_path = args[i + 1]
                i += 2
            else:
                i += 1
        gp = Path(graph_path).resolve()
        if not gp.exists():
            print(f"error: graph file not found: {gp}", file=sys.stderr)
            sys.exit(1)
        if not gp.suffix == ".json":
            print(f"error: graph file must be a .json file", file=sys.stderr)
            sys.exit(1)
        _enforce_graph_size_cap_or_exit(gp)
        try:
            import json as _json
            import networkx as _nx

            _raw = _json.loads(gp.read_text(encoding="utf-8"))
            if "links" not in _raw and "edges" in _raw:
                _raw = dict(_raw, links=_raw["edges"])
            # `query` deliberately keeps the graph undirected (unlike `path` /
            # `explain`, which force directed=True): BFS/DFS here must explore
            # both callers and callees of the seed node to build useful
            # context, and forcing a DiGraph would make G.neighbors() return
            # successors only, silently dropping every caller-side result for
            # a seed with no outgoing edges. Direction is instead preserved
            # per-edge below (mirrors graphify/build.py's _src/_tgt pattern)
            # so the *rendering* stays correct without narrowing traversal.
            _raw = dict(
                _raw,
                links=[
                    {**link, "_src": link.get("source"), "_tgt": link.get("target")}
                    for link in _raw.get("links", [])
                ],
            )
            try:
                G = json_graph.node_link_graph(_raw, edges="links")
            except TypeError:
                G = json_graph.node_link_graph(_raw)
            try:
                from graphify.build import graph_has_legacy_ids as _legacy
                if _legacy(_raw.get("nodes", [])):
                    print(
                        "[graphify] note: this graph uses the pre-#1504 node-ID scheme; "
                        "rebuild with `graphify extract --force` to get path-qualified IDs "
                        "(fixes same-name-file collisions).",
                        file=sys.stderr,
                    )
            except Exception:
                pass
        except Exception as exc:
            print(f"error: could not load graph: {exc}", file=sys.stderr)
            sys.exit(1)
        import time as _time
        _t0 = _time.perf_counter()
        _mode = "dfs" if use_dfs else "bfs"
        _result = _query_graph_text(
            G,
            question,
            mode=_mode,
            depth=2,
            token_budget=budget,
            context_filters=context_filters,
        )
        querylog.log_query(
            kind="query",
            question=question,
            corpus=str(gp),
            result=_result,
            mode=_mode,
            depth=2,
            token_budget=budget,
            duration_ms=(_time.perf_counter() - _t0) * 1000,
        )
        _touch_query_stamp(gp)
        _disclose_stale_semantic_evidence(gp)
        print(_result)
    elif cmd == "affected":
        if len(sys.argv) < 3:
            print("Usage: graphify affected \"<node-or-label>\" [--relation R] [--depth N] [--graph path]", file=sys.stderr)
            sys.exit(1)
        from graphify.affected import DEFAULT_AFFECTED_RELATIONS, format_affected, load_graph
        query = sys.argv[2]
        graph_path = _default_graph_path()
        depth = 2
        relations: list[str] = []
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--graph" and i + 1 < len(args):
                graph_path = args[i + 1]
                i += 2
            elif args[i].startswith("--graph="):
                graph_path = args[i].split("=", 1)[1]
                i += 1
            elif args[i] == "--depth" and i + 1 < len(args):
                try:
                    depth = int(args[i + 1])
                except ValueError:
                    print("error: --depth must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 2
            elif args[i].startswith("--depth="):
                try:
                    depth = int(args[i].split("=", 1)[1])
                except ValueError:
                    print("error: --depth must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 1
            elif args[i] == "--relation" and i + 1 < len(args):
                relations.append(args[i + 1])
                i += 2
            elif args[i].startswith("--relation="):
                relations.append(args[i].split("=", 1)[1])
                i += 1
            else:
                i += 1
        gp = Path(graph_path).resolve()
        if not gp.exists():
            print(f"error: graph file not found: {gp}", file=sys.stderr)
            sys.exit(1)
        if not gp.suffix == ".json":
            print("error: graph file must be a .json file", file=sys.stderr)
            sys.exit(1)
        try:
            graph = load_graph(gp)
        except Exception as exc:
            print(f"error: could not load graph: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            format_affected(
                graph,
                query,
                relations=relations or DEFAULT_AFFECTED_RELATIONS,
                depth=depth,
            )
        )
    elif cmd in ("god-nodes", "god_nodes"):
        # god_nodes has long been an analyzer (analyze.py), an MCP tool, and a
        # README-advertised capability, but never a CLI subcommand — `graphify
        # god_nodes` fell through to "unknown command" (#2004). Wire it as a
        # read-only graph query, mirroring `affected`.
        from graphify.affected import load_graph
        from graphify.analyze import god_nodes as _god_nodes
        from graphify.security import sanitize_label as _sanitize_label
        graph_path = _default_graph_path()
        top_n = 10
        as_json = "--json" in sys.argv
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--graph" and i + 1 < len(args):
                graph_path = args[i + 1]
                i += 2
            elif args[i].startswith("--graph="):
                graph_path = args[i].split("=", 1)[1]
                i += 1
            elif args[i] == "--top" and i + 1 < len(args):
                try:
                    top_n = int(args[i + 1])
                except ValueError:
                    print("error: --top must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 2
            elif args[i].startswith("--top="):
                try:
                    top_n = int(args[i].split("=", 1)[1])
                except ValueError:
                    print("error: --top must be an integer", file=sys.stderr)
                    sys.exit(1)
                i += 1
            else:
                i += 1
        gp = Path(graph_path).resolve()
        if not gp.exists():
            print(f"error: graph file not found: {gp}", file=sys.stderr)
            sys.exit(1)
        if not gp.suffix == ".json":
            print("error: graph file must be a .json file", file=sys.stderr)
            sys.exit(1)
        try:
            G = load_graph(gp)
        except Exception as exc:
            print(f"error: could not load graph: {exc}", file=sys.stderr)
            sys.exit(1)
        gods = _god_nodes(G, top_n=top_n)
        if as_json:
            print(json.dumps(gods, indent=2))
        else:
            print("God nodes (most connected):")
            for rank, n in enumerate(gods, 1):
                print(f"  {rank}. {_sanitize_label(str(n['label']))} - {n['degree']} edges")
    elif cmd == "save-result":
        # graphify save-result --question Q --answer A [--type T] [--nodes N1 N2 ...]
        #                      [--outcome useful|dead_end|corrected] [--correction TEXT]
        import argparse as _ap

        p = _ap.ArgumentParser(prog="graphify save-result")
        p.add_argument("--question", required=True)
        p.add_argument("--answer", default=None)
        p.add_argument("--answer-file", dest="answer_file", default=None)
        p.add_argument("--type", dest="query_type", default="query")
        p.add_argument("--nodes", nargs="*", default=[])
        p.add_argument("--outcome", choices=("useful", "dead_end", "corrected"), default=None)
        p.add_argument("--correction", default=None)
        p.add_argument("--memory-dir", default=str(Path(_GRAPHIFY_OUT) / "memory"))
        opts = p.parse_args(sys.argv[2:])
        if opts.answer_file:
            opts.answer = Path(opts.answer_file).read_text(encoding="utf-8").strip()
        elif not opts.answer:
            p.error("--answer or --answer-file is required")
        from graphify.ingest import save_query_result as _sqr

        out = _sqr(
            question=opts.question,
            answer=opts.answer,
            memory_dir=Path(opts.memory_dir),
            query_type=opts.query_type,
            source_nodes=opts.nodes or None,
            outcome=opts.outcome,
            correction=opts.correction,
        )
        print(f"Saved to {out}")
    elif cmd == "reflect":
        import argparse as _ap

        p = _ap.ArgumentParser(prog="graphify reflect")
        p.add_argument("--memory-dir", default=str(Path(_GRAPHIFY_OUT) / "memory"))
        p.add_argument(
            "--out",
            default=str(Path(_GRAPHIFY_OUT) / "reflections" / "LESSONS.md"),
        )
        p.add_argument("--graph", default=None)
        p.add_argument("--analysis", default=None)
        p.add_argument("--labels", default=None)
        p.add_argument("--half-life-days", type=float, default=30.0,
                       help="signal weight halves every N days (default 30)")
        p.add_argument("--min-corroboration", type=int, default=2,
                       help="distinct useful results to promote a node to preferred (default 2)")
        p.add_argument("--if-stale", action="store_true",
                       help="skip when LESSONS.md is already newer than every input "
                            "(e.g. the git hook just refreshed it)")
        opts = p.parse_args(sys.argv[2:])
        from graphify.reflect import reflect as _reflect, lessons_fresh as _lessons_fresh

        graph_arg = opts.graph
        if graph_arg is None:
            default_graph = Path(_GRAPHIFY_OUT) / "graph.json"
            if default_graph.exists():
                graph_arg = str(default_graph)

        _gp = Path(graph_arg) if graph_arg else None
        _analysis_path = None
        _labels_path = None
        if _gp is not None:
            _analysis_path = Path(opts.analysis) if opts.analysis else (
                _gp.parent / ".graphify_analysis.json")
            _labels_path = Path(opts.labels) if opts.labels else (
                _gp.parent / ".graphify_labels.json")

        if opts.if_stale and _lessons_fresh(
            Path(opts.out), Path(opts.memory_dir), _gp, _analysis_path, _labels_path
        ):
            print(f"Lessons already up to date -> {opts.out} (skipped; omit --if-stale to force)")
        else:
            out_path, agg = _reflect(
                memory_dir=Path(opts.memory_dir),
                out_path=Path(opts.out),
                graph_path=_gp,
                analysis_path=_analysis_path,
                labels_path=_labels_path,
                half_life_days=opts.half_life_days,
                min_corroboration=opts.min_corroboration,
            )
            c = agg["counts"]
            print(
                f"Reflected {agg['total']} memories "
                f"({c['useful']} useful, {c['dead_end']} dead ends, "
                f"{c['corrected']} corrected) -> {out_path}"
            )
    elif cmd == "path":
        if len(sys.argv) < 4:
            print(
                'Usage: graphify path "<source>" "<target>" [--graph path]',
                file=sys.stderr,
            )
            sys.exit(1)
        from graphify.serve import _pick_scored_endpoint, _score_nodes
        from networkx.readwrite import json_graph
        import networkx as _nx

        source_label = sys.argv[2]
        target_label = sys.argv[3]
        graph_path = _default_graph_path()
        args = sys.argv[4:]
        for i, a in enumerate(args):
            if a == "--graph" and i + 1 < len(args):
                graph_path = args[i + 1]
        gp = Path(graph_path).resolve()
        if not gp.exists():
            print(f"error: graph file not found: {gp}", file=sys.stderr)
            sys.exit(1)
        _enforce_graph_size_cap_or_exit(gp)
        _raw = json.loads(gp.read_text(encoding="utf-8"))
        if "links" not in _raw and "edges" in _raw:
            _raw = dict(_raw, links=_raw["edges"])
        # Force directed so the renderer can recover stored caller→callee
        # direction, and multigraph so exact-pair parallel links (e.g. a
        # `references` and a `calls` edge between the same two nodes) survive load
        # instead of being silently collapsed last-writer-wins — otherwise the
        # printed relation could be one the traversed pair doesn't actually
        # carry (#2074). Local to this read; serve's shared graph is untouched.
        _raw = {**_raw, "directed": True, "multigraph": True}
        try:
            G = json_graph.node_link_graph(_raw, edges="links")
        except TypeError:
            G = json_graph.node_link_graph(_raw)
        # Ahead of every early exit below: "no path found" is itself an answer
        # the reader can misread as settled when part of the graph is stale.
        _disclose_stale_semantic_evidence(gp)
        src_scored = _score_nodes(G, [t.lower() for t in source_label.split()])
        tgt_scored = _score_nodes(G, [t.lower() for t in target_label.split()])
        if not src_scored:
            print(f"No node matching '{source_label}' found.", file=sys.stderr)
            sys.exit(1)
        if not tgt_scored:
            print(f"No node matching '{target_label}' found.", file=sys.stderr)
            sys.exit(1)
        src_nid = _pick_scored_endpoint(G, src_scored, source_label)
        tgt_nid = _pick_scored_endpoint(G, tgt_scored, target_label)
        # Ambiguity guard: when both queries resolve to the same node, the
        # shortest path is trivially zero hops, which is almost never what the
        # caller wanted (see bug #828).
        if src_nid == tgt_nid:
            print(
                f"'{source_label}' and '{target_label}' both resolved to the same "
                f"node '{src_nid}'. Use a more specific label or the exact node ID.",
                file=sys.stderr,
            )
            sys.exit(1)
        for _name, _scored, _nid in (
            ("source", src_scored, src_nid),
            ("target", tgt_scored, tgt_nid),
        ):
            # A close runner-up only made the resolution ambiguous when the raw
            # score head is what got picked; a full-token override was chosen on
            # token coverage, not score, so the head's margin is irrelevant.
            if len(_scored) >= 2 and _nid == _scored[0][1]:
                _top, _runner = _scored[0][0], _scored[1][0]
                if _top > 0 and (_top - _runner) / _top < 0.10:
                    print(
                        f"warning: {_name} match was ambiguous "
                        f"(top score {_top:g}, runner-up {_runner:g})",
                        file=sys.stderr,
                    )
        # Deterministic shortest path (#2074): to_undirected(as_view=True)
        # iterates neighbors via a hash-seeded set union, so among equal-length
        # paths BFS returned an arbitrary route that varied per process. Build a
        # sorted, materialized undirected graph so neighbor order — and thus the
        # chosen path — is canonical for a given graph.json.
        _und = _nx.Graph()
        _und.add_nodes_from(sorted(G.nodes))
        _und.add_edges_from(sorted((min(u, v), max(u, v)) for u, v in G.edges()))
        try:
            path_nodes = _nx.shortest_path(_und, src_nid, tgt_nid)
        except (_nx.NetworkXNoPath, _nx.NodeNotFound):
            print(f"No path found between '{source_label}' and '{target_label}'.")
            sys.exit(0)
        hops = len(path_nodes) - 1
        segments = []
        from graphify.build import edge_datas
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            # Report the ACTUAL stored relation(s) of the traversed pair and
            # direction — never a fabricated `calls` (#2074). A pair may carry
            # several parallel relations; show all, and fall back to an honest
            # "related" when the stored edge has no relation.
            if G.has_edge(u, v):
                datas = edge_datas(G, u, v)
                forward = True
            else:
                datas = edge_datas(G, v, u)
                forward = False
            rels = sorted({d.get("relation") for d in datas if d.get("relation")})
            rel = "/".join(rels) if rels else "related"
            confs = sorted({d.get("confidence") for d in datas if d.get("confidence")})
            conf_str = f" [{'/'.join(confs)}]" if confs else ""
            if i == 0:
                segments.append(G.nodes[u].get("label", u))
            if forward:
                segments.append(f"--{rel}{conf_str}--> {G.nodes[v].get('label', v)}")
            else:
                segments.append(f"<--{rel}{conf_str}-- {G.nodes[v].get('label', v)}")
        print(f"Shortest path ({hops} hops):\n  " + " ".join(segments))
        from graphify import querylog
        querylog.log_query(
            kind="path",
            question=f"{sys.argv[2]} -> {sys.argv[3]}",
            corpus=str(gp),
            nodes_returned=hops,
        )
        _touch_query_stamp(gp)

    elif cmd == "explain":
        if len(sys.argv) < 3:
            print('Usage: graphify explain "<node>" [--graph path]', file=sys.stderr)
            sys.exit(1)
        from graphify.serve import _find_node
        from networkx.readwrite import json_graph

        label = sys.argv[2]
        graph_path = _default_graph_path()
        args = sys.argv[3:]
        for i, a in enumerate(args):
            if a == "--graph" and i + 1 < len(args):
                graph_path = args[i + 1]
        gp = Path(graph_path).resolve()
        if not gp.exists():
            print(f"error: graph file not found: {gp}", file=sys.stderr)
            sys.exit(1)
        _enforce_graph_size_cap_or_exit(gp)
        _raw = json.loads(gp.read_text(encoding="utf-8"))
        if "links" not in _raw and "edges" in _raw:
            _raw = dict(_raw, links=_raw["edges"])
        # Force directed so the renderer can recover stored caller→callee direction.
        _raw = {**_raw, "directed": True}
        try:
            G = json_graph.node_link_graph(_raw, edges="links")
        except TypeError:
            G = json_graph.node_link_graph(_raw)
        _disclose_stale_semantic_evidence(gp)
        matches = _find_node(G, label)
        if not matches:
            print(f"No node matching '{label}' found.")
            sys.exit(0)
        nid = matches[0]
        d = G.nodes[nid]
        print(f"Node: {d.get('label', nid)}")
        print(f"  ID:        {nid}")
        print(
            f"  Source:    {d.get('source_file', '')} {d.get('source_location', '')}".rstrip()
        )
        print(f"  Type:      {d.get('file_type', '')}")
        print(f"  Community: {d.get('community_name') or d.get('community', '')}")
        # Work-memory overlay: a derived experiential hint from `graphify reflect`,
        # merged in display-only from the .graphify_learning.json sidecar next to
        # graph.json. No line when the node has no overlay entry.
        try:
            from graphify.reflect import load_learning_overlay as _llo
            from graphify.security import sanitize_label as _sl
            _overlay = _llo(gp)
            _entry = _overlay.get(str(nid))
            if _entry:
                _status = _sl(str(_entry.get("status", "")))
                if _status == "contested":
                    _line = (f"  Lesson: contested (useful {_entry.get('uses', 0)} / "
                             f"dead-end {_entry.get('neg', 0)})")
                elif _status == "preferred":
                    _line = (f"  Lesson: preferred source (start here) — "
                             f"{_entry.get('uses', 0)} useful, score={_entry.get('score', 0)}")
                else:
                    _line = (f"  Lesson: {_status or 'tentative'} — "
                             f"{_entry.get('uses', 0)} useful, score={_entry.get('score', 0)}")
                if _entry.get("stale"):
                    _line += " [code changed since — re-verify]"
                print(_line)
        except Exception:
            pass
        print(f"  Degree:    {G.degree(nid)}")
        from graphify.build import edge_data
        connections: list[tuple[str, str, dict]] = []  # (direction, neighbor_id, edge_data)
        for nb in G.successors(nid):
            connections.append(("out", nb, edge_data(G, nid, nb)))
        for nb in G.predecessors(nid):
            connections.append(("in", nb, edge_data(G, nb, nid)))
        if connections:
            print(f"\nConnections ({len(connections)}):")
            connections.sort(key=lambda c: G.degree(c[1]), reverse=True)
            for direction, nb, edata in connections[:20]:
                rel = edata.get("relation", "")
                conf = edata.get("confidence", "")
                arrow = "-->" if direction == "out" else "<--"
                # Append the edge's location — the actual call/import/reference
                # SITE (in the caller's file for an incoming call), not a def
                # line (#BUG1). Labeled by [rel] so the meaning is unambiguous.
                loc = edata.get("source_location") or ""
                sfile = edata.get("source_file") or ""
                at = f" {sfile}:{loc}" if loc else ""
                print(f"  {arrow} {G.nodes[nb].get('label', nb)} [{rel}] [{conf}]{at}")
            if len(connections) > 20:
                remainder = connections[20:]
                print(f"  ... and {len(remainder)} more")
                # #2009: a bare count silently hides the answer on high-degree
                # nodes ("who calls this, what's the impact?"). Group the cut
                # connections by direction + file so their shape is visible
                # without falling back to a repo-wide grep.
                by_file: dict[tuple[str, str], int] = {}
                for direction, _nb, edata in remainder:
                    sfile = edata.get("source_file") or "(unknown file)"
                    key = (direction, sfile)
                    by_file[key] = by_file.get(key, 0) + 1
                # Count desc, then (direction, file) so equal-count groups have a
                # byte-stable order (not the degree-derived insertion order).
                grouped = sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0]))
                print("  Grouped by file:")
                for (direction, sfile), count in grouped[:20]:
                    arrow = "-->" if direction == "out" else "<--"
                    noun = "connection" if count == 1 else "connections"
                    print(f"    {arrow} {sfile}: {count} {noun}")
                if len(grouped) > 20:
                    print(f"    ... and {len(grouped) - 20} more files")
        from graphify import querylog
        querylog.log_query(
            kind="explain",
            question=sys.argv[2],
            corpus=str(gp),
            nodes_returned=len(connections),
        )
        _touch_query_stamp(gp)

    elif cmd == "diagnose":
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd != "multigraph":
            print(
                "Usage: graphify diagnose multigraph "
                "[--graph path] [--json] [--max-examples N] "
                "[--directed] [--undirected] [--extract-path path]",
                file=sys.stderr,
            )
            sys.exit(1)

        graph_path = Path(_default_graph_path())
        max_examples = 5
        directed: bool | None = None
        direction_flag: str | None = None
        json_output = False
        extract_path: Path | None = None

        i = 3
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--graph":
                i += 1
                if i >= len(sys.argv):
                    print("error: --graph requires a path", file=sys.stderr)
                    sys.exit(1)
                graph_path = Path(sys.argv[i])
            elif arg == "--json":
                json_output = True
            elif arg == "--max-examples":
                i += 1
                if i >= len(sys.argv):
                    print("error: --max-examples requires an integer", file=sys.stderr)
                    sys.exit(1)
                try:
                    max_examples = int(sys.argv[i])
                except ValueError:
                    print("error: --max-examples requires an integer", file=sys.stderr)
                    sys.exit(1)
                if max_examples < 0:
                    print("error: --max-examples must be >= 0", file=sys.stderr)
                    sys.exit(1)
            elif arg == "--directed":
                if direction_flag == "undirected":
                    print(
                        "error: --directed and --undirected are mutually exclusive",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                direction_flag = "directed"
                directed = True
            elif arg == "--undirected":
                if direction_flag == "directed":
                    print(
                        "error: --directed and --undirected are mutually exclusive",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                direction_flag = "undirected"
                directed = False
            elif arg == "--extract-path":
                i += 1
                if i >= len(sys.argv):
                    print("error: --extract-path requires a path", file=sys.stderr)
                    sys.exit(1)
                extract_path = Path(sys.argv[i])
            else:
                print(f"error: unknown diagnose option {arg}", file=sys.stderr)
                sys.exit(1)
            i += 1

        from graphify.diagnostics import (
            diagnose_file,
            format_diagnostic_json,
            format_diagnostic_report,
        )

        try:
            summary = diagnose_file(
                graph_path,
                directed=directed,
                root=Path(".").resolve(),
                max_examples=max_examples,
                extract_path=extract_path,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

        if json_output:
            print(json.dumps(format_diagnostic_json(summary), indent=2))
        else:
            print(format_diagnostic_report(summary))

    elif cmd == "add":
        if len(sys.argv) < 3:
            print(
                "Usage: graphify add <url> [--author Name] [--contributor Name] [--dir ./raw]",
                file=sys.stderr,
            )
            sys.exit(1)
        from graphify.ingest import ingest as _ingest

        url = sys.argv[2]
        author: str | None = None
        contributor: str | None = None
        target_dir = Path("raw")
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--author" and i + 1 < len(args):
                author = args[i + 1]
                i += 2
            elif args[i] == "--contributor" and i + 1 < len(args):
                contributor = args[i + 1]
                i += 2
            elif args[i] == "--dir" and i + 1 < len(args):
                target_dir = Path(args[i + 1])
                i += 2
            else:
                i += 1
        try:
            saved = _ingest(url, target_dir, author=author, contributor=contributor)
            print(f"Saved to {saved}")
            print("Run /graphify --update in your AI assistant to update the graph.")
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "watch":
        watch_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
        if not watch_path.exists():
            print(f"error: path not found: {watch_path}", file=sys.stderr)
            sys.exit(1)
        from graphify.watch import watch as _watch

        try:
            _watch(watch_path)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    elif cmd in ("cluster-only", "label"):
        # `label` is `cluster-only` that always (re)generates community names with
        # the configured backend, even when a .graphify_labels.json already exists.
        force_relabel = cmd == "label"
        # Mirror the tree/export arg-parsing pattern: walk argv so flags and
        # the optional positional path can appear in any order (#724).
        no_viz = "--no-viz" in sys.argv
        no_label = "--no-label" in sys.argv
        missing_only = "--missing-only" in sys.argv
        co_timing = "--timing" in sys.argv
        _backend_arg = next((a for a in sys.argv if a.startswith("--backend=")), None)
        label_backend = _backend_arg.split("=", 1)[1] if _backend_arg else None
        _model_arg = next((a for a in sys.argv if a.startswith("--model=")), None)
        label_model = _model_arg.split("=", 1)[1] if _model_arg else None
        _min_cs_arg = next((a for a in sys.argv if a.startswith("--min-community-size=")), None)
        min_community_size = int(_min_cs_arg.split("=")[1]) if _min_cs_arg else 3
        args = sys.argv[2:]
        watch_path: Path | None = None
        graph_override: Path | None = None
        co_resolution: float = 1.0
        co_exclude_hubs: float | None = None
        label_max_concurrency: int = 4
        label_batch_size: int = 100
        i_arg = 0
        while i_arg < len(args):
            a = args[i_arg]
            if a == "--graph" and i_arg + 1 < len(args):
                graph_override = Path(args[i_arg + 1]); i_arg += 2
            elif a == "--backend" and i_arg + 1 < len(args):
                label_backend = args[i_arg + 1]; i_arg += 2
            elif a.startswith("--backend="):
                label_backend = a.split("=", 1)[1]; i_arg += 1
            elif a == "--model" and i_arg + 1 < len(args):
                label_model = args[i_arg + 1]; i_arg += 2
            elif a.startswith("--model="):
                label_model = a.split("=", 1)[1]; i_arg += 1
            elif a == "--resolution" and i_arg + 1 < len(args):
                co_resolution = float(args[i_arg + 1]); i_arg += 2
            elif a.startswith("--resolution="):
                co_resolution = float(a.split("=", 1)[1]); i_arg += 1
            elif a == "--exclude-hubs" and i_arg + 1 < len(args):
                co_exclude_hubs = float(args[i_arg + 1]); i_arg += 2
            elif a.startswith("--exclude-hubs="):
                co_exclude_hubs = float(a.split("=", 1)[1]); i_arg += 1
            elif a == "--max-concurrency" and i_arg + 1 < len(args):
                label_max_concurrency = int(args[i_arg + 1]); i_arg += 2
            elif a.startswith("--max-concurrency="):
                label_max_concurrency = int(a.split("=", 1)[1]); i_arg += 1
            elif a == "--batch-size" and i_arg + 1 < len(args):
                label_batch_size = int(args[i_arg + 1]); i_arg += 2
            elif a.startswith("--batch-size="):
                label_batch_size = int(a.split("=", 1)[1]); i_arg += 1
            elif a in ("--no-viz", "--missing-only") or a.startswith("--min-community-size="):
                i_arg += 1
            elif a.startswith("--"):
                i_arg += 1
            elif watch_path is None:
                watch_path = Path(a); i_arg += 1
            else:
                i_arg += 1
        if watch_path is None:
            watch_path = Path(".")
        graph_json = graph_override if graph_override is not None else watch_path / _GRAPHIFY_OUT / "graph.json"
        if not graph_json.exists():
            print(
                f"error: no graph found at {graph_json} — run /graphify first",
                file=sys.stderr,
            )
            sys.exit(1)
        from networkx.readwrite import json_graph as _jg
        from graphify.build import build_from_json
        from graphify.cluster import cluster, score_all, remap_communities_to_previous
        from graphify.analyze import (
            god_nodes,
            surprising_connections,
            suggest_questions,
        )
        from graphify.report import generate
        from graphify.export import to_html

        stages = _StageTimer(co_timing)
        print("Loading existing graph...")
        # Solution 3 (#1019): don't hard-exit on an oversized graph.json here.
        # Core outputs (graph.json + GRAPH_REPORT.md) still get written; the
        # graph.html render below falls back to the community-aggregation view
        # (node_limit=5000) when over the cap.
        from graphify.security import check_graph_file_size_cap as _check_cap
        _over_cap = False
        try:
            _check_cap(graph_json)
        except ValueError:
            _over_cap = True
            try:
                _over_cap_bytes = graph_json.stat().st_size
            except OSError:
                _over_cap_bytes = -1
            print(
                f"warning: graph.json exceeds cap ({_over_cap_bytes} bytes); "
                f"falling back to community-aggregation view (node_limit=5000)",
                file=sys.stderr,
            )
        _raw = json.loads(graph_json.read_text(encoding="utf-8"))
        _directed = bool(_raw.get("directed", False))
        G = build_from_json(_raw, directed=_directed)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        stages.mark("load")
        print("Re-clustering...")
        communities = cluster(G, resolution=co_resolution, exclude_hubs_percentile=co_exclude_hubs)
        # Mirror the watch/update path (#822): map new cids to prior ones by
        # node-overlap so the existing .graphify_labels.json keeps attaching
        # to the same conceptual community after re-clustering. Without this,
        # labels follow raw cid index and become misaligned whenever the
        # graph has changed between labeling and cluster-only (#1027).
        previous_node_community = {
            n["id"]: n["community"]
            for n in _raw.get("nodes", [])
            if n.get("community") is not None and n.get("id") is not None
        }
        if previous_node_community:
            communities = remap_communities_to_previous(communities, previous_node_community)
        stages.mark("cluster")
        cohesion = score_all(G, communities)
        gods = god_nodes(G)
        surprises = surprising_connections(G, communities)
        stages.mark("analyze")
        # Where outputs (GRAPH_REPORT.md, re-clustered graph.json, labels,
        # analysis, html) land. When `--graph` points at a graph INSIDE a
        # graphify-out/ dir (another project/tenant's output), write beside it,
        # not into a stray graphify-out/ in the CWD (#1747). But when `--graph`
        # points at an arbitrary path — e.g. a `backup/graph.json` archived
        # before re-clustering (#934) — fall back to the CWD's graphify-out/,
        # which is the restore-into-place workflow that test pins. The default
        # (no --graph) case already has graph_json under watch_path/graphify-out.
        _out_name = Path(_GRAPHIFY_OUT).name
        if graph_override is not None and graph_json.parent.name == _out_name:
            out = graph_json.parent
        else:
            out = watch_path / _GRAPHIFY_OUT
        out.mkdir(parents=True, exist_ok=True)
        labels_path = out / ".graphify_labels.json"
        existing_labels: dict[int, str] = {}
        if labels_path.exists():
            try:
                existing_labels = {
                    int(k): v
                    for k, v in json.loads(labels_path.read_text(encoding="utf-8")).items()
                    if isinstance(v, str)
                }
            except Exception:
                existing_labels = {}
        # Accumulate token usage from the labeling LLM calls so cluster-only mode
        # reports real cost instead of a hardcoded zero (#1694). Stays {0, 0} on
        # the reuse / no-label paths, which make no LLM calls.
        label_token_usage = {"input": 0, "output": 0}
        # #2073: a --no-label run produces only "Community N" placeholders.
        # Persisting them (plus a matching .sig) made the reuse branch treat them
        # as fresh forever, permanently blocking real labeling on later runs.
        placeholder_only = False
        if labels_path.exists() and not force_relabel:
            # Reuse saved labels, but don't blindly trust them: the graph may have
            # been re-scoped/re-clustered since labeling, in which case a cid now
            # covers a DIFFERENT community and its old (LLM) name is wrong (#label-stale).
            # Validate each community against the membership signature saved beside the
            # labels; any community that changed (or has no saved label) is renamed by
            # its current hub — deterministic and correct-by-construction — and the user
            # is told to `graphify label` for fresh LLM names. Unchanged communities keep
            # their saved label. When no signature sidecar exists (labels predate this),
            # fall back to hub-filling only the communities missing a label.
            from graphify.cluster import community_member_sigs, label_communities_by_hub
            sig_path = labels_path.parent / (labels_path.name + ".sig")
            saved_sigs: dict[int, str] = {}
            if sig_path.exists():
                try:
                    saved_sigs = {
                        int(k): v for k, v in
                        json.loads(sig_path.read_text(encoding="utf-8")).items()
                        if isinstance(v, str)
                    }
                except Exception:
                    saved_sigs = {}
            cur_sigs = community_member_sigs(communities)
            count_mismatch = len(existing_labels) != len(communities)
            labels = {}
            hub_labels: dict[int, str] | None = None
            changed = 0
            for cid in communities:
                # A persisted "Community {cid}" is a placeholder, not an earned
                # label — treat it as absent so the hub labeler replaces it and an
                # already-polluted sidecar (e.g. from a prior --no-label run) heals
                # instead of suppressing real labels forever (#2073).
                have_label = (
                    cid in existing_labels
                    and existing_labels[cid] != f"Community {cid}"
                )
                if saved_sigs:
                    # Precise: the membership signature tells us if this exact
                    # community changed since it was labeled.
                    fresh = have_label and saved_sigs.get(cid) == cur_sigs.get(cid)
                else:
                    # No signature sidecar (labels predate it). A differing community
                    # COUNT means the labels describe a different clustering, so a cid's
                    # old label can't be trusted; equal count is the best "same" signal.
                    fresh = have_label and not count_mismatch
                if fresh:
                    labels[cid] = existing_labels[cid]
                else:
                    if hub_labels is None:
                        hub_labels = label_communities_by_hub(G, communities)
                    labels[cid] = hub_labels[cid]
                    if have_label:
                        changed += 1
            if changed:
                print(
                    f"[graphify] community set changed since labeling "
                    f"({len(existing_labels)} saved labels, {len(communities)} communities now; "
                    f"renamed {changed} community(ies) by their hub). "
                    f"Run `graphify label` to refresh names with the LLM.",
                    file=sys.stderr,
                )
        elif no_label and not force_relabel:
            labels = {cid: f"Community {cid}" for cid in communities}
            placeholder_only = True
        else:
            # No labels file yet (or `graphify label` forced a refresh). When run
            # standalone there is no orchestrating agent to do skill.md Step 5, so
            # auto-name communities rather than leave "Community N" (#1097).
            from graphify.cluster import label_communities_by_hub
            from graphify.llm import generate_community_labels
            print("Labeling communities...")
            # Deterministic, LLM-free base labels: name each community after its
            # highest-degree hub, so the report is readable even with no backend
            # (previously bare "Community N"). A configured LLM backend overrides these
            # with richer names below; its no-backend placeholder fallback does NOT.
            hub_labels = label_communities_by_hub(G, communities)
            label_communities_input = communities
            labels = dict(hub_labels)
            if missing_only:
                labels = {
                    cid: existing_labels.get(cid, hub_labels[cid])
                    for cid in communities
                }
                label_communities_input = {
                    cid: members
                    for cid, members in communities.items()
                    if cid not in existing_labels or existing_labels.get(cid) == f"Community {cid}"
                }
            generated_labels, _ = generate_community_labels(
                G, label_communities_input, backend=label_backend, model=label_model, gods=gods,
                max_concurrency=label_max_concurrency, batch_size=label_batch_size,
                usage_out=label_token_usage,
            )
            # Only let the LLM OVERRIDE where it produced a real name — its no-backend
            # fallback returns "Community {cid}" placeholders, which must not clobber
            # the deterministic hub labels.
            labels.update({
                cid: v for cid, v in generated_labels.items()
                if v and v != f"Community {cid}"
            })
        stages.mark("label")
        questions = suggest_questions(G, communities, labels)
        tokens = label_token_usage
        from graphify.export import _git_head as _gh
        _commit = _gh()
        from graphify.report import load_learning_for_report as _llfr
        from graphify.report import load_stale_semantic_sources as _lsss
        report = generate(G, communities, cohesion, labels, gods, surprises,
                          {"warning": "cluster-only mode — file stats not available"},
                          tokens, str(watch_path), suggested_questions=questions,
                          min_community_size=min_community_size, built_at_commit=_commit,
                          learning=_llfr(out / "graph.json"),
                          stale_semantic_sources=_lsss(out / "graph.json"))
        analysis = {
            "communities": {str(k): v for k, v in communities.items()},
            "cohesion": {str(k): v for k, v in cohesion.items()},
            "gods": gods,
            "surprises": surprises,
            "questions": questions,
        }
        # Canonical graph, report, analysis, and label state cross the CorpusGraph
        # boundary together. HTML below remains a post-commit derivative.
        from graphify.generation import (
            Corpus,
            CorpusGraph,
            OperationFailed,
            PublicationRefused,
            ReclusteringRequest,
        )
        from graphify.generation._publication import _GraphModel, _Publication
        from graphify.cluster import community_member_sigs as _cms

        _label_payload = (
            {str(k): v for k, v in labels.items()}
            if not placeholder_only
            else None
        )
        _signature_payload = (
            {str(k): v for k, v in _cms(communities).items()}
            if not placeholder_only
            else None
        )
        _outcome = CorpusGraph(
            Corpus(root=watch_path, output=out)
        ).reclustering(
            ReclusteringRequest(),
            _publication=_Publication(
                graph=_GraphModel(
                    G,
                    communities,
                    community_labels=labels,
                ),
                report=report,
                analysis=analysis,
                labels=_label_payload,
                label_signatures=_signature_payload,
                protect_previous=True,
            ),
        )
        if isinstance(_outcome, (PublicationRefused, OperationFailed)):
            print(f"error: {_outcome.reason}", file=sys.stderr)
            sys.exit(1)
        stages.mark("report")

        # Mirror watch.py pattern: gate to_html so core outputs (graph.json +
        # GRAPH_REPORT.md) always land. Honor --no-viz explicitly; otherwise
        # fall back to ValueError handling so an oversized graph doesn't crash
        # the CLI mid-write and leave a stale graph.html on disk.
        html_target = out / "graph.html"
        if no_viz:
            if html_target.exists():
                html_target.unlink()
            stages.mark("export"); stages.total()
            print(f"Done - {len(communities)} communities. GRAPH_REPORT.md and graph.json updated (--no-viz; graph.html removed).")
        else:
            try:
                # Over-cap fallback (#1019): force the community-aggregation
                # path so an oversized graph still renders a usable graph.html.
                _node_limit = 5000 if _over_cap else None
                to_html(G, communities, str(html_target), community_labels=labels or None,
                        node_limit=_node_limit)
                stages.mark("export"); stages.total()
                print(f"Done - {len(communities)} communities. GRAPH_REPORT.md, graph.json and graph.html updated.")
            except ValueError as viz_err:
                if html_target.exists():
                    html_target.unlink()
                print(f"Skipped graph.html: {viz_err}")
                stages.mark("export"); stages.total()
                print(f"Done - {len(communities)} communities. GRAPH_REPORT.md and graph.json updated.")

    elif cmd == "update":
        force = os.environ.get("GRAPHIFY_FORCE", "").lower() in ("1", "true", "yes")
        no_cluster = False
        args = sys.argv[2:]
        watch_arg: str | None = None
        for a in args:
            if a == "--force":
                force = True
                continue
            if a == "--no-cluster":
                no_cluster = True
                continue
            if a.startswith("-"):
                print(f"error: unknown update option: {a}", file=sys.stderr)
                sys.exit(2)
            if watch_arg is not None:
                print("error: update accepts at most one path argument", file=sys.stderr)
                sys.exit(2)
            watch_arg = a

        if watch_arg is not None:
            watch_path = Path(watch_arg)
        else:
            # Try to recover the scan root saved by the last full build. The
            # marker records the absolute Corpus the owning module published, so
            # a graphify-out/ that travelled to another checkout names a root
            # that is not there any more; fall back to the working directory
            # rather than refusing to update a graph that is otherwise fine
            # (#777 kept this portable by storing the relative spelling, which a
            # Corpus identity cannot do without being retargetable by a chdir).
            saved = Path(_GRAPHIFY_OUT) / ".graphify_root"
            recorded = (
                Path(saved.read_text(encoding="utf-8").strip())
                if saved.exists()
                else None
            )
            watch_path = recorded if recorded is not None and recorded.exists() else Path(".")
        if not watch_path.exists():
            print(f"error: path not found: {watch_path}", file=sys.stderr)
            sys.exit(1)
        from graphify.watch import _submit_code_update

        print(f"Re-extracting code files in {watch_path} (no LLM needed)...")
        if no_cluster:
            # Accepted for compatibility and now implied: a Code update publishes
            # a Raw graph generation, which is exactly what this flag asked for.
            print("Note: --no-cluster is implied; a Code update publishes a Raw graph generation.")
        # The interactive shape of the shared adapter: it submits with
        # WaitUntilCovered, so this command returns only once the requested state
        # has been published — by this process or by whichever executor covered it
        # first — rather than exiting silently when a hook-driven rebuild happens
        # to hold the Corpus lease. ``prefix=""`` keeps the CLI's plain voice.
        if _submit_code_update(watch_path, force=force, prefix=""):
            print("Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.")
            if not (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("MOONSHOT_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("GRAPHIFY_NO_TIPS")
            ):
                print("Tip: set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini for semantic extraction.")
        else:
            # The outcome itself was already rendered above; this is the exit
            # contract scripts depend on, not a second explanation.
            print(
                "The Corpus was not updated — see the refusal or failure above.",
                file=sys.stderr,
            )
            sys.exit(1)

    elif cmd == "hook-check":
        # Codex Desktop rejects hookSpecificOutput.additionalContext on PreToolUse.
        # Keep this as a cross-platform no-op so installed hooks never break Bash
        # tool calls. Graph guidance reaches the agent via AGENTS.md / skill instead.
        sys.exit(0)
    elif cmd == "hook-guard":
        # Shell-agnostic Claude/Codebuddy PreToolUse guard (#522). Replaces the old
        # inline-bash hooks that failed on Windows. Prints an additionalContext nudge
        # toward graphify when a fresh in-project graph exists; always exits 0. In
        # strict mode (opt-in, `hook-guard read --strict`) it blocks the first raw
        # read per session via the JSON permissionDecision payload — never via exit
        # code — and downgrades to the nudge thereafter.
        _run_hook_guard(
            sys.argv[2] if len(sys.argv) > 2 else "",
            strict="--strict" in sys.argv[3:],
        )
        sys.exit(0)
    elif cmd == "check-update":
        if len(sys.argv) < 3:
            print("Usage: graphify check-update <path>", file=sys.stderr)
            sys.exit(1)
        from graphify.watch import check_update

        check_update(Path(sys.argv[2]).resolve())
        sys.exit(0)
    elif cmd == "tree":
        # Emit a D3 v7 collapsible-tree HTML view of graph.json:
        # expand-all / collapse-all / reset-view buttons, multi-line
        # wrapText labels with separately-coloured name + count,
        # depth-based palette, click-to-toggle subtree, hover inspector
        # showing top-K outbound edges per symbol.
        from typing import Optional as _Opt
        from graphify.tree_html import write_tree_html, DEFAULT_MAX_CHILDREN
        graph_path = Path(_GRAPHIFY_OUT) / "graph.json"
        output_path: "_Opt[Path]" = None
        root: "_Opt[str]" = None
        max_children = DEFAULT_MAX_CHILDREN
        top_k_edges = 0
        project_label: "_Opt[str]" = None
        args = sys.argv[2:]
        i_arg = 0
        while i_arg < len(args):
            a = args[i_arg]
            if a == "--graph" and i_arg + 1 < len(args):
                graph_path = Path(args[i_arg + 1]); i_arg += 2
            elif a == "--output" and i_arg + 1 < len(args):
                output_path = Path(args[i_arg + 1]); i_arg += 2
            elif a == "--root" and i_arg + 1 < len(args):
                root = args[i_arg + 1]; i_arg += 2
            elif a == "--max-children" and i_arg + 1 < len(args):
                max_children = int(args[i_arg + 1]); i_arg += 2
            elif a == "--top-k-edges" and i_arg + 1 < len(args):
                top_k_edges = int(args[i_arg + 1]); i_arg += 2
            elif a == "--label" and i_arg + 1 < len(args):
                project_label = args[i_arg + 1]; i_arg += 2
            elif a in ("-h", "--help"):
                print("Usage: graphify tree [--graph PATH] [--output HTML]")
                print("  --graph PATH         path to graph.json (default graphify-out/graph.json)")
                print("  --output HTML        output path (default graphify-out/GRAPH_TREE.html)")
                print("  --root PATH          filesystem root (default: longest common dir of all source_files)")
                print("  --max-children N     cap visible children per node (default 200)")
                print("  --top-k-edges N      pre-compute top-K outbound edges per symbol (default 12)")
                print("  --label NAME         project label shown in the page header")
                return
            else:
                i_arg += 1
        if not graph_path.is_file():
            print(f"error: graph.json not found at {graph_path}", file=sys.stderr)
            sys.exit(1)
        _enforce_graph_size_cap_or_exit(graph_path)
        if output_path is None:
            output_path = graph_path.parent / "GRAPH_TREE.html"
        out = write_tree_html(
            graph_path=graph_path, output_path=output_path,
            root=root, max_children=max_children,
            top_k_edges=top_k_edges, project_label=project_label,
        )
        size_kb = out.stat().st_size / 1024
        print(f"wrote {out} ({size_kb:.1f} KB)")
        print(f"open with: xdg-open {out}  (or file://{out.resolve()})")
        sys.exit(0)

    elif cmd == "merge-driver":
        # git merge driver for graph.json — takes (base, current, other) and writes
        # the union of current+other nodes/edges back to current. Exits 1 on
        # corrupt input so git surfaces the conflict instead of silently
        # accepting a poisoned merge (see F-005).
        # Usage: graphify merge-driver %O %A %B  (set in .git/config merge driver)
        if len(sys.argv) < 5:
            print("Usage: graphify merge-driver <base> <current> <other>", file=sys.stderr)
            sys.exit(1)
        _base_path, _current_path, _other_path = sys.argv[2], sys.argv[3], sys.argv[4]
        # Hard caps so a malicious or corrupted graph.json cannot exhaust memory
        # at parse time. 50 MB / 100k nodes are well above any realistic graph
        # (typical graphs are <5 MB / <50k nodes); anything larger should fail
        # the merge so a human can investigate.
        _MERGE_MAX_BYTES = 50 * 1024 * 1024
        _MERGE_MAX_NODES = 100_000
        import networkx as _nx
        from networkx.readwrite import json_graph as _jg
        def _load_graph(p: str):
            path_obj = Path(p)
            try:
                size = path_obj.stat().st_size
            except OSError as exc:
                raise RuntimeError(f"cannot stat {p}: {exc}") from exc
            if size > _MERGE_MAX_BYTES:
                raise RuntimeError(
                    f"graph.json {p} is {size} bytes, exceeds {_MERGE_MAX_BYTES}-byte cap"
                )
            data = json.loads(path_obj.read_text(encoding="utf-8"))
            # A committed raw (--no-cluster) graph stores edges under "edges";
            # parse via the shared links/edges-normalizing loader (#2212).
            from graphify.paths import load_node_link_graph as _lnlg
            return _lnlg(data), data
        try:
            G_cur, _ = _load_graph(_current_path)
            G_oth, _ = _load_graph(_other_path)
        except Exception as exc:
            print(f"[graphify merge-driver] error loading graphs: {exc}", file=sys.stderr)
            sys.exit(1)  # surface the conflict so git doesn't accept a corrupt merge
        merged = _nx.compose(G_cur, G_oth)
        if merged.number_of_nodes() > _MERGE_MAX_NODES:
            print(
                f"[graphify merge-driver] merged graph has {merged.number_of_nodes()} nodes, "
                f"exceeds {_MERGE_MAX_NODES}-node cap; aborting merge.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            out_data = _jg.node_link_data(merged, edges="links")
        except TypeError:
            out_data = _jg.node_link_data(merged)
        from graphify.generation import (
            CodeUpdateRequest,
            Corpus,
            CorpusGraph,
            OperationFailed,
            PublicationRefused,
        )
        from graphify.generation._publication import _GraphData, _Publication

        _merge_target = Path(_current_path)
        _merge_output = _merge_target.parent
        _merge_root = (
            _merge_output.parent
            if _merge_output.name == Path(_GRAPHIFY_OUT).name
            else _merge_output
        )
        _merge_outcome = CorpusGraph(
            Corpus(root=_merge_root, output=_merge_output)
        ).code_update(
            CodeUpdateRequest(),
            _publication=_Publication(
                graph=_GraphData(out_data, force=True),
            ),
        )
        if isinstance(_merge_outcome, (PublicationRefused, OperationFailed)):
            print(
                f"[graphify merge-driver] publication failed: {_merge_outcome.reason}",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(0)

    elif cmd == "merge-graphs":
        # graphify merge-graphs graph1.json graph2.json ... --out merged.json
        args = sys.argv[2:]
        graph_paths: list[Path] = []
        out_path = Path(_GRAPHIFY_OUT) / "merged-graph.json"
        i = 0
        while i < len(args):
            if args[i] == "--out" and i + 1 < len(args):
                out_path = Path(args[i + 1])
                i += 2
            else:
                graph_paths.append(Path(args[i]))
                i += 1
        if len(graph_paths) < 2:
            print(
                "Usage: graphify merge-graphs <graph1.json> <graph2.json> [...] [--out merged.json]",
                file=sys.stderr,
            )
            sys.exit(1)
        import networkx as _nx
        from networkx.readwrite import json_graph as _jg
        from graphify.build import prefix_graph_for_global as _prefix, distinct_repo_tags as _repo_tags
        graphs = []
        for gp in graph_paths:
            if not gp.exists():
                print(f"error: not found: {gp}", file=sys.stderr)
                sys.exit(1)
            _enforce_graph_size_cap_or_exit(gp)
            data = json.loads(gp.read_text(encoding="utf-8"))
            # Normalize edges/links key before loading — graphify writes "links"
            # via node_link_data but older runs may have used "edges" (#738).
            if "links" not in data and "edges" in data:
                data = dict(data, links=data["edges"])
            try:
                G = _jg.node_link_graph(data, edges="links")
            except TypeError:
                G = _jg.node_link_graph(data)
            graphs.append(G)
        # nx.compose requires all graphs to be the same type.  When input graphs
        # come from different sources (e.g. an AST-only run vs a full LLM run) one
        # may be a MultiGraph and another a Graph.  Normalise everything to Graph
        # (the graphify default) by converting MultiGraphs with nx.Graph().
        def _to_simple(g: "_nx.Graph") -> "_nx.Graph":
            # nx.compose requires every graph to be the same type. Inputs may
            # disagree on BOTH axes — directed vs undirected, and multi vs simple
            # — because per-repo graph.json files are written by different extract
            # paths at different times. Normalise everything to a plain undirected
            # Graph (the merged cross-repo view is undirected anyway), which covers
            # DiGraph / MultiGraph / MultiDiGraph. Without this a directed input
            # crashed compose with "All graphs must be directed or undirected" (#1606).
            if type(g) is not _nx.Graph:
                return _nx.Graph(g)
            return g
        # Unique repo tag per graph. The bare `graphify-out/..` dir name is not
        # unique across inputs (src/graphify-out and frontend/src/graphify-out both
        # → "src"), which collides same-stem node ids and silently merges unrelated
        # entities (#1729). distinct_repo_tags guarantees a distinct prefix per graph.
        repo_tags = _repo_tags(graph_paths)
        naive_tags = [gp.parent.parent.name for gp in graph_paths]
        if len(set(naive_tags)) != len(naive_tags):
            print(f"  note: repo dir names collide; using distinct tags: {', '.join(repo_tags)}")
        merged = _nx.Graph()
        for G, repo_tag in zip(graphs, repo_tags):
            prefixed = _to_simple(_prefix(G, repo_tag))
            merged = _nx.compose(merged, prefixed)
        try:
            out_data = _jg.node_link_data(merged, edges="links")
        except TypeError:
            out_data = _jg.node_link_data(merged)
        if (
            out_path.name == "graph.json"
            and out_path.parent.name == Path(_GRAPHIFY_OUT).name
        ):
            # The monorepo runbook intentionally targets canonical graph.json;
            # enter CorpusGraph instead of letting this export path become a
            # competing generation writer.
            from graphify.generation import (
                CodeUpdateRequest,
                Corpus,
                CorpusGraph,
                OperationFailed,
                PublicationRefused,
            )
            from graphify.generation._publication import _GraphData, _Publication

            _merge_outcome = CorpusGraph(
                Corpus(root=out_path.parent.parent, output=out_path.parent)
            ).code_update(
                CodeUpdateRequest(),
                _publication=_Publication(
                    graph=_GraphData(out_data, force=True),
                ),
            )
            if isinstance(_merge_outcome, (PublicationRefused, OperationFailed)):
                print(f"error: {_merge_outcome.reason}", file=sys.stderr)
                sys.exit(1)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            from graphify.paths import write_json_atomic as _wja
            _wja(out_path, out_data, indent=2)
        print(f"Merged {len(graphs)} graphs -> {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
        print(f"Written to: {out_path}")

    elif cmd == "clone":
        if len(sys.argv) < 3:
            print(
                "Usage: graphify clone <github-url> [--branch <branch>] [--out <dir>]",
                file=sys.stderr,
            )
            sys.exit(1)
        url = sys.argv[2]
        branch: str | None = None
        out_dir: Path | None = None
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--branch" and i + 1 < len(args):
                branch = args[i + 1]
                i += 2
            elif args[i] == "--out" and i + 1 < len(args):
                out_dir = Path(args[i + 1])
                i += 2
            else:
                i += 1
        local_path = _clone_repo(url, branch=branch, out_dir=out_dir)
        print(local_path)

    elif cmd == "export":
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd not in ("html", "callflow-html", "obsidian", "wiki", "svg", "graphml", "neo4j", "falkordb"):
            print("Usage: graphify export <format>", file=sys.stderr)
            print("  html      [--graph PATH] [--labels PATH] [--node-limit N] [--no-viz]", file=sys.stderr)
            print("  callflow-html [GRAPH|DIR] [--graph PATH] [--labels PATH] [--report PATH] [--sections PATH] [--output HTML]", file=sys.stderr)
            print("            [--lang auto|zh-CN|en] [--max-sections N] [--diagram-scale N]", file=sys.stderr)
            print("  obsidian  [--graph PATH] [--labels PATH] [--dir PATH]", file=sys.stderr)
            print("  wiki      [--graph PATH] [--labels PATH]", file=sys.stderr)
            print("  svg       [--graph PATH] [--labels PATH]", file=sys.stderr)
            print("  graphml   [--graph PATH]", file=sys.stderr)
            print("  neo4j     [--graph PATH] [--push URI] [--user U] [--password P]", file=sys.stderr)
            print("            (or set NEO4J_PASSWORD instead of --password to keep it off argv)", file=sys.stderr)
            print("  falkordb  [--graph PATH] [--push URI] [--user U] [--password P]", file=sys.stderr)
            print("            (or set FALKORDB_PASSWORD instead of --password to keep it off argv)", file=sys.stderr)
            sys.exit(1)

        # Parse shared args
        args = sys.argv[3:]
        graph_path = Path(_GRAPHIFY_OUT) / "graph.json"
        graph_path_explicit = False
        labels_path = Path(_GRAPHIFY_OUT) / ".graphify_labels.json"
        labels_path_explicit = False
        report_path = Path(_GRAPHIFY_OUT) / "GRAPH_REPORT.md"
        report_path_explicit = False
        sections_path: Path | None = None
        callflow_output: Path | None = None
        callflow_lang = "auto"
        callflow_max_sections = 15
        callflow_diagram_scale = 1.0
        callflow_max_diagram_nodes = 18
        callflow_max_diagram_edges = 24
        analysis_path = Path(_GRAPHIFY_OUT) / ".graphify_analysis.json"
        node_limit = 5000
        no_viz = False
        obsidian_dir = Path(_GRAPHIFY_OUT) / "obsidian"
        # Shared push-connection settings for the graph-database sinks (neo4j,
        # falkordb), parsed from the generic --push/--user/--password flags below.
        push_uri: str | None = None
        push_user = "neo4j"  # Neo4j default user; FalkorDB auth is optional and ignores it
        # F-031: prefer an env var so the password never appears on argv (visible
        # in `ps` output / shell history). The explicit --password flag still
        # overrides it. Each sink reads its own var: FALKORDB_PASSWORD for falkordb,
        # NEO4J_PASSWORD otherwise.
        push_password: str | None = (
            os.environ.get("FALKORDB_PASSWORD") if subcmd == "falkordb"
            else os.environ.get("NEO4J_PASSWORD")
        ) or None
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--graph" and i + 1 < len(args):
                graph_path = Path(args[i + 1])
                graph_path_explicit = True
                i += 2
            elif a == "--labels" and i + 1 < len(args):
                labels_path = Path(args[i + 1])
                labels_path_explicit = True
                i += 2
            elif a == "--report" and i + 1 < len(args):
                report_path = Path(args[i + 1])
                report_path_explicit = True
                i += 2
            elif a == "--sections" and i + 1 < len(args):
                sections_path = Path(args[i + 1]); i += 2
            elif a == "--output" and i + 1 < len(args):
                callflow_output = Path(args[i + 1]).expanduser()
                if not callflow_output.is_absolute():
                    callflow_output = Path.cwd() / callflow_output
                i += 2
            elif a == "--lang" and i + 1 < len(args):
                callflow_lang = args[i + 1]; i += 2
            elif a == "--max-sections" and i + 1 < len(args):
                callflow_max_sections = int(args[i + 1]); i += 2
            elif a == "--diagram-scale" and i + 1 < len(args):
                callflow_diagram_scale = float(args[i + 1]); i += 2
            elif a == "--max-diagram-nodes" and i + 1 < len(args):
                callflow_max_diagram_nodes = int(args[i + 1]); i += 2
            elif a == "--max-diagram-edges" and i + 1 < len(args):
                callflow_max_diagram_edges = int(args[i + 1]); i += 2
            elif a in ("-h", "--help") and subcmd == "callflow-html":
                print("Usage: graphify export callflow-html [GRAPH|DIR] [--graph PATH] [--labels PATH]")
                print("  --report PATH          path to GRAPH_REPORT.md")
                print("  --sections PATH        JSON section definitions")
                print("  --output HTML          output path (default graphify-out/<project>-callflow.html)")
                print("  --lang LANG            auto, zh-CN, en, etc. (default auto)")
                print("  --max-sections N       maximum auto-derived sections (default 15)")
                print("  --diagram-scale N      Mermaid diagram scale (default 1.0)")
                print("  --max-diagram-nodes N  representative nodes per section (default 18)")
                print("  --max-diagram-edges N  representative edges per section (default 24)")
                sys.exit(0)
            elif a == "--node-limit" and i + 1 < len(args):
                node_limit = int(args[i + 1]); i += 2
            elif a == "--no-viz":
                no_viz = True; i += 1
            elif a == "--dir" and i + 1 < len(args):
                obsidian_dir = Path(args[i + 1]); i += 2
            elif a == "--push" and i + 1 < len(args):
                push_uri = args[i + 1]; i += 2
            elif a == "--user" and i + 1 < len(args):
                push_user = args[i + 1]; i += 2
            elif a == "--password" and i + 1 < len(args):
                push_password = args[i + 1]; i += 2
            elif subcmd == "callflow-html" and not a.startswith("-") and not graph_path_explicit:
                candidate = Path(a)
                if candidate.name == "graph.json" or candidate.suffix.lower() == ".json":
                    graph_path = candidate
                elif (candidate / "graph.json").exists():
                    graph_path = candidate / "graph.json"
                else:
                    graph_path = candidate / _GRAPHIFY_OUT / "graph.json"
                graph_path_explicit = True
                i += 1
            else:
                i += 1

        graph_path = graph_path.expanduser()
        if graph_path_explicit:
            graph_out_dir = graph_path.parent
            if not labels_path_explicit:
                labels_path = graph_out_dir / ".graphify_labels.json"
            if not report_path_explicit:
                report_path = graph_out_dir / "GRAPH_REPORT.md"
        labels_path = labels_path.expanduser()
        report_path = report_path.expanduser()

        if not graph_path.exists():
            print(f"error: graph not found: {graph_path}. Run /graphify <path> first.", file=sys.stderr)
            sys.exit(1)

        if subcmd == "callflow-html":
            from graphify.callflow_html import write_callflow_html as _write_callflow_html
            out = _write_callflow_html(
                graph=graph_path,
                report=report_path,
                labels=labels_path,
                sections=sections_path,
                output=callflow_output,
                lang=callflow_lang,
                max_sections=callflow_max_sections,
                diagram_scale=callflow_diagram_scale,
                max_diagram_nodes=callflow_max_diagram_nodes,
                max_diagram_edges=callflow_max_diagram_edges,
                verbose=True,
            )
            print(f"callflow HTML written - open in any browser: {out}")
            sys.exit(0)

        from networkx.readwrite import json_graph as _jg
        from graphify.build import build_from_json as _bfj
        from graphify.security import check_graph_file_size_cap as _check_cap

        # Solution 3 (#1019): for the HTML view, an oversized graph.json should
        # not be a hard error. Detect the over-cap condition here and fall back
        # to the community-aggregation view (node_limit=5000) below instead of
        # exiting 1. All other subcommands keep the hard cap.
        _over_cap = False
        try:
            _check_cap(graph_path)
        except ValueError as _cap_err:
            if subcmd == "html":
                _over_cap = True
                try:
                    _over_cap_bytes = graph_path.stat().st_size
                except OSError:
                    _over_cap_bytes = -1
                print(
                    f"warning: graph.json exceeds cap ({_over_cap_bytes} bytes); "
                    f"falling back to community-aggregation view (node_limit=5000)",
                    file=sys.stderr,
                )
            else:
                print(f"error: {_cap_err}", file=sys.stderr)
                sys.exit(1)
        _raw = json.loads(graph_path.read_text(encoding="utf-8"))
        if "links" not in _raw and "edges" in _raw:
            _raw = dict(_raw, links=_raw["edges"])
        try:
            G = _jg.node_link_graph(_raw, edges="links")
        except TypeError:
            G = _jg.node_link_graph(_raw)

        # Load optional analysis/labels
        communities: dict[int, list[str]] = {}
        if analysis_path.exists():
            _an = json.loads(analysis_path.read_text(encoding="utf-8"))
            communities = {int(k): v for k, v in _an.get("communities", {}).items()}
            cohesion: dict[int, float] = {int(k): v for k, v in _an.get("cohesion", {}).items()}
            gods_data = _an.get("gods", [])
        else:
            cohesion = {}
            gods_data = []

        # Fallback: graph.json carries the per-node community as a node attribute
        # (`to_json` writes it on every node). The analysis sidecar is the
        # canonical source — but the post-commit / watch rebuild path doesn't
        # regenerate it, and `extract` may have its temp files cleaned up. When
        # that happens, `graphify export html` previously bailed with
        # "Single community - aggregated view not useful." even though the
        # per-node attribute had the right data all along. Reconstruct from
        # the graph itself so downstream subcommands (html, obsidian, wiki,
        # svg, graphml, neo4j) don't silently produce a degraded artifact.
        if not communities:
            reconstructed: dict[int, list[str]] = {}
            for node_id, data in G.nodes(data=True):
                cid_raw = data.get("community")
                if cid_raw is None:
                    continue
                try:
                    cid = int(cid_raw)
                except (TypeError, ValueError):
                    continue
                reconstructed.setdefault(cid, []).append(str(node_id))
            if reconstructed:
                communities = reconstructed

        labels: dict[int, str] = {}
        if labels_path.exists():
            labels = {int(k): v for k, v in json.loads(labels_path.read_text(encoding="utf-8")).items()}

        out_dir = graph_path.parent

        if subcmd == "html":
            from graphify.export import to_html as _to_html
            if no_viz:
                html_target = out_dir / "graph.html"
                if html_target.exists():
                    html_target.unlink()
                print("--no-viz: skipped graph.html")
            else:
                # Over-cap fallback (#1019): force the community-aggregation
                # path so the oversized graph still renders a usable artifact.
                _effective_node_limit = 5000 if _over_cap else node_limit
                _to_html(G, communities, str(out_dir / "graph.html"),
                         community_labels=labels or None, node_limit=_effective_node_limit)
                if G.number_of_nodes() <= _effective_node_limit:
                    print(f"graph.html written - open in any browser, no server needed")
                if _over_cap:
                    sys.exit(0)

        elif subcmd == "obsidian":
            from graphify.export import to_obsidian as _to_obsidian, to_canvas as _to_canvas
            n = _to_obsidian(G, communities, str(obsidian_dir),
                             community_labels=labels or None, cohesion=cohesion or None)
            print(f"Obsidian vault: {n} notes in {obsidian_dir}/")
            _to_canvas(G, communities, str(obsidian_dir / "graph.canvas"),
                       community_labels=labels or None)
            print(f"Canvas: {obsidian_dir}/graph.canvas")
            print(f"Open {obsidian_dir}/ as a vault in Obsidian.")

        elif subcmd == "wiki":
            from graphify.wiki import to_wiki as _to_wiki
            from graphify.analyze import god_nodes as _god_nodes
            if not communities:
                print(
                    "error: .graphify_analysis.json is missing or empty — refusing to export wiki to prevent data loss.\n"
                    "Run `graphify extract .` (or `graphify cluster-only .`) to regenerate community data first.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not gods_data:
                gods_data = _god_nodes(G)
            n = _to_wiki(G, communities, str(out_dir / "wiki"),
                         community_labels=labels or None, cohesion=cohesion or None,
                         god_nodes_data=gods_data)
            print(f"Wiki: {n} articles written to {out_dir}/wiki/")
            print(f"  {out_dir}/wiki/index.md  ->  agent entry point")

        elif subcmd == "svg":
            from graphify.export import to_svg as _to_svg
            _to_svg(G, communities, str(out_dir / "graph.svg"),
                    community_labels=labels or None)
            print(f"graph.svg written - embeds in Obsidian, Notion, GitHub READMEs")

        elif subcmd == "graphml":
            from graphify.export import to_graphml as _to_graphml
            _to_graphml(G, communities, str(out_dir / "graph.graphml"))
            print(f"graph.graphml written - open in Gephi, yEd, or any GraphML tool")

        elif subcmd == "neo4j":
            if push_uri:
                from graphify.export import push_to_neo4j as _push
                if push_password is None:
                    print("error: --password required for --push", file=sys.stderr)
                    sys.exit(1)
                result = _push(G, uri=push_uri, user=push_user,
                               password=push_password, communities=communities)
                print(f"Pushed to Neo4j: {result['nodes']} nodes, {result['edges']} edges")
            else:
                from graphify.export import to_cypher as _to_cypher
                _to_cypher(G, str(out_dir / "cypher.txt"))
                print(f"cypher.txt written - import with: cypher-shell < {out_dir}/cypher.txt")

        elif subcmd == "falkordb":
            if push_uri:
                from graphify.export import push_to_falkordb as _push
                result = _push(G, uri=push_uri, user=push_user,
                               password=push_password, communities=communities)
                print(f"Pushed to FalkorDB: {result['nodes']} nodes, {result['edges']} edges")
            else:
                from graphify.export import to_cypher as _to_cypher
                _to_cypher(G, str(out_dir / "cypher.txt"))
                print(f"cypher.txt written ({out_dir}/cypher.txt) - statements are OpenCypher. "
                      f"FalkorDB's GRAPH.QUERY runs one statement at a time (no bulk script "
                      f"import), so load a graph with: graphify export falkordb --push "
                      f"falkordb://localhost:6379")

    elif cmd == "benchmark":
        from graphify.benchmark import run_benchmark, print_benchmark

        graph_path = sys.argv[2] if len(sys.argv) > 2 else _default_graph_path()
        _enforce_graph_size_cap_or_exit(Path(graph_path))
        # Try to load corpus_words from detect output
        corpus_words = None
        detect_path = Path(".graphify_detect.json")
        if detect_path.exists():
            try:
                detect_data = json.loads(detect_path.read_text(encoding="utf-8"))
                corpus_words = detect_data.get("total_words")
            except Exception:
                pass
        result = run_benchmark(graph_path, corpus_words=corpus_words)
        print_benchmark(result)

    elif cmd == "global":
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        from graphify.global_graph import (
            global_add as _global_add,
            global_remove as _global_remove,
            global_list as _global_list,
            global_path as _global_path,
        )
        if subcmd == "add":
            # graphify global add <graph.json> [--as <tag>]
            args = sys.argv[3:]
            source = None
            tag = None
            i = 0
            while i < len(args):
                if args[i] == "--as" and i + 1 < len(args):
                    tag = args[i + 1]; i += 2
                elif not source:
                    source = Path(args[i]); i += 1
                else:
                    i += 1
            if not source:
                print("Usage: graphify global add <graph.json> [--as <repo-tag>]", file=sys.stderr)
                sys.exit(1)
            tag = tag or source.parent.parent.name
            try:
                result = _global_add(source, tag)
                if result["skipped"]:
                    print(f"'{tag}' unchanged since last add - global graph not modified.")
                else:
                    print(f"Added '{tag}' to global graph: +{result['nodes_added']} nodes, "
                          f"-{result['nodes_removed']} pruned. Global: {_global_path()}")
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr); sys.exit(1)
        elif subcmd == "remove":
            tag = sys.argv[3] if len(sys.argv) > 3 else ""
            if not tag:
                print("Usage: graphify global remove <repo-tag>", file=sys.stderr); sys.exit(1)
            try:
                removed = _global_remove(tag)
                print(f"Removed '{tag}' from global graph ({removed} nodes pruned).")
            except KeyError as exc:
                print(f"error: {exc}", file=sys.stderr); sys.exit(1)
        elif subcmd == "list":
            repos = _global_list()
            if not repos:
                print("Global graph is empty. Use 'graphify global add' to add a project.")
            else:
                print(f"Global graph: {_global_path()}")
                for tag, info in repos.items():
                    print(f"  {tag}: {info.get('node_count', '?')} nodes, added {info.get('added_at', '?')[:10]}")
        elif subcmd == "path":
            print(_global_path())
        else:
            print("Usage: graphify global [add|remove|list|path]", file=sys.stderr); sys.exit(1)

    elif cmd == "extract":
        # Full extraction: the interpreting half of the Graph-generation
        # lifecycle, and the only operation that costs LLM tokens. Everything in
        # this branch translates flags into one ``FullExtractionRequest``,
        # renders what the operation reports, and picks an exit code. Discovery,
        # AST extraction, interpretation, reconciliation, the safety rules, and
        # every canonical write belong to ``graphify.generation``.
        if len(sys.argv) < 3:
            print(
                "Usage: graphify extract <path> [--backend gemini|kimi|claude|openai|deepseek|ollama] "
                "[--model M] [--mode deep] [--out DIR|--output DIR] [--google-workspace] [--no-cluster] "
                "[--no-gitignore] [--code-only] "
                "[--max-workers N] [--token-budget N] [--max-concurrency N] "
                "[--api-timeout S] [--postgres DSN] [--cargo] [--allow-partial] [--timing]",
                file=sys.stderr,
            )
            sys.exit(1)

        has_path = True
        if sys.argv[2].startswith("-"):
            has_path = False
            target = Path(".").resolve()
        else:
            target = Path(sys.argv[2]).resolve()
            if not target.exists():
                print(f"error: path not found: {target}", file=sys.stderr)
                sys.exit(1)

        backend: str | None = None
        model: str | None = None
        extract_mode: str | None = None
        out_dir: Path | None = None
        cli_postgres_dsn: str | None = None
        cli_cargo: bool = False
        cli_allow_partial: bool = False
        no_cluster = False
        dedup_llm = False
        google_workspace = False
        global_merge = False
        code_only = False
        no_gitignore = False
        global_repo_tag: str | None = None
        # Performance/tuning knobs (issue #792). None means "use library default".
        cli_max_workers: int | None = None
        cli_token_budget: int | None = None
        cli_max_concurrency: int | None = None
        cli_api_timeout: float | None = None
        # Clustering tuning knobs
        cli_resolution: float = 1.0
        cli_exclude_hubs: float | None = None
        cli_excludes: list[str] = []
        cli_timing: bool = False
        # --force parity with `graphify update`: the flag or GRAPHIFY_FORCE=1
        # re-interprets every semantic source instead of serving the provider's
        # cache, and authorizes replacing the active graph with a smaller one
        # (#1894). It never weakens a fail-closed publication rule; that is
        # --allow-partial's separate authority.
        force = os.environ.get("GRAPHIFY_FORCE", "").lower() in ("1", "true", "yes")

        def _parse_int(name: str, raw: str) -> int:
            try:
                v = int(raw)
            except ValueError:
                print(f"error: {name} must be a positive integer (got {raw!r})", file=sys.stderr)
                sys.exit(2)
            if v <= 0:
                print(f"error: {name} must be > 0 (got {v})", file=sys.stderr)
                sys.exit(2)
            return v

        def _parse_float(name: str, raw: str) -> float:
            try:
                v = float(raw)
            except ValueError:
                print(f"error: {name} must be a positive number (got {raw!r})", file=sys.stderr)
                sys.exit(2)
            if v <= 0:
                print(f"error: {name} must be > 0 (got {v})", file=sys.stderr)
                sys.exit(2)
            return v

        args = sys.argv[3:] if has_path else sys.argv[2:]
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--backend" and i + 1 < len(args):
                backend = args[i + 1]; i += 2
            elif a.startswith("--backend="):
                backend = a.split("=", 1)[1]; i += 1
            elif a == "--model" and i + 1 < len(args):
                model = args[i + 1]; i += 2
            elif a.startswith("--model="):
                model = a.split("=", 1)[1]; i += 1
            elif a == "--mode" and i + 1 < len(args):
                extract_mode = args[i + 1]; i += 2
            elif a.startswith("--mode="):
                extract_mode = a.split("=", 1)[1]; i += 1
            elif a in ("--out", "--output") and i + 1 < len(args):
                # --output is an alias of --out (#2004): it was silently dropped
                # before, and `graphify tree` already documents --output, so the
                # mistake is natural. (--output= does not startswith --out=.)
                out_dir = Path(args[i + 1]); i += 2
            elif a.startswith(("--out=", "--output=")):
                out_dir = Path(a.split("=", 1)[1]); i += 1
            elif a == "--no-cluster":
                no_cluster = True; i += 1
            elif a == "--dedup-llm":
                dedup_llm = True; i += 1
            elif a == "--code-only":
                code_only = True; i += 1
            elif a == "--google-workspace":
                google_workspace = True; i += 1
            elif a == "--no-gitignore":
                no_gitignore = True; i += 1
            elif a == "--global":
                global_merge = True; i += 1
            elif a == "--as" and i + 1 < len(args):
                global_repo_tag = args[i + 1]; i += 2
            elif a == "--max-workers" and i + 1 < len(args):
                cli_max_workers = _parse_int("--max-workers", args[i + 1]); i += 2
            elif a.startswith("--max-workers="):
                cli_max_workers = _parse_int("--max-workers", a.split("=", 1)[1]); i += 1
            elif a == "--token-budget" and i + 1 < len(args):
                cli_token_budget = _parse_int("--token-budget", args[i + 1]); i += 2
            elif a.startswith("--token-budget="):
                cli_token_budget = _parse_int("--token-budget", a.split("=", 1)[1]); i += 1
            elif a == "--max-concurrency" and i + 1 < len(args):
                cli_max_concurrency = _parse_int("--max-concurrency", args[i + 1]); i += 2
            elif a.startswith("--max-concurrency="):
                cli_max_concurrency = _parse_int("--max-concurrency", a.split("=", 1)[1]); i += 1
            elif a == "--api-timeout" and i + 1 < len(args):
                cli_api_timeout = _parse_float("--api-timeout", args[i + 1]); i += 2
            elif a.startswith("--api-timeout="):
                cli_api_timeout = _parse_float("--api-timeout", a.split("=", 1)[1]); i += 1
            elif a == "--resolution" and i + 1 < len(args):
                cli_resolution = _parse_float("--resolution", args[i + 1]); i += 2
            elif a.startswith("--resolution="):
                cli_resolution = _parse_float("--resolution", a.split("=", 1)[1]); i += 1
            elif a == "--exclude-hubs" and i + 1 < len(args):
                cli_exclude_hubs = float(args[i + 1]); i += 2
            elif a.startswith("--exclude-hubs="):
                cli_exclude_hubs = float(a.split("=", 1)[1]); i += 1
            elif a == "--exclude" and i + 1 < len(args):
                cli_excludes.append(args[i + 1]); i += 2
            elif a.startswith("--exclude="):
                cli_excludes.append(a.split("=", 1)[1]); i += 1
            elif a == "--postgres" and i + 1 < len(args):
                cli_postgres_dsn = args[i + 1]; i += 2
            elif a.startswith("--postgres="):
                cli_postgres_dsn = a.split("=", 1)[1]; i += 1
            elif a == "--cargo":
                cli_cargo = True
                i += 1
            elif a == "--force":
                force = True; i += 1
            elif a == "--allow-partial":
                cli_allow_partial = True; i += 1
            elif a == "--timing":
                cli_timing = True; i += 1
            else:
                i += 1

        if not has_path and cli_postgres_dsn is None:
            print("error: must specify a path to scan or a --postgres DSN", file=sys.stderr)
            sys.exit(1)

        _VALID_MODES = {"deep"}
        if extract_mode is not None and extract_mode not in _VALID_MODES:
            print(
                f"error: unknown --mode '{extract_mode}'. "
                f"Available: {', '.join(sorted(_VALID_MODES))}",
                file=sys.stderr,
            )
            sys.exit(2)
        deep_mode = extract_mode == "deep"
        if deep_mode:
            print("[graphify extract] deep mode enabled: richer semantic extraction")

        # Argument validation, so a typo is refused before the request is made
        # rather than surfacing as a failed evidence source much later.
        if backend is not None:
            from graphify.llm import BACKENDS as _BACKENDS

            if backend not in _BACKENDS:
                print(
                    f"error: unknown backend '{backend}'. "
                    f"Available: {', '.join(sorted(_BACKENDS))}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if dedup_llm:
            # Accepted so an existing invocation still runs, but it no longer
            # has anything to act on: a Graph generation's evidence is the
            # per-source contributions it was reconciled from, and fuzzy entity
            # merging across sources is not part of that authority.
            print(
                "[graphify extract] note: --dedup-llm no longer applies — a Graph "
                "generation is materialized from its Source contributions.",
                file=sys.stderr,
            )

        # CLI flag wins over env var. Setting GRAPHIFY_API_TIMEOUT here so
        # _call_openai_compat picks it up without needing a new kwarg path.
        if cli_api_timeout is not None:
            os.environ["GRAPHIFY_API_TIMEOUT"] = str(cli_api_timeout)
        if cli_max_workers is not None:
            os.environ["GRAPHIFY_MAX_WORKERS"] = str(cli_max_workers)

        # Resolve output dir. The user-facing contract is "<out>/graphify-out/"
        # so a fresh checkout writes graphify-out/ at the project root, matching
        # the skill.md pipeline.
        out_root = (out_dir.resolve() if out_dir else target)
        graphify_out = out_root / _GRAPHIFY_OUT
        graphify_out.mkdir(parents=True, exist_ok=True)

        stages = _StageTimer(cli_timing)

        from graphify.generation import (
            BuildPolicy as _BuildPolicy,
            CargoSource as _CargoSource,
            Corpus as _Corpus,
            CorpusGraph as _CorpusGraph,
            FullExtractionRequest as _FullExtractionRequest,
            GoogleWorkspaceSource as _GoogleWorkspaceSource,
            PostgresSource as _PostgresSource,
            ReplaceBuildPolicy as _ReplaceBuildPolicy,
            SemanticSource as _SemanticSource,
            WaitUntilCovered as _WaitUntilCovered,
        )

        _corpus_graph = _CorpusGraph(_Corpus(root=target, output=graphify_out))

        # Corpus-shaping policy. Replacement is whole by design, so a flag that
        # changes one half composes with the recorded policy for the other:
        # --no-gitignore alone must not drop the persisted --exclude patterns
        # (#1886), and a flag-less run must not clobber a persisted
        # --no-gitignore back to True (#1971). No flag at all sends no policy
        # request, which preserves whatever the active generation was built
        # under.
        _policy_request = None
        if cli_excludes or no_gitignore:
            try:
                _active_policy = _corpus_graph.active_build_policy()
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            _policy_request = _ReplaceBuildPolicy(
                _BuildPolicy(
                    excludes=tuple(cli_excludes) or _active_policy.excludes,
                    gitignore=False if no_gitignore else _active_policy.gitignore,
                )
            )

        # --code-only requests no Semantic provider at all, which is what makes
        # it free: with no provider the operation never asks for an API key, and
        # documents are represented by whatever structure can be derived locally.
        _provider = None
        if not code_only:
            from graphify.semantic_provider import LlmSemanticProvider

            _provider = LlmSemanticProvider(
                backend=backend,
                model=model,
                deep=deep_mode,
                refresh=force,
                token_budget=cli_token_budget,
                max_concurrency=cli_max_concurrency,
                on_progress=lambda done, total: print(
                    f"{_EXTRACT_PREFIX}chunk {done}/{total} done", flush=True
                ),
            )

        _sources: list = []
        if _provider is not None:
            _sources.append(_SemanticSource(provider=_provider))
        if cli_postgres_dsn is not None:
            _sources.append(_PostgresSource(dsn=cli_postgres_dsn))
        if cli_cargo:
            _sources.append(_CargoSource())
        if google_workspace:
            _sources.append(_GoogleWorkspaceSource())

        if force:
            print(
                "[graphify extract] --force: re-interpreting every semantic "
                "source; a smaller graph is authorized"
            )
        print(f"[graphify extract] scanning {target}")

        # Completion is stated rather than defaulted, and there is deliberately
        # no flag for the other half of the union: this request carries live
        # in-process evidence sources, so no other executor could ever carry out
        # a durable record of it. Waiting until the request is covered is the
        # only honest shape a Full extraction has.
        _outcome = _corpus_graph.full_extraction(
            _FullExtractionRequest(
                sources=tuple(_sources),
                build_policy=_policy_request,
                force=force,
                allow_partial_publication=cli_allow_partial,
            ),
            completion=_WaitUntilCovered(),
            observer=_ExtractionProgress(stages, code_only=code_only),
        )
        if not _render_full_extraction(_outcome):
            stages.total()
            sys.exit(1)
        stages.mark("publish")

        graph_json_path = graphify_out / "graph.json"
        analysis_path = graphify_out / ".graphify_analysis.json"

        if no_cluster:
            _nodes, _edges = _published_graph_size(graph_json_path)
            print(
                f"[graphify extract] wrote {graph_json_path} — "
                f"{_nodes} nodes, {_edges} edges (no clustering)"
            )
        elif not _publish_clustered_generation(
            _corpus_graph,
            graph_json_path,
            resolution=cli_resolution,
            exclude_hubs=cli_exclude_hubs,
            stages=stages,
        ):
            stages.total()
            sys.exit(1)
        else:
            print(f"[graphify extract] wrote {analysis_path}")

        # After the local generation is final, never before: an unavailable
        # global graph must not be able to affect what this Corpus published.
        if global_merge:
            from graphify.global_graph import global_add as _global_add
            _tag = global_repo_tag or target.name
            try:
                result = _global_add(graph_json_path, _tag)
                if result["skipped"]:
                    print(f"[graphify global] '{_tag}' unchanged since last add - skipped.")
                else:
                    print(f"[graphify global] '{_tag}' merged into global graph "
                          f"(+{result['nodes_added']} nodes, -{result['nodes_removed']} pruned).")
            except Exception as exc:
                print(f"[graphify global] warning: failed to merge into global graph: {exc}", file=sys.stderr)

        if _provider is not None:
            if _provider.cache_hits or _provider.cache_misses:
                print(
                    f"[graphify extract] semantic cache: {_provider.cache_hits} "
                    f"cached, {_provider.cache_misses} re-extracted"
                )
            if _provider.input_tokens or _provider.output_tokens:
                from graphify.llm import estimate_cost as _estimate_cost

                cost = _estimate_cost(
                    _provider.backend or "",
                    _provider.input_tokens,
                    _provider.output_tokens,
                )
                print(
                    f"[graphify extract] tokens: {_provider.input_tokens:,} in / "
                    f"{_provider.output_tokens:,} out, est. cost "
                    f"(~{_provider.backend}): ${cost:.4f}"
                )
        # extract intentionally stops at graph.json + analysis; the report and
        # community labels are produced by `cluster-only` (or an agent's Step 5).
        # Point standalone users at it so communities get named (#1097).
        print(
            "[graphify extract] next: run "
            f"`graphify cluster-only {graphify_out.parent}` "
            "to generate GRAPH_REPORT.md and name communities"
        )
        stages.total()
        # The success half of the exit contract scripts read, kept explicit so
        # both the clustered and --no-cluster shapes end the same way.
        sys.exit(0)

    elif cmd == "cache-check":
        # graphify cache-check <files_from> [--root <dir>] [--mode <m> | --deep]
        #                       [--prompt-file <path>]
        # Reads file paths (one per line) from <files_from>, checks semantic cache.
        # --mode deep (or --deep) checks the cache/semantic-deep/ namespace
        # written by `extract --mode deep` instead of cache/semantic/ (#1894).
        # --prompt-file names the extraction prompt the caller will use (an agent's
        # references/extraction-spec.md), restricting hits to entries produced by
        # that same prompt (#1939). Omitting it reads the unattributed layout, which
        # cannot see entries a fingerprinted run wrote.
        # Writes:
        #   graphify-out/.graphify_cached.json   — already-cached nodes/edges/hyperedges
        #   graphify-out/.graphify_uncached.txt  — paths that need extraction
        # Stdout: "Cache: N hit, M miss"
        from graphify.cache import check_semantic_cache
        if len(sys.argv) < 3:
            print("Usage: graphify cache-check <files_from> [--root <dir>] "
                  "[--mode <m> | --deep] [--prompt-file <path>]", file=sys.stderr)
            sys.exit(1)
        files_from = Path(sys.argv[2])
        root = Path(".")
        cache_mode: str | None = None
        prompt_file: str | None = None
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--root" and i + 1 < len(sys.argv):
                root = Path(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
                cache_mode = sys.argv[i + 1]
                i += 2
            elif sys.argv[i].startswith("--mode="):
                cache_mode = sys.argv[i].split("=", 1)[1]
                i += 1
            elif sys.argv[i] == "--deep":
                cache_mode = "deep"
                i += 1
            elif sys.argv[i] == "--prompt-file" and i + 1 < len(sys.argv):
                prompt_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i].startswith("--prompt-file="):
                prompt_file = sys.argv[i].split("=", 1)[1]
                i += 1
            else:
                i += 1
        files = [f for f in files_from.read_text(encoding="utf-8").splitlines() if f.strip()]
        cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(
            files, root, mode=cache_mode, prompt_file=prompt_file
        )
        out = root / _GRAPHIFY_OUT
        out.mkdir(parents=True, exist_ok=True)
        if cached_nodes or cached_edges or cached_hyperedges:
            (out / ".graphify_cached.json").write_text(
                json.dumps({"nodes": cached_nodes, "edges": cached_edges, "hyperedges": cached_hyperedges},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        (out / ".graphify_uncached.txt").write_text("\n".join(uncached), encoding="utf-8")
        print(f"Cache: {len(files) - len(uncached)} hit, {len(uncached)} miss")

    elif cmd == "merge-chunks":
        # graphify merge-chunks <chunk_glob_or_files...> --out <path>
        # Concatenates .graphify_chunk_*.json files written by semantic subagents.
        # Deduplicates nodes by id (first writer wins). Sums token counts.
        import glob as _glob
        if len(sys.argv) < 3:
            print("Usage: graphify merge-chunks <chunk_files...> --out <path>", file=sys.stderr)
            sys.exit(1)
        out_path: Path | None = None
        chunk_args: list[str] = []
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--out" and i + 1 < len(sys.argv):
                out_path = Path(sys.argv[i + 1])
                i += 2
            else:
                chunk_args.append(sys.argv[i])
                i += 1
        if not out_path:
            print("error: --out <path> required", file=sys.stderr)
            sys.exit(1)
        chunk_files: list[str] = []
        for arg in chunk_args:
            expanded = _glob.glob(arg)
            chunk_files.extend(sorted(expanded) if expanded else [arg])
        merged: dict = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
        seen_ids: set[str] = set()
        valid_chunks = 0
        # These chunk files are untrusted subagent output. load_validated_...
        # stats the file size BEFORE reading it (so a multi-GB chunk can't blow up
        # memory), parses the JSON, and validates the security caps + the node/
        # edge id charset that blocks path traversal (#825) — the same enforcement
        # the skill merge path applies. A bad chunk is skipped with a warning
        # while valid siblings still merge; if every chunk is invalid, fail
        # closed instead of reporting success and replacing --out with an empty
        # semantic layer. Deliberately NOT wired into
        # build_from_json/load_graph_json, which must keep loading valid
        # pre-existing graphs. file_type is left to build's coercion (#840).
        from graphify.semantic_cleanup import load_validated_semantic_fragment
        for cf in chunk_files:
            chunk, _chunk_errs = load_validated_semantic_fragment(Path(cf))
            if _chunk_errs:
                print(
                    f"[graphify merge-chunks] warning: skipping invalid chunk {cf}: "
                    f"{'; '.join(_chunk_errs[:3])}",
                    file=sys.stderr,
                )
                continue
            valid_chunks += 1
            for n in chunk.get("nodes", []):
                if n.get("id") not in seen_ids:
                    seen_ids.add(n["id"])
                    merged["nodes"].append(n)
            merged["edges"].extend(chunk.get("edges", []))
            merged["hyperedges"].extend(chunk.get("hyperedges", []))
            # Coerce token counts: a chunk is untrusted, so a non-numeric
            # input_tokens/output_tokens must not abort the whole merge with a
            # TypeError after other chunks already merged.
            for _tok in ("input_tokens", "output_tokens"):
                _v = chunk.get(_tok, 0)
                merged[_tok] += _v if isinstance(_v, (int, float)) else 0
        if not valid_chunks:
            print(
                f"[graphify merge-chunks] error: no valid chunks to merge; "
                f"refusing to write {out_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from graphify.paths import write_json_atomic as _wja
        _wja(out_path, merged, ensure_ascii=False)
        chunk_summary = (
            f"{valid_chunks} chunks"
            if valid_chunks == len(chunk_files)
            else f"{valid_chunks} of {len(chunk_files)} chunks"
        )
        print(
            f"Merged {chunk_summary}: {len(merged['nodes'])} nodes, {len(merged['edges'])} edges, "
            f"{merged['input_tokens']:,} in / {merged['output_tokens']:,} out tokens"
        )

    elif cmd == "merge-semantic":
        # graphify merge-semantic --cached <path> --new <path> --out <path>
        # Merges cached semantic results with freshly-extracted chunk results.
        # Deduplicates nodes by id (cached entries take priority over new ones).
        if len(sys.argv) < 3:
            print("Usage: graphify merge-semantic --cached <path> --new <path> --out <path>", file=sys.stderr)
            sys.exit(1)
        cached_path: Path | None = None
        new_path: Path | None = None
        out_path2: Path | None = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--cached" and i + 1 < len(sys.argv):
                cached_path = Path(sys.argv[i + 1]); i += 2
            elif sys.argv[i] == "--new" and i + 1 < len(sys.argv):
                new_path = Path(sys.argv[i + 1]); i += 2
            elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
                out_path2 = Path(sys.argv[i + 1]); i += 2
            else:
                i += 1
        if not out_path2:
            print("error: --out <path> required", file=sys.stderr)
            sys.exit(1)
        empty: dict = {"nodes": [], "edges": [], "hyperedges": []}
        cached_data = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path and cached_path.exists() else empty
        new_data = json.loads(new_path.read_text(encoding="utf-8")) if new_path and new_path.exists() else empty
        seen_ids2: set[str] = set()
        all_nodes: list[dict] = []
        for n in cached_data.get("nodes", []) + new_data.get("nodes", []):
            if n.get("id") not in seen_ids2:
                seen_ids2.add(n["id"])
                all_nodes.append(n)
        merged2 = {
            "nodes": all_nodes,
            "edges": cached_data.get("edges", []) + new_data.get("edges", []),
            "hyperedges": cached_data.get("hyperedges", []) + new_data.get("hyperedges", []),
        }
        out_path2.parent.mkdir(parents=True, exist_ok=True)
        from graphify.paths import write_json_atomic as _wja
        _wja(out_path2, merged2, ensure_ascii=False)
        print(f"Merged: {len(merged2['nodes'])} nodes, {len(merged2['edges'])} edges")

    elif Path(cmd).exists() or cmd in (".", "..") or cmd.startswith(("./", "../", "/", "~")):
        # User ran `graphify <path>` directly — treat as `graphify extract <path>`.
        # Common when following the PowerShell note in README (`graphify .`) or
        # copy-pasting skill invocations without the leading slash.
        sys.argv.insert(2, sys.argv[1])
        sys.argv[1] = "extract"
        _reenter_main()
    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        print("Run 'graphify --help' for usage.", file=sys.stderr)
        sys.exit(1)
