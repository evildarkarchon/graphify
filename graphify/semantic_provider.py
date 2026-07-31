"""The LLM-backed Semantic provider a Full extraction interprets through.

``graphify.generation`` owns the Graph-generation lifecycle and admits evidence
from outside itself through two seams — a Semantic provider and the requests
that name a source system — because the systems behind them genuinely are
external. This module implements the first. A Semantic provider is asked about a
set of Corpus documents and answers with the evidence it derived from the ones
it interpreted *completely* — nothing else. Which
sources those are decides what the operation may replace, so the rule this
module exists to enforce is narrow and absolute:

* a source is reported interpreted only when this run produced its complete
  evidence, or when the provider's own accelerator cache already held it;
* a source the backend truncated, omitted, or failed on is left out of the
  answer, and whatever fragment the attempt produced stays in the cache;
* an empty answer for a source the backend genuinely had nothing to say about is
  still a complete interpretation, and is reported as one.

The cache here is an accelerator and nothing more. It is keyed by content, it
belongs to this provider, and losing it costs re-interpretation — never
correctness, because the authoritative evidence is the ledger the operation
publishes, not this cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from graphify.generation import SemanticInterpretation, SemanticRequest, SourceEvidence

# What a corpus with no configured backend is told. Both ways forward are named
# because both are legitimate: configure a backend, or index the code alone.
_NO_BACKEND = (
    "no LLM API key found ({reason}). Set GEMINI_API_KEY or GOOGLE_API_KEY "
    "(gemini), MOONSHOT_API_KEY (kimi), ANTHROPIC_API_KEY (claude), "
    "OPENAI_API_KEY (openai), DEEPSEEK_API_KEY (deepseek), or pass --backend. "
    "A code-only corpus needs no key — pass --code-only to index just the code "
    "(local AST, no key) and skip the non-code files."
)


class LlmSemanticProvider:
    """Interpret Corpus documents with a configured LLM backend.

    One instance interprets one Full extraction's documents. It accumulates what
    that cost — tokens, cache hits and misses, the backend it resolved — so the
    adapter that constructed it can report the run without inspecting anything
    private. Reused across two operations the counters simply keep adding, which
    is why callers build one per run.
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        deep: bool = False,
        refresh: bool = False,
        token_budget: int | None = None,
        max_concurrency: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Configure how this provider interprets, without resolving anything yet.

        ``backend`` and ``model`` name the LLM to use; an omitted backend is
        detected from the environment at the first interpretation that actually
        needs one, so a Corpus with no documents never requires a key.

        ``deep`` selects the richer extraction prompt and its own cache
        namespace, so deep and standard results for one document never shadow
        each other. ``refresh`` skips the cache *read* — the write still happens,
        replacing stale entries — which is how a caller asks for the whole
        semantic corpus to be re-interpreted.

        ``token_budget`` and ``max_concurrency`` tune chunking and parallelism;
        omitted, each keeps the extraction library's own default.
        ``on_progress(done, total)`` is called as chunks complete.
        """
        self._backend = backend
        self._model = model
        self._deep = deep
        self._refresh = refresh
        self._token_budget = token_budget
        self._max_concurrency = max_concurrency
        self._on_progress = on_progress
        self.backend: str | None = backend
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def interpret(self, request: SemanticRequest) -> SemanticInterpretation:
        """Return the complete interpretations of ``request.sources``.

        Raises when the provider cannot work at all — no configured backend, an
        unknown one, or a dispatch in which no chunk completed. Those are the
        cases where an empty answer would be a lie: the operation would read it
        as "nothing could be interpreted" and, with partial authority, publish a
        generation that quietly dropped every document. A run where *some* work
        succeeded never raises; it answers with what completed and lets the
        operation's own publication rules decide whether that may be published.
        """
        sources = [Path(source) for source in request.sources]
        if not sources:
            return SemanticInterpretation()
        backend = self._resolved_backend(len(sources))
        cache_root = request.output.parent
        prompt = self._prompt()
        interpreted: dict[str, SourceEvidence] = {}

        cached, uncached = self._from_cache(sources, request, cache_root, prompt)
        interpreted.update(cached)
        self.cache_hits += len(cached)
        self.cache_misses += len(uncached)

        if uncached:
            interpreted.update(
                self._from_backend(uncached, request, cache_root, prompt, backend)
            )
        self._prune_cache(sources, request.root, cache_root)
        return SemanticInterpretation(interpreted=interpreted)

    # --- backend resolution --------------------------------------------------

    def _resolved_backend(self, pending: int) -> str:
        """Return the backend to interpret with, or explain why there is none.

        Resolution is deferred to here rather than done in the constructor
        because a provider that is never asked to interpret anything must not
        require a key — a code-only Corpus is a legitimate, free operation.
        """
        from graphify.llm import BACKENDS, detect_backend

        backend = self._backend or detect_backend()
        if backend is None:
            raise ValueError(
                _NO_BACKEND.format(
                    reason=f"{pending} doc/paper/image file(s) need semantic extraction"
                )
            )
        if backend not in BACKENDS:
            raise ValueError(
                f"unknown backend '{backend}'. Available: {', '.join(sorted(BACKENDS))}"
            )
        self._require_credentials(backend)
        self.backend = backend
        return backend

    @staticmethod
    def _require_credentials(backend: str) -> None:
        """Refuse a backend whose credentials or endpoint are not usable.

        Checked before dispatch so a corpus-sized run fails in a sentence rather
        than in a wall of per-chunk errors. Three backends authenticate without
        an API key — a loopback Ollama, an AWS-profile Bedrock, an authenticated
        ``claude`` CLI — so each is asked on its own terms.
        """
        from graphify.llm import (
            BACKENDS,
            _format_backend_env_keys,
            _get_backend_api_key,
            _validate_ollama_base_url,
        )

        ollama_url = str(
            os.environ.get("OLLAMA_BASE_URL")
            or BACKENDS.get("ollama", {}).get("base_url", "")
        )
        if backend == "ollama":
            # Raises on a link-local or metadata endpoint, which is never a real
            # Ollama host and always an SSRF target.
            _validate_ollama_base_url(ollama_url, warn=False)
        if _get_backend_api_key(backend):
            return
        if backend == "ollama":
            from urllib.parse import urlparse

            try:
                host = str(urlparse(ollama_url).hostname or "").lower()
            except ValueError:
                host = ""
            if host in ("localhost", "127.0.0.1", "::1") or host.startswith("127."):
                return
        elif backend == "bedrock":
            if any(
                os.environ.get(name)
                for name in (
                    "AWS_PROFILE",
                    "AWS_REGION",
                    "AWS_DEFAULT_REGION",
                    "AWS_ACCESS_KEY_ID",
                )
            ):
                return
        elif backend == "claude-cli":
            import shutil

            if shutil.which("claude") is not None:
                return
            raise ValueError(
                "backend 'claude-cli' requires the `claude` CLI on $PATH (install "
                "Claude Code and run `claude` once to authenticate)."
            )
        raise ValueError(
            f"backend '{backend}' requires {_format_backend_env_keys(backend)} "
            "to be set."
        )

    def _prompt(self) -> str:
        """Return the extraction prompt this run's cache entries are keyed to.

        Read and write must pass the same prompt: an entry written under one
        prompt and read under another is a different vintage of interpretation,
        and serving it would replay an older release's understanding forever.
        """
        from graphify.llm import _extraction_system

        return _extraction_system(deep=self._deep)

    # --- the cache -----------------------------------------------------------

    def _namespace(self) -> str | None:
        """Return the cache namespace this mode reads and writes."""
        return "deep" if self._deep else None

    def _from_cache(
        self,
        sources: list[Path],
        request: SemanticRequest,
        cache_root: Path,
        prompt: str,
    ) -> tuple[dict[str, SourceEvidence], list[Path]]:
        """Return the interpretations already cached, and what is still missing.

        A cache hit is a complete interpretation by construction: a truncated one
        is stamped partial when it is written and never served. That is what lets
        a hit be reported as interpreted even when it carries no evidence — the
        document was read, and it had nothing to say.
        """
        if self._refresh:
            return {}, list(sources)
        from graphify.cache import check_semantic_cache

        nodes, edges, hyperedges, uncached = check_semantic_cache(
            [str(source) for source in sources],
            root=request.root,
            cache_root=cache_root,
            mode=self._namespace(),
            prompt=prompt,
        )
        missing = {Path(path).resolve() for path in uncached}
        hits = [source for source in sources if source.resolve() not in missing]
        grouped = self._group(
            {"nodes": nodes, "edges": edges, "hyperedges": hyperedges},
            hits,
            request.root,
        )
        return (
            {str(source): grouped.get(str(source), SourceEvidence()) for source in hits},
            [source for source in sources if source.resolve() in missing],
        )

    def _prune_cache(
        self, sources: list[Path], root: Path, cache_root: Path
    ) -> None:
        """Sweep cache entries no live document could ever match again.

        The semantic cache is content-hash keyed and deliberately unversioned, so
        nothing else ever removes an entry: every edit and every deletion leaves
        one behind forever. Best effort by design — a failed sweep costs disk,
        never correctness, so it must not be able to fail an interpretation.
        """
        try:
            from graphify.cache import file_hash, prune_semantic_cache

            live: set[str] = set()
            for source in sources:
                if not source.is_file():
                    continue
                try:
                    live.add(file_hash(source, root, cache_root=cache_root))
                except OSError:
                    continue
            prune_semantic_cache(cache_root, live)
        except Exception as exc:  # noqa: BLE001 — an accelerator sweep, never authority
            print(
                f"[graphify] warning: could not prune the semantic cache: {exc}",
                file=sys.stderr,
            )

    # --- dispatch ------------------------------------------------------------

    def _from_backend(
        self,
        uncached: list[Path],
        request: SemanticRequest,
        cache_root: Path,
        prompt: str,
        backend: str,
    ) -> dict[str, SourceEvidence]:
        """Interpret the uncached sources and keep only the complete answers."""
        from graphify.cache import save_semantic_cache
        from graphify.llm import (
            _partial_source_files,
            _strip_partial_markers,
            extract_corpus_parallel,
        )

        completed = 0

        def _chunk_done(index: int, total: int, _result: dict) -> None:
            """Count a completed chunk and pass the progress on to the caller."""
            nonlocal completed
            completed += 1
            if self._on_progress is not None:
                self._on_progress(index + 1, total)

        options: dict[str, Any] = {
            "backend": backend,
            "model": self._model,
            "root": request.root,
            "cache_root": cache_root,
            "on_chunk_done": _chunk_done,
        }
        if self._deep:
            options["deep_mode"] = True
        if self._token_budget is not None:
            options["token_budget"] = self._token_budget
        if self._max_concurrency is not None:
            options["max_concurrency"] = self._max_concurrency

        fresh = extract_corpus_parallel(list(uncached), **options)
        if completed == 0:
            # Not one chunk came back. Answering "nothing was interpreted" here
            # would be indistinguishable from a corpus of empty documents, and an
            # authorized partial run would publish that as a finished generation.
            raise RuntimeError(
                f"every semantic chunk failed for backend '{backend}' "
                f"({len(uncached)} file(s) dispatched) — see the per-chunk errors "
                "above. If one reads 'requires the X package', install it and retry."
            )
        self.input_tokens += int(fresh.get("input_tokens", 0) or 0)
        self.output_tokens += int(fresh.get("output_tokens", 0) or 0)

        # Computed before the cache write, which consumes the markers, and before
        # they are stripped out of the evidence itself.
        truncated = {
            self._resolved(path, request.root)
            for path in _partial_source_files(fresh)
        }
        try:
            save_semantic_cache(
                fresh.get("nodes", []),
                fresh.get("edges", []),
                fresh.get("hyperedges", []),
                root=request.root,
                cache_root=cache_root,
                allowed_source_files=[str(path) for path in uncached],
                mode=self._namespace(),
                prompt=prompt,
                partial_source_files=[str(path) for path in truncated] or None,
            )
        except Exception as exc:  # noqa: BLE001 — the cache is an accelerator
            print(
                f"[graphify] warning: could not write the semantic cache: {exc}",
                file=sys.stderr,
            )
        _strip_partial_markers(fresh)

        omitted = {
            self._resolved(path, request.root)
            for path in fresh.get("uncovered_files", []) or ()
        }
        complete = [
            source
            for source in uncached
            if source.resolve() not in truncated and source.resolve() not in omitted
        ]
        grouped = self._group(fresh, complete, request.root)
        return {
            str(source): grouped.get(str(source), SourceEvidence())
            for source in complete
        }

    # --- attribution ---------------------------------------------------------

    @staticmethod
    def _resolved(source: "str | Path", root: Path) -> Path:
        """Return one reported source path in the form dispatch keys are in.

        The backend reports a source either as the absolute path it was given or
        as the Corpus-relative spelling it attributed evidence to, depending on
        which half of the answer it came from. Both name the same document, so
        both are anchored here rather than at each call site — a mismatch would
        silently read a truncated interpretation as a complete one.
        """
        path = Path(source)
        if not path.is_absolute():
            path = root / path
        try:
            return path.resolve()
        except OSError:
            return path

    @staticmethod
    def _group(
        result: Mapping[str, Any],
        sources: Iterable[Path],
        root: Path,
    ) -> dict[str, SourceEvidence]:
        """Group one flat extraction result into per-source evidence.

        Evidence is matched to the source it is attributed to, resolved against
        the Corpus root so a relative cache entry and an absolute fresh answer
        land on the same document. Anything attributed elsewhere is dropped: a
        provider may only speak for the sources it was asked about, and the
        operation would refuse such evidence anyway.
        """
        wanted = {source.resolve(): str(source) for source in sources}
        buckets: dict[str, dict[str, list[Any]]] = {
            key: {"nodes": [], "edges": [], "hyperedges": []} for key in wanted.values()
        }
        for bucket in ("nodes", "edges", "hyperedges"):
            for item in result.get(bucket) or ():
                if not isinstance(item, dict):
                    continue
                attributed = item.get("source_file")
                if not attributed:
                    continue
                path = Path(attributed)
                if not path.is_absolute():
                    path = root / path
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                key = wanted.get(resolved)
                if key is not None:
                    buckets[key][bucket].append(item)
        return {
            key: SourceEvidence(
                nodes=tuple(group["nodes"]),
                edges=tuple(group["edges"]),
                hyperedges=tuple(group["hyperedges"]),
            )
            for key, group in buckets.items()
        }
