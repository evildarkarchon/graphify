"""Tests for the LLM-backed Semantic provider adapter.

The provider is the production seam a Full extraction interprets through, so
these tests drive the real :class:`graphify.semantic_provider.LlmSemanticProvider`
over real temporary files and a real on-disk semantic cache. Only the backend
call itself is substituted — it is the one part that would reach a paid network
service — and it is substituted at the same function the provider calls in
production, so nothing about the seam is invented for the test.
"""

from pathlib import Path

import pytest


def _request(root: Path, *sources: str):
    """Return the interpretation request a Full extraction would submit."""
    from graphify.generation import SemanticRequest

    return SemanticRequest(
        root=root,
        output=root / "graphify-out",
        sources=tuple(root / source for source in sources),
    )


def _corpus(tmp_path: Path) -> Path:
    """Return a Corpus root holding two documents."""
    (tmp_path / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    return tmp_path


def _backend(monkeypatch, result: dict):
    """Substitute the backend call, recording the files it was given."""
    dispatched: list[list[str]] = []

    def _extract(files, **kwargs):
        dispatched.append([str(path) for path in files])
        on_chunk_done = kwargs.get("on_chunk_done")
        answer = dict(result)
        if callable(on_chunk_done) and not answer.get("failed_chunks"):
            on_chunk_done(0, 1, answer)
        return answer

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _extract)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return dispatched


def _node(source: str, label: str) -> dict:
    """Return one semantic node attributed to a Corpus-relative source."""
    return {
        "id": f"{source}::{label}",
        "label": label,
        "file_type": "concept",
        "source_file": source,
    }


def test_only_completely_interpreted_sources_are_reported(monkeypatch, tmp_path):
    """Leave a source the model omitted out of the answer entirely."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "failed_chunks": 0,
            "uncovered_files": [str(root / "notes.md")],
        },
    )

    interpretation = LlmSemanticProvider(backend="gemini").interpret(
        _request(root, "guide.md", "notes.md")
    )

    assert set(interpretation.interpreted) == {str(root / "guide.md")}
    evidence = interpretation.interpreted[str(root / "guide.md")]
    assert [node["id"] for node in evidence.nodes] == ["guide.md::Guide"]


def test_a_truncated_source_is_not_reported_as_interpreted(monkeypatch, tmp_path):
    """Treat a truncated interpretation as incomplete, fragment and all."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide"), _node("notes.md", "Notes")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "failed_chunks": 0,
            "uncovered_files": [],
            "_partial_files": [str(root / "notes.md")],
        },
    )

    interpretation = LlmSemanticProvider(backend="gemini").interpret(
        _request(root, "guide.md", "notes.md")
    )

    assert set(interpretation.interpreted) == {str(root / "guide.md")}


def test_a_source_with_nothing_to_say_is_still_interpreted(monkeypatch, tmp_path):
    """Report an empty but complete interpretation as complete."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "failed_chunks": 0,
            "uncovered_files": [],
        },
    )

    interpretation = LlmSemanticProvider(backend="gemini").interpret(
        _request(root, "guide.md", "notes.md")
    )

    assert set(interpretation.interpreted) == {
        str(root / "guide.md"),
        str(root / "notes.md"),
    }
    assert interpretation.interpreted[str(root / "notes.md")].nodes == ()


def test_a_cached_source_is_served_without_dispatch(monkeypatch, tmp_path):
    """Serve a repeat interpretation from the provider's own accelerator cache."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    dispatched = _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide"), _node("notes.md", "Notes")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "failed_chunks": 0,
            "uncovered_files": [],
        },
    )
    first = LlmSemanticProvider(backend="gemini")
    first.interpret(_request(root, "guide.md", "notes.md"))
    assert first.cache_hits == 0
    assert first.cache_misses == 2

    second = LlmSemanticProvider(backend="gemini")
    interpretation = second.interpret(_request(root, "guide.md", "notes.md"))

    assert len(dispatched) == 1  # nothing was dispatched the second time
    assert second.cache_hits == 2
    assert set(interpretation.interpreted) == {
        str(root / "guide.md"),
        str(root / "notes.md"),
    }
    assert [
        node["id"] for node in interpretation.interpreted[str(root / "guide.md")].nodes
    ] == ["guide.md::Guide"]


def test_refresh_re_dispatches_a_cached_source(monkeypatch, tmp_path):
    """Skip the cache read when the caller asked for fresh interpretation."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    dispatched = _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide"), _node("notes.md", "Notes")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "failed_chunks": 0,
            "uncovered_files": [],
        },
    )
    LlmSemanticProvider(backend="gemini").interpret(_request(root, "guide.md"))

    LlmSemanticProvider(backend="gemini", refresh=True).interpret(
        _request(root, "guide.md")
    )

    assert len(dispatched) == 2


def test_every_chunk_failing_is_a_provider_failure(monkeypatch, tmp_path):
    """Raise rather than answer emptily when no chunk completed at all."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    _backend(
        monkeypatch,
        {
            "nodes": [],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "failed_chunks": 1,
            "uncovered_files": [str(root / "guide.md")],
        },
    )

    with pytest.raises(RuntimeError, match="semantic chunk"):
        LlmSemanticProvider(backend="gemini").interpret(_request(root, "guide.md"))


def test_tokens_are_accumulated_for_the_caller(monkeypatch, tmp_path):
    """Report what interpretation cost so an adapter need not infer it."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    _backend(
        monkeypatch,
        {
            "nodes": [_node("guide.md", "Guide")],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 120,
            "output_tokens": 34,
            "failed_chunks": 0,
            "uncovered_files": [],
        },
    )

    provider = LlmSemanticProvider(backend="gemini")
    provider.interpret(_request(root, "guide.md"))

    assert (provider.input_tokens, provider.output_tokens) == (120, 34)


def test_no_configured_backend_names_the_code_only_escape(monkeypatch, tmp_path):
    """Refuse before any work, explaining both ways out."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)
    monkeypatch.setattr("graphify.llm.detect_backend", lambda: None)

    with pytest.raises(ValueError) as failure:
        LlmSemanticProvider().interpret(_request(root, "guide.md"))

    assert "no LLM API key found" in str(failure.value)
    assert "code-only corpus needs no key" in str(failure.value)


def test_an_unknown_backend_is_refused(monkeypatch, tmp_path):
    """Name the available backends rather than failing at the network."""
    from graphify.semantic_provider import LlmSemanticProvider

    root = _corpus(tmp_path)

    with pytest.raises(ValueError, match="unknown backend"):
        LlmSemanticProvider(backend="not-a-backend").interpret(
            _request(root, "guide.md")
        )


def test_nothing_to_interpret_never_needs_a_backend(tmp_path):
    """Answer an empty request without resolving a backend at all."""
    from graphify.semantic_provider import LlmSemanticProvider

    interpretation = LlmSemanticProvider().interpret(_request(tmp_path))

    assert interpretation.interpreted == {}
