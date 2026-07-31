"""Adapter tests for how `graphify extract` translates publication authority.

Full extraction refuses to replace the active Graph generation from work that
did not finish, and the two authorities that can change what a run may publish —
partial publication and ``force`` — are deliberately separate. Those rules live
in ``graphify.generation`` and are tested there against the operation. What is
tested here is the CLI's half: that ``--allow-partial`` and ``--force`` reach the
request independently, and that the terminal outcome becomes the exit code and
the stream a script reads.

The backend call is the only substitution — it is the one part that would reach
a paid network service — and it is substituted at the same function the
production Semantic provider calls.
"""
from __future__ import annotations

import json

import pytest

import graphify.__main__ as mainmod


def _docs_corpus(tmp_path):
    """Return a documents-only Corpus, so interpretation is the whole run."""
    (tmp_path / "README.md").write_text("# Notes\nThe entry point overview.\n")
    (tmp_path / "GUIDE.md").write_text("# Guide\nHow to use the thing.\n")
    return tmp_path


def _arm_extract(monkeypatch, tmp_path, *, interpret_all: bool, extra_argv=()):
    """Point the CLI at a Corpus whose interpretation completes, or does not.

    ``interpret_all=False`` reproduces the shape a real backend reports when it
    answers about one document and silently omits another: the omitted file
    comes back in ``uncovered_files``, which is how "not completely interpreted"
    is expressed.
    """
    corpus = _docs_corpus(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def _stub_corpus(paths, **kwargs):
        """Interpret README.md, and GUIDE.md only when the run is complete."""
        on_chunk = kwargs.get("on_chunk_done")
        result = {
            "nodes": [
                {
                    "id": "readme_notes",
                    "source_file": str(corpus / "README.md"),
                    "file_type": "document",
                    "label": "Notes",
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "failed_chunks": 0,
            "uncovered_files": [],
        }
        if interpret_all:
            result["nodes"].append(
                {
                    "id": "guide_howto",
                    "source_file": str(corpus / "GUIDE.md"),
                    "file_type": "document",
                    "label": "Guide",
                }
            )
        else:
            result["uncovered_files"] = [str(corpus / "GUIDE.md")]
            result["failed_chunks"] = 1
        if on_chunk:
            on_chunk(0, 1, result)
        return result

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _stub_corpus)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--backend",
            "claude",
            "--no-cluster",
            "--out",
            str(out_dir),
            *extra_argv,
        ],
    )
    return out_dir / "graphify-out"


def _run() -> int:
    """Run the CLI and return the exit code it chose."""
    with pytest.raises(SystemExit) as exit_status:
        mainmod.main()
    return int(exit_status.value.code or 0)


def _node_ids(graph_path) -> set[str]:
    """Return the node identities in the published graph."""
    return {
        node["id"]
        for node in json.loads(graph_path.read_text(encoding="utf-8"))["nodes"]
    }


def test_incomplete_interpretation_refuses_and_exits_nonzero(
    monkeypatch, tmp_path, capsys
):
    """Publish nothing, and say why, when a source was not interpreted."""
    out = _arm_extract(monkeypatch, tmp_path, interpret_all=False)

    assert _run() == 1

    assert "interpretation was incomplete" in capsys.readouterr().err
    # Nothing at all was published: a refused run leaves the Corpus as it was,
    # and in particular does not stamp a manifest for a graph it declined.
    assert not (out / "graph.json").exists()
    assert not (out / "manifest.json").exists()


def test_force_alone_does_not_authorize_an_incomplete_publication(
    monkeypatch, tmp_path, capsys
):
    """Keep the two authorities separate: force is not partial authority."""
    out = _arm_extract(monkeypatch, tmp_path, interpret_all=False, extra_argv=["--force"])

    assert _run() == 1

    assert "interpretation was incomplete" in capsys.readouterr().err
    assert not (out / "graph.json").exists()


def test_allow_partial_publishes_the_sources_that_completed(monkeypatch, tmp_path):
    """Commit the interpreted source, and leave the other one pending."""
    out = _arm_extract(
        monkeypatch, tmp_path, interpret_all=False, extra_argv=["--allow-partial"]
    )

    assert _run() == 0

    assert "readme_notes" in _node_ids(out / "graph.json")
    assert "guide_howto" not in _node_ids(out / "graph.json")
    # The source that was not interpreted is disclosed as outstanding work
    # rather than being presented as done.
    assert (out / "needs_update").is_file()


def test_a_complete_interpretation_needs_no_authority(monkeypatch, tmp_path):
    """Publish a finished run without any of the safety authorities."""
    out = _arm_extract(monkeypatch, tmp_path, interpret_all=True)

    assert _run() == 0

    assert {"readme_notes", "guide_howto"} <= _node_ids(out / "graph.json")
    assert not (out / "needs_update").is_file()


def test_an_incomplete_run_preserves_the_active_generation(monkeypatch, tmp_path):
    """Leave a published generation exactly as it was when a later run fails."""
    out = _arm_extract(monkeypatch, tmp_path, interpret_all=True)
    assert _run() == 0
    published = (out / "graph.json").read_bytes()

    # --force so the second run re-interprets instead of being served the first
    # run's cached answers, which would make it complete again.
    _arm_extract(monkeypatch, tmp_path, interpret_all=False, extra_argv=["--force"])
    assert _run() == 1

    assert (out / "graph.json").read_bytes() == published


def test_an_unreadable_active_graph_is_refused_not_overwritten(
    monkeypatch, tmp_path, capsys
):
    """Refuse to build on state that cannot be validated, and keep it intact."""
    out = _arm_extract(monkeypatch, tmp_path, interpret_all=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text("{corrupt json", encoding="utf-8")

    assert _run() == 1

    assert "cannot validate" in capsys.readouterr().err.lower()
    assert (out / "graph.json").read_text(encoding="utf-8") == "{corrupt json"
