"""Adapter tests for the ``graphify update`` CLI.

``update`` is a Code-update adapter: it translates the command line into one
request, submits it with the interactive completion policy, renders the terminal
outcome, and turns that outcome into an exit code. What the operation itself
does with the Corpus is covered in tests/test_generation_code_update.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _corpus(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _break_the_ledger(out: Path) -> None:
    """Corrupt the authoritative ledger so publication refuses."""
    (out / ".graphify_contributions.jsonl").write_text("not a ledger\n", encoding="utf-8")


def test_update_publishes_and_exits_zero(tmp_path):
    root = _corpus(tmp_path)

    result = _run(["update", "."], root)

    assert result.returncode == 0, result.stderr
    assert "Graph generation published" in result.stdout
    assert "Code graph updated." in result.stdout
    assert (root / "graphify-out" / "graph.json").is_file()


def test_update_reports_an_already_current_corpus_without_failing(tmp_path):
    """Nothing to do is a successful update, not an error."""
    root = _corpus(tmp_path)
    assert _run(["update", "."], root).returncode == 0

    result = _run(["update", "."], root)

    assert result.returncode == 0, result.stderr
    assert "already covers this request" in result.stdout


def test_update_exits_nonzero_when_publication_is_refused(tmp_path):
    root = _corpus(tmp_path)
    assert _run(["update", "."], root).returncode == 0
    _break_the_ledger(root / "graphify-out")

    result = _run(["update", "."], root)

    assert result.returncode == 1
    assert "Publication refused" in result.stderr
    assert "The Corpus was not updated" in result.stderr


def test_update_accepts_no_cluster_as_implied(tmp_path):
    """A Code update publishes a Raw graph generation, so the flag is redundant.

    It stays accepted rather than becoming an unknown-option error, because
    installed scripts and hooks pass it.
    """
    root = _corpus(tmp_path)

    result = _run(["update", ".", "--no-cluster"], root)

    assert result.returncode == 0, result.stderr
    assert "--no-cluster is implied" in result.stdout


def test_update_rejects_an_unknown_option(tmp_path):
    result = _run(["update", ".", "--recluster"], _corpus(tmp_path))

    assert result.returncode == 2
    assert "unknown update option" in result.stderr


def test_update_falls_back_when_the_recorded_root_is_gone(tmp_path):
    """A graphify-out/ that travelled keeps working from the new checkout.

    The root marker records the absolute Corpus the owning module published, so
    it cannot stay portable the way the old relative spelling did (#777). The
    adapter absorbs that: a recorded root that is not there is not a reason to
    refuse to update an otherwise healthy graph.
    """
    original = _corpus(tmp_path / "checkout")
    assert _run(["update", "."], original).returncode == 0
    moved = tmp_path / "elsewhere"
    original.rename(moved)
    assert (moved / "graphify-out" / ".graphify_root").read_text(
        encoding="utf-8"
    ) == str(original)

    result = _run(["update"], moved)

    assert result.returncode == 0, result.stderr
    assert "path not found" not in result.stderr
    assert (moved / "graphify-out" / ".graphify_root").read_text(
        encoding="utf-8"
    ) == str(moved)
