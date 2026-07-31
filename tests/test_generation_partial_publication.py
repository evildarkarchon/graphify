"""Production-operation tests for what may replace an active Graph generation.

These are the publication-safety rules that decide whether a Full extraction is
*allowed* to commit, as distinct from what it managed to produce. Issue #10 made
partial progress possible — a source whose interpretation did not complete keeps
its last complete evidence, or keeps having none, and stays pending — and these
tests cover whether publishing that state is permitted.

Three rules, each its own section below:

* An incomplete Corpus discovery or an incomplete interpretation refuses to
  replace the active generation unless the request carried partial-publication
  authority, which publishes the sources that completed and retires nothing.
* Code update never carries that authority, and ``force`` — the authority to
  replace the active graph with a smaller one — neither implies it nor is implied
  by it.
* A protected generation may not be replaced when its backup cannot be written,
  unless backup is explicitly disabled.

Every test drives the real ``CorpusGraph`` operations over a real temporary
Corpus and reads the published generation's own artifacts. The Semantic provider
is a real implementation of the production provider seam: it interprets Markdown
headings deterministically, so an interpretation that "fails" fails for the same
reason a real one does — the provider did not report that source as complete.

The tests that make discovery itself incomplete need a directory the scan cannot
enumerate, which only a POSIX host can produce; they skip elsewhere rather than
pretending Windows can express ``chmod 000``.
"""

import json
import os
import shutil
from dataclasses import fields
from datetime import date
from pathlib import Path

import pytest

from tests._generation_support import (
    _HeadingInterpreter,
    _has_record,
    _ledger_records,
    _node_ids,
)

# Coordination state lives beside the canonical artifacts but is not part of the
# Graph generation: a refused request stays durably accepted on purpose, so it
# must not read as a change to the active generation.
_COORDINATION_PREFIXES = (".graphify_requests", ".graphify_executor")


def _skip_unless_discovery_can_be_broken() -> None:
    """Skip when the host cannot produce a directory a scan may not enumerate."""
    if os.name != "posix":
        pytest.skip("only a POSIX host can make a directory unreadable reversibly")
    if getattr(os, "geteuid", lambda: 1)() == 0:
        pytest.skip("root reads an unreadable directory anyway")


def _generation(output: Path) -> dict[str, bytes]:
    """Return the active generation's canonical artifacts, by name and bytes.

    Coordination records are excluded because they are not part of the
    generation: a refusal deliberately leaves the request it refused durably
    accepted for whichever executor covers it next.
    """
    return {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
        and not path.name.startswith(_COORDINATION_PREFIXES)
    }


def _corpus(tmp_path: Path):
    """Return an owner over a small mixed Corpus, with its output directory."""
    from graphify.generation import Corpus, CorpusGraph

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    (tmp_path / "design.md").write_text("# Design\n\nOriginal.\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    return CorpusGraph(Corpus(root=tmp_path, output=output)), output


# --- incomplete work refuses by default --------------------------------------


def test_incomplete_interpretation_refuses_publication_by_default(tmp_path) -> None:
    """Keep the active generation when a requested source was not interpreted."""
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    published = _generation(output)
    (tmp_path / "guide.md").write_text("# Rewritten\n", encoding="utf-8")

    outcome = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        )
    )

    assert isinstance(outcome, PublicationRefused)
    # The refusal names what was left undone, because that is the part a caller
    # can act on — by fixing it, or by authorizing the partial result.
    assert "guide.md" in outcome.reason
    assert "partial" in outcome.reason
    assert _generation(output) == published


def test_incomplete_discovery_refuses_publication_by_default(tmp_path) -> None:
    """Never replace a complete generation from a Corpus that was half-read."""
    _skip_unless_discovery_can_be_broken()
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
    )

    owner, output = _corpus(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("class Hidden:\n    pass\n", encoding="utf-8")
    seeded = owner.full_extraction(FullExtractionRequest())
    assert isinstance(seeded, GenerationPublished)
    published = _generation(output)
    os.chmod(locked, 0o000)

    try:
        outcome = owner.full_extraction(FullExtractionRequest())
    finally:
        os.chmod(locked, 0o755)

    assert isinstance(outcome, PublicationRefused)
    assert "discovery was incomplete" in outcome.reason
    assert "partial" in outcome.reason
    assert _generation(output) == published


def test_a_failed_google_workspace_export_refuses_publication_by_default(
    tmp_path,
) -> None:
    """Treat a shortcut that produced no document as incomplete discovery.

    Requesting Google Workspace admits shortcuts as Corpus documents by exporting
    them to Markdown sidecars. A shortcut with no sidecar is a source the Corpus
    was asked to reconcile and does not have, which is the same incompleteness an
    unreadable directory produces and refuses the same way.
    """
    if shutil.which("gws") is not None:
        pytest.skip("the Google Workspace exporter is installed; this would go online")
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        GoogleWorkspaceSource,
        PublicationRefused,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(FullExtractionRequest())
    assert isinstance(seeded, GenerationPublished)
    published = _generation(output)
    (tmp_path / "notes.gdoc").write_text(
        json.dumps({"url": "https://docs.google.com/document/d/abc123/edit"}),
        encoding="utf-8",
    )

    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(GoogleWorkspaceSource(),))
    )

    assert isinstance(outcome, PublicationRefused)
    assert "discovery was incomplete" in outcome.reason
    assert "Google Workspace" in outcome.reason
    assert _generation(output) == published


def test_force_does_not_authorize_publishing_an_incomplete_extraction(
    tmp_path,
) -> None:
    """Keep the two authorities separate: ``force`` is not partial authority.

    ``force`` says the active graph may be replaced by a smaller one. It says
    nothing about whether the run that produced the smaller graph finished, so
    it may not stand in for the authority that does.
    """
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    published = _generation(output)

    outcome = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            ),
            force=True,
        )
    )

    assert isinstance(outcome, PublicationRefused)
    assert _generation(output) == published


# --- explicit partial-publication authority ----------------------------------


def test_authorized_partial_publication_commits_the_sources_that_completed(
    tmp_path,
) -> None:
    """Publish successful sources while a failed one stays absent and pending."""
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)

    outcome = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            ),
            allow_partial_publication=True,
        )
    )

    assert isinstance(outcome, GenerationPublished)
    assert _has_record(output, "design.md", "semantic")
    assert "design.md::Design" in _node_ids(output)
    assert _has_record(output, "service.py", "structural")
    # The failed source was never interpreted, so it has no evidence to keep —
    # and the generation says plainly that its interpretation is outstanding.
    assert not _has_record(output, "guide.md", "semantic")
    assert "guide.md::Guide" not in _node_ids(output)
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_authorized_partial_publication_keeps_a_failed_sources_prior_evidence(
    tmp_path,
) -> None:
    """Retain what a failed source already had rather than losing it on failure.

    This is also why partial-publication authority never needs ``force``: an
    authorized partial run replaces the sources that completed and retires
    nothing else, so the published graph never shrinks for a reason the run
    could not account for.
    """
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        SemanticSource,
        stale_semantic_sources,
    )

    owner, output = _corpus(tmp_path)
    # No provider, so both documents are represented by structural evidence.
    seeded = owner.full_extraction(FullExtractionRequest())
    assert isinstance(seeded, GenerationPublished)
    assert _has_record(output, "guide.md", "structural")

    outcome = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            ),
            allow_partial_publication=True,
            force=False,
        )
    )

    assert isinstance(outcome, GenerationPublished)
    # The source that interpreted is now represented by its interpretation.
    assert _has_record(output, "design.md", "semantic")
    assert not _has_record(output, "design.md", "structural")
    # The source that did not keeps the only evidence the Corpus has for it.
    assert _has_record(output, "guide.md", "structural")
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"
    # Nothing is stale: stale describes retained *Semantic* evidence that now
    # predates its source, and this source has never been interpreted.
    assert stale_semantic_sources(output) == ()


def test_authorized_partial_publication_keeps_undiscovered_sources(tmp_path) -> None:
    """Treat a source discovery could not reach as retained, not as deleted.

    Absence is deletion evidence only when the scan that produced it was
    complete. Authorizing a partial publication says the run may commit what it
    did see; it never says the part it could not see has gone away.
    """
    _skip_unless_discovery_can_be_broken()
    from graphify.generation import (
        CorpusStateAdvanced,
        FullExtractionRequest,
        GenerationPublished,
    )

    owner, output = _corpus(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("class Hidden:\n    pass\n", encoding="utf-8")
    seeded = owner.full_extraction(FullExtractionRequest())
    assert isinstance(seeded, GenerationPublished)
    hidden_nodes = {
        node["id"]
        for record in _ledger_records(output)
        if record["source"] == "locked/hidden.py"
        for node in record["nodes"]
    }
    assert hidden_nodes
    os.chmod(locked, 0o000)

    try:
        outcome = owner.full_extraction(
            FullExtractionRequest(allow_partial_publication=True)
        )
    finally:
        os.chmod(locked, 0o755)

    # The ledger is unchanged — every live source was re-derived to the same
    # evidence and the unreachable one was kept — so only Corpus state advances.
    assert isinstance(outcome, CorpusStateAdvanced)
    assert _has_record(output, "locked/hidden.py", "structural")
    assert hidden_nodes <= _node_ids(output)
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


def test_authorized_partial_publication_keeps_an_undiscovered_sources_staleness(
    tmp_path,
) -> None:
    """Never withdraw a stale disclosure this run had no evidence to withdraw.

    Staleness is re-derived every run from the manifest, which only describes
    sources the scan actually saw. A source behind an unreadable location was
    disclosed as stale by the last run that could see it, and this run learned
    nothing that would change that — so republishing it as current would quietly
    present evidence known to predate its source as though it described it now.
    """
    _skip_unless_discovery_can_be_broken()
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        SemanticSource,
        stale_semantic_sources,
    )

    owner, output = _corpus(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "notes.md").write_text("# Notes\n", encoding="utf-8")
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    # Change the source and fail to reinterpret it, so it is disclosed as stale.
    (locked / "notes.md").write_text("# Rewritten\n", encoding="utf-8")
    stale = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"locked/notes.md"})),
            ),
            allow_partial_publication=True,
        )
    )
    assert isinstance(stale, GenerationPublished)
    assert stale_semantic_sources(output) == ("locked/notes.md",)
    os.chmod(locked, 0o000)

    try:
        owner.full_extraction(
            FullExtractionRequest(
                sources=(SemanticSource(provider=_HeadingInterpreter()),),
                allow_partial_publication=True,
            )
        )
    finally:
        os.chmod(locked, 0o755)

    assert stale_semantic_sources(output) == ("locked/notes.md",)
    assert (output / "needs_update").read_text(encoding="utf-8") == "1"


# --- Code update stays fail closed -------------------------------------------


def test_code_update_cannot_carry_partial_publication_authority(tmp_path) -> None:
    """Refuse the authority itself, at the type and at the durable queue.

    Code update is the cheap, deterministic operation. Letting it publish a
    partial result would make the low-cost path the one that weakens publication
    safety, so it cannot ask for the authority and the queue will not record a
    request claiming it.
    """
    from graphify.generation import CodeUpdateRequest, Corpus
    from graphify.generation._coordination import (
        _RequestCoordinator,
        _RequestedAuthority,
    )

    assert "allow_partial_publication" not in {
        field.name for field in fields(CodeUpdateRequest)
    }

    coordinator = _RequestCoordinator(
        Corpus(root=tmp_path, output=tmp_path / "graphify-out")
    )
    with pytest.raises(ValueError):
        coordinator.accept(
            "code-update",
            authority=_RequestedAuthority(allow_partial_publication=True),
        )


def test_code_update_refuses_incomplete_discovery_even_when_forced(tmp_path) -> None:
    """Keep ``force`` from becoming a way to update from an unknown Corpus."""
    _skip_unless_discovery_can_be_broken()
    from graphify.generation import (
        CodeUpdateRequest,
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
    )

    owner, output = _corpus(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("class Hidden:\n    pass\n", encoding="utf-8")
    seeded = owner.full_extraction(FullExtractionRequest())
    assert isinstance(seeded, GenerationPublished)
    published = _generation(output)
    os.chmod(locked, 0o000)

    try:
        outcome = owner.code_update(CodeUpdateRequest(force=True))
    finally:
        os.chmod(locked, 0o755)

    assert isinstance(outcome, PublicationRefused)
    assert "discovery was incomplete" in outcome.reason
    assert _generation(output) == published


# --- protected generations may not be replaced without a backup ---------------


def test_protected_backup_failure_preserves_the_entire_active_generation(
    tmp_path,
) -> None:
    """Refuse a protected replacement whose snapshot could not be taken.

    A curated Community label makes the active generation protected. Backup is
    then a precondition of replacing it rather than a best-effort courtesy, so a
    backup that cannot be written stops the whole operation — every canonical
    artifact keeps the bytes it had.
    """
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        PublicationRefused,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    (output / ".graphify_labels.json").write_text(
        json.dumps({"0": "Curated Naming"}), encoding="utf-8"
    )
    # A file where the dated backup folder has to go, so the snapshot fails for
    # a reason that has nothing to do with the candidate.
    blocked_backup = output / date.today().isoformat()
    blocked_backup.write_text("not a directory", encoding="utf-8")
    protected = _generation(output)
    (tmp_path / "extra.py").write_text("class Extra:\n    pass\n", encoding="utf-8")

    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )

    assert isinstance(outcome, PublicationRefused)
    assert "protected backup failed" in outcome.reason
    assert _generation(output) == protected
    assert "extra.py" not in json.dumps(_ledger_records(output))
    assert blocked_backup.read_text(encoding="utf-8") == "not a directory"
    assert not (output / ".graphify_publication").exists()
    assert not (output / ".graphify_publication.json").exists()


def test_explicitly_disabling_backup_allows_the_protected_publication(
    tmp_path,
    monkeypatch,
) -> None:
    """Publish over a protected generation only on the explicit opt-out."""
    from graphify.generation import (
        FullExtractionRequest,
        GenerationPublished,
        SemanticSource,
    )

    owner, output = _corpus(tmp_path)
    seeded = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )
    assert isinstance(seeded, GenerationPublished)
    (output / ".graphify_labels.json").write_text(
        json.dumps({"0": "Curated Naming"}), encoding="utf-8"
    )
    blocked_backup = output / date.today().isoformat()
    blocked_backup.write_text("not a directory", encoding="utf-8")
    (tmp_path / "extra.py").write_text("class Extra:\n    pass\n", encoding="utf-8")
    monkeypatch.setenv("GRAPHIFY_NO_BACKUP", "1")

    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),))
    )

    assert isinstance(outcome, GenerationPublished)
    assert _has_record(output, "extra.py", "structural")
    assert blocked_backup.read_text(encoding="utf-8") == "not a directory"
