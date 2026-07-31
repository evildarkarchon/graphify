"""Production-operation tests for the lifecycle observation seam.

An adapter that renders progress must not have to guess at private function
names or scrape printed output, and it must not be able to break the operation
it is watching. These tests drive the real ``CorpusGraph.full_extraction``
operation over a real temporary Corpus and assert both halves of that contract
from the outside: what an adapter is told, and what happens when the adapter
itself misbehaves.
"""

import pytest

from tests._generation_support import _HeadingInterpreter


class _Recorder:
    """Record every observation an operation reports, in order."""

    def __init__(self) -> None:
        """Start with nothing observed."""
        self.observations: list = []

    def observe(self, observation) -> None:
        """Append one lifecycle observation."""
        self.observations.append(observation)

    def kinds(self) -> list[str]:
        """Return the observed types by name, which is what ordering is about."""
        return [type(observation).__name__ for observation in self.observations]


def _corpus(tmp_path):
    """Return an owner over a small mixed Corpus, with its output directory."""
    from graphify.generation import Corpus, CorpusGraph

    (tmp_path / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n\nOriginal.\n", encoding="utf-8")
    output = tmp_path / "graphify-out"
    return CorpusGraph(Corpus(root=tmp_path, output=output)), output


def test_full_extraction_reports_ordered_lifecycle_facts(tmp_path) -> None:
    """Report discovery, then each source's evidence, then the commit."""
    from graphify.generation import (
        CorpusDiscovered,
        EvidenceCollected,
        EvidenceOrigin,
        FullExtractionRequest,
        GenerationPublished,
        PublicationStarted,
        SemanticSource,
        SourcesInterpreted,
    )

    owner, _ = _corpus(tmp_path)
    recorder = _Recorder()

    outcome = owner.full_extraction(
        FullExtractionRequest(sources=(SemanticSource(provider=_HeadingInterpreter()),)),
        observer=recorder,
    )

    assert isinstance(outcome, GenerationPublished)
    kinds = recorder.kinds()
    assert kinds[0] == "CorpusDiscovered"
    assert kinds[-1] == "PublicationStarted"
    # Interpretation is reported before the commit it feeds, and the commit is
    # announced once — the terminal outcome is not repeated as an observation.
    assert kinds.count("PublicationStarted") == 1
    assert kinds.index("SourcesInterpreted") < kinds.index("PublicationStarted")

    discovered = recorder.observations[0]
    assert isinstance(discovered, CorpusDiscovered)
    assert discovered.sources == 2
    assert discovered.semantic_sources == 1
    assert discovered.complete is True

    interpreted = next(
        observation
        for observation in recorder.observations
        if isinstance(observation, SourcesInterpreted)
    )
    assert (interpreted.requested, interpreted.interpreted) == (1, 1)

    collected = {
        observation.origin: observation
        for observation in recorder.observations
        if isinstance(observation, EvidenceCollected)
    }
    assert collected[EvidenceOrigin.SEMANTIC].nodes == 1
    assert collected[EvidenceOrigin.CORPUS].nodes > 0
    assert isinstance(recorder.observations[-1], PublicationStarted)


def test_a_refused_publication_never_announces_a_commit(tmp_path) -> None:
    """Report discovery's incompleteness and stop short of the commit."""
    from graphify.generation import (
        FullExtractionRequest,
        PublicationRefused,
        SemanticSource,
        SourcesInterpreted,
    )

    owner, _ = _corpus(tmp_path)
    recorder = _Recorder()

    outcome = owner.full_extraction(
        FullExtractionRequest(
            sources=(
                SemanticSource(provider=_HeadingInterpreter(refuse={"guide.md"})),
            )
        ),
        observer=recorder,
    )

    assert isinstance(outcome, PublicationRefused)
    assert "PublicationStarted" not in recorder.kinds()
    interpreted = next(
        observation
        for observation in recorder.observations
        if isinstance(observation, SourcesInterpreted)
    )
    assert (interpreted.requested, interpreted.interpreted) == (1, 0)


def test_a_failing_observation_adapter_cannot_fail_the_operation(tmp_path) -> None:
    """Warn about a broken adapter and publish the generation regardless."""
    from graphify.generation import FullExtractionRequest, GenerationPublished

    class _Broken:
        """An adapter whose rendering is the thing that is broken."""

        def observe(self, observation) -> None:
            """Fail the way a rendering bug would."""
            raise RuntimeError("the progress renderer is broken")

    owner, output = _corpus(tmp_path)

    with pytest.warns(RuntimeWarning, match="observation adapter"):
        outcome = owner.full_extraction(FullExtractionRequest(), observer=_Broken())

    assert isinstance(outcome, GenerationPublished)
    assert (output / "graph.json").is_file()


def test_an_operation_without_an_observer_publishes_unchanged(tmp_path) -> None:
    """Keep observation optional: no adapter, no behavior change."""
    from graphify.generation import FullExtractionRequest, GenerationPublished

    owner, output = _corpus(tmp_path)

    outcome = owner.full_extraction(FullExtractionRequest())

    assert isinstance(outcome, GenerationPublished)
    assert (output / "graph.json").is_file()
