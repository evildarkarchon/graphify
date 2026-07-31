"""Shared helpers for the production-operation Graph-generation tests.

Not a test module — pytest collects ``test_*.py`` only, so this is imported by
the suites that drive ``CorpusGraph`` over a real temporary Corpus. It holds the
Semantic provider they interpret with and the readers that inspect a published
generation's own artifacts, so two suites cannot read the same ledger, graph, or
manifest in two subtly different ways.

Each suite still builds its own Corpus: what a test puts in the Corpus is part of
what it is testing, and sharing that would couple unrelated suites to one shape.
"""

import json
import os
from pathlib import Path


class _HeadingInterpreter:
    """Interpret Markdown headings into concept nodes.

    A genuine :class:`graphify.generation.SemanticProvider`: it reads the sources
    it is given and answers with the evidence it derived from them. ``refuse``
    names Corpus-relative sources it reports as *not* completely interpreted, by
    leaving them out of its answer — the same way a real provider reports a chunk
    that failed or truncated.

    A refused source still had a fragment extracted before it was cut short, and
    that fragment is written to this provider's own cache under the Corpus output
    — which is exactly where a partial result is allowed to live, and nowhere
    else.
    """

    def __init__(self, *, refuse: frozenset[str] | set[str] = frozenset()) -> None:
        """Interpret every requested source except those named in ``refuse``."""
        self.refuse = frozenset(refuse)
        self.requested: list[str] = []

    def interpret(self, request):
        """Return concept evidence for every requested source it could read."""
        from graphify.generation import SemanticInterpretation, SourceEvidence

        interpreted: dict[str, SourceEvidence] = {}
        for source in request.sources:
            identity = Path(os.path.relpath(source, request.root)).as_posix()
            self.requested.append(identity)
            nodes = tuple(
                {
                    "id": f"{identity}::{heading}",
                    "label": heading,
                    "file_type": "concept",
                }
                for heading in [
                    line[2:].strip()
                    for line in Path(source).read_text(encoding="utf-8").splitlines()
                    if line.startswith("# ")
                ]
            )
            if identity in self.refuse:
                self._cache_fragment(request, identity, nodes)
                continue
            interpreted[identity] = SourceEvidence(nodes=nodes)
        return SemanticInterpretation(interpreted=interpreted)

    @staticmethod
    def fragment_path(output: Path, identity: str) -> Path:
        """Return where this provider caches the fragment of a cut-short source."""
        return output / "cache" / "fragments" / f"{identity.replace('/', '_')}.json"

    def _cache_fragment(self, request, identity: str, nodes: tuple) -> None:
        """Keep what was extracted before the interpretation was cut short."""
        path = self.fragment_path(request.output, identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": list(nodes)}), encoding="utf-8")


def _ledger_records(output: Path) -> list[dict]:
    """Return the active Source-contribution records for one Corpus output."""
    lines = (
        (output / ".graphify_contributions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return [json.loads(line) for line in lines[1:]]


def _record_for(output: Path, source: str, interpretation: str) -> dict:
    """Return one ledger record, failing the test when it is absent."""
    for record in _ledger_records(output):
        if record["source"] == source and record["interpretation"] == interpretation:
            return record
    raise AssertionError(
        f"no {interpretation} contribution for {source!r} in {_ledger_records(output)}"
    )


def _has_record(output: Path, source: str, interpretation: str) -> bool:
    """Return whether the active ledger carries one contribution."""
    return any(
        record["source"] == source and record["interpretation"] == interpretation
        for record in _ledger_records(output)
    )


def _node_ids(output: Path) -> set[str]:
    """Return the node identities in the published materialized graph."""
    graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    return {node["id"] for node in graph["nodes"]}


def _manifest(output: Path) -> dict:
    """Return the published manifest keyed by its portable relative paths."""
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
