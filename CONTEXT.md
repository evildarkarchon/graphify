# Graphify

Graphify turns a corpus of source material into a navigable knowledge graph. This glossary distinguishes the source interpretation operations that keep that graph current.

## Language

**Corpus**:
The source material within a requested root that remains after discovery and exclusions.

**Full extraction**:
An operation that reconciles all supported corpus sources into the graph, including sources that require semantic interpretation.
_Avoid_: Update

**Code update**:
An operation that reconciles structurally extractable source changes into the graph without reinterpreting semantic sources.
_Avoid_: Full extraction

**Reclustering**:
An operation that republishes community identity and analysis from the active source contributions without reinterpreting corpus sources.
_Avoid_: Full extraction, Code update

**Semantic evidence**:
Graph knowledge derived from the meaning of corpus sources rather than solely from their structural syntax.

**Stale semantic evidence**:
Semantic evidence whose attributed source has changed since it was last interpreted.
_Avoid_: Needs-update flag

**Source contribution**:
The authoritative structural or semantic graph evidence admitted from one corpus source before graph-wide deduplication.
_Avoid_: Cache entry

**Graph generation**:
The mutually consistent graph and corpus state published by one completed full extraction, code update, or reclustering.
_Avoid_: Output files

**Raw graph generation**:
A graph generation that materializes the Corpus topology without community identity, analysis, or community labels.
_Avoid_: Incomplete graph generation

**Corpus graph**:
The durable knowledge graph derived from one corpus and represented by its active graph generation.

**Community label**:
The display name published for one graph community. Every clustered graph generation has a deterministic community label; semantic interpretation may enrich it but is not required for completeness.

**Curated community label**:
A community label intentionally supplied by a human or skill. It outranks generated labels while its recorded community membership signature remains valid.

**Semantic labeling**:
Optional interpretation that enriches community labels without changing the source contributions or topology of the corpus graph.
_Avoid_: Reclustering
