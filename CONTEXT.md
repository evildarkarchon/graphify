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

**Evidence source**:
A canonical origin of graph evidence a full extraction may be asked to reconcile: the corpus filesystem, an external semantic provider, or a source system. Every requested evidence source must complete before the generation is committed.
_Avoid_: Backend, Integration

**Source system**:
An evidence source that is not a corpus file, such as a PostgreSQL schema or a Cargo workspace. Its evidence is keyed by the system's own address, and corpus discovery is not authoritative about whether it still exists.
_Avoid_: External file

**Semantic provider**:
The external system a full extraction asks to interpret corpus sources. It reports only the sources it interpreted completely; a source it omits was not interpreted.
_Avoid_: LLM, Backend

**Source evidence**:
The graph evidence one evidence source reports for one corpus source. It becomes a source contribution once the owning operation admits it.
_Avoid_: Result, Payload

**Corpus build policy**:
The recorded discovery shaping — extra exclusions, whether VCS ignore files are honored — that a graph generation was built under. Full extraction preserves it unless replacement or clearing is explicitly requested.
_Avoid_: Config, Settings

**Incomplete discovery**:
A corpus scan that could not enumerate every source it was asked about, so absence is not evidence that a source left the corpus.
_Avoid_: Partial scan

**Partial-publication authority**:
The requested authority that lets one full extraction commit after incomplete discovery or incomplete interpretation. The sources that completed are published; the rest keep their prior evidence or keep having none and remain pending. Code update never carries it, and it is separate from the authority to publish a smaller graph.
_Avoid_: Force, Override

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

**Accepted request**:
One durably recorded ask for a corpus operation, owned by the corpus rather than by the process that submitted it.
_Avoid_: Pending change

**Requested authority**:
An explicit permission an accepted request carries to bypass one safety rule, such as replacing a smaller graph, publishing a partial full extraction, replacing the corpus build policy, or overwriting curated community labels.
_Avoid_: Flag, Option

**Request coalescing**:
The deterministic merge that turns the accepted requests for one corpus into the fewest operations that still cover every one of them, carrying each request's changed-path hints and requested authority onto the operation that covers it.
_Avoid_: Debounce, Deduplication
