# Architecture

graphify is a Claude Code skill backed by a Python library. The skill orchestrates the library; the library can be used standalone.

## Pipeline

```
detect()  →  extract()  →  build_graph()  →  cluster()  →  analyze()  →  report()  →  export()
```

Each stage is a single function in its own module. They communicate through plain Python dicts and NetworkX graphs - no shared state, no side effects outside `graphify-out/`.

## Module responsibilities

| Module | Function | Input → Output |
|--------|----------|----------------|
| `detect.py` | `collect_files(root)` | directory → `[Path]` filtered list |
| `extract.py` | `extract(path)` | file path → `{nodes, edges}` dict |
| `build.py` | `build_graph(extractions)` | list of extraction dicts → `nx.Graph` |
| `cluster.py` | `cluster(G)` | graph → graph with `community` attr on each node |
| `analyze.py` | `analyze(G)` | graph → analysis dict (god nodes, surprises, questions) |
| `report.py` | `render_report(G, analysis)` | graph + analysis → GRAPH_REPORT.md string |
| `export.py` | `export(G, out_dir, ...)` | graph → Obsidian vault, graph.json, graph.html, graph.svg |
| `callflow_html.py` | `write_callflow_html(...)` | graphify-out files → Mermaid architecture/call-flow HTML |
| `ingest.py` | `ingest(url, ...)` | URL → file saved to corpus dir |
| `cache.py` | `check_semantic_cache / save_semantic_cache` | files → (cached, uncached) split |
| `security.py` | validation helpers | URL / path / label → validated or raises |
| `validate.py` | `validate_extraction(data)` | extraction dict → raises on schema errors |
| `serve.py` | `start_server(graph_path)` | graph file path → MCP stdio server |
| `watch.py` | `watch(root)` | directory → submits a code update on change; also hosts the shared code-update adapter helpers the CLI and hooks call |
| `generation/` | `CorpusGraph(corpus).full_extraction(...)` / `.code_update(...)` | request → published graph generation |
| `benchmark.py` | `run_benchmark(graph_path)` | graph file → corpus vs subgraph token comparison |

## Graph generation ownership

`graphify.generation` owns the graph-generation lifecycle for one corpus and is
the only module that publishes canonical artifacts. Its public interface is the
corpus-bound `CorpusGraph`, which exposes full extraction, code update, and
reclustering. Every other entrypoint is an adapter: it translates a request,
picks a completion policy, renders the terminal outcome, and does nothing else.

| Entrypoint | Completion | Notes |
|------------|------------|-------|
| `graphify update` | `WaitUntilCovered` | returns only once the request is covered |
| `watch` | `WaitUntilCovered` | submits the debounced batch as one request |
| git hooks | `ReturnWhenQueued`, then executes | the change set is durable before any work starts |

### Compatibility transition window

`graphify.watch._rebuild_code` is the helper hooks installed before this
refactor call. It remains importable with its old signature, but it now only
submits a code update and maps the terminal outcome to its legacy boolean;
`follow_symlinks`, `no_cluster`, `acquire_lock`, and `block_on_lock` are
accepted and inert. New hooks call `_background_code_update` instead.

Retirement criteria for `_rebuild_code`: the remaining lifecycle operations are
routed through `CorpusGraph` (issues #12–#14) and installed hooks have been
re-emitted by `graphify hook install`. Until then a hook from an older install
keeps working unchanged.

A code update publishes a *raw* graph generation: contributions, `graph.json`,
and the manifest. Community identity, labels, analysis, and `GRAPH_REPORT.md`
are republished by reclustering, which is a separate operation. Full extraction
publishes a raw generation too, for the same reason: clustering is reclustering's
work, so an extraction that produced one would be publishing a second operation's
output.

### Full extraction and its evidence sources

Full extraction owns the interpreting half of the lifecycle. A request names the
evidence sources beyond the corpus filesystem — a semantic provider, a PostgreSQL
schema, a Cargo workspace, Google Workspace shortcuts — and every one of them
must finish before anything is staged, so a generation never describes some
sources at one moment of the corpus and the rest at another.

Two seams are public because the systems behind them genuinely are external:
`SemanticProvider`, which interprets documents, and the source-system requests
that name a database or workspace. Discovery, caches, structural extraction, and
publication stay implementation details.

A source's contribution is replaced atomically: either the run produced that
source's complete evidence, or the prior contribution stands. A source whose
interpretation did not complete keeps its last complete evidence — marked stale —
or keeps having none, and stays pending either way; any fragment the attempt
produced belongs in the provider's cache, never in the ledger. Only a run that
interpreted every live semantic source may clear pending state.

Full extraction preserves the active corpus build policy unless the request
carries an explicit replacement or clearing.

The `graphify extract` CLI is still an unrerouted adapter: it prepares a
candidate and hands it over privately. Rerouting it is issue #12, and refusing an
incomplete extraction by default is issue #11.

## Extraction output schema

Every extractor returns:

```json
{
  "nodes": [
    {"id": "unique_string", "label": "human name", "source_file": "path", "source_location": "L42"}
  ],
  "edges": [
    {"source": "id_a", "target": "id_b", "relation": "calls|imports|uses|...", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS"}
  ]
}
```

`validate.py` enforces this schema before `build_graph()` consumes it.

## Confidence labels

| Label | Meaning |
|-------|---------|
| `EXTRACTED` | Relationship is explicitly stated in the source (e.g., an import statement, a direct call) |
| `INFERRED` | Relationship is a reasonable deduction (e.g., call-graph second pass, co-occurrence in context) |
| `AMBIGUOUS` | Relationship is uncertain; flagged for human review in GRAPH_REPORT.md |

## Adding a new language extractor

1. Add a `extract_<lang>(path: Path) -> dict` function in `extract.py` following the existing pattern (tree-sitter parse → walk nodes → collect `nodes` and `edges` → call-graph second pass for INFERRED `calls` edges).
2. Register the file suffix in `extract()` dispatch and `collect_files()`.
3. Add the suffix to `CODE_EXTENSIONS` in `detect.py` and `_WATCHED_EXTENSIONS` in `watch.py`.
4. Add the tree-sitter package to `pyproject.toml` dependencies.
5. Add a fixture file to `tests/fixtures/` and tests to `tests/test_languages.py`.

## Security

All external input passes through `graphify/security.py` before use:

- URLs → `validate_url()` (http/https only) + `_NoFileRedirectHandler` (blocks file:// redirects)
- Fetched content → `safe_fetch()` / `safe_fetch_text()` (size cap, timeout)
- Graph file paths → `validate_graph_path()` (must resolve inside `graphify-out/`)
- Node labels → `sanitize_label()` (strips control chars, caps 256 chars, HTML-escapes)

See `SECURITY.md` for the full threat model.

## Testing

One test file per module under `tests/`. Run with:

```bash
pytest tests/ -q
```

All tests are pure unit tests - no network calls, no file system side effects outside `tmp_path`.
