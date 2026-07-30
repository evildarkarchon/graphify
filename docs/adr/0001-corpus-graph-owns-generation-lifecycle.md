---
status: accepted
---

# CorpusGraph owns the Graph generation lifecycle

Graph generation orchestration is currently distributed across CLI, watcher, and hook paths, making publication and state invariants difficult to preserve. `graphify.generation.CorpusGraph` will own the complete lifecycle through explicit Full extraction, Code update, and Reclustering operations; callers remain adapters that submit requests, observe stable lifecycle facts, and render closed terminal outcomes. This concentrates the production behavior behind one deep module while preserving existing entrypoints through temporary compatibility adapters.
