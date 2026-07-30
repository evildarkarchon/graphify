---
status: accepted
---

# Model Semantic evidence freshness per source

A Code update must remain LLM-free without pretending that changed semantic sources were reinterpreted. Semantic evidence from a changed live source remains queryable as Stale semantic evidence and the source remains pending; evidence from deleted or newly excluded sources is removed, and only successful source-atomic interpretation during Full extraction clears pending state. Incomplete discovery refuses publication by default, with partial publication available only through explicit Full extraction authority.
