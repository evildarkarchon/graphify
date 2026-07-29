---
status: accepted
---

# Source contributions are authoritative

The deduplicated `graph.json` view cannot preserve custody when multiple Corpus sources contribute the same graph evidence. Each Graph generation will therefore persist a deterministic, portable ledger of Source contributions, and raw and clustered graphs will be topology-equivalent materialized views of that ledger; caches remain non-authoritative. Contributions are replaced source-atomically, while valid legacy graphs are adopted as provisional contributions until later operations establish complete modern custody.
