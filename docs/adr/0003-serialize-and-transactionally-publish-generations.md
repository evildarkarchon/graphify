---
status: accepted
---

# Serialize and transactionally publish Graph generations

Concurrent processes and interrupted writes must not expose an internally inconsistent Graph generation. Requests will join one durable per-Corpus queue, one executor will prepare and validate a candidate, and publication will use a recoverable journal and protected prior snapshot before promoting the candidate to the stable public paths; recovery rolls forward a verified candidate or restores the prior generation. Internal readers use short-lived validated snapshots, while derived renderings and global synchronization occur after the canonical local commit.
