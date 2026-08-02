# Data sources

DELPHI never commits source datasets. Fetch scripts, checksums, licence records, retrieval dates,
cohort rules, and small test fixtures are committed; raw data remains under ignored `data/`.

| Source | Licence status | Granularity | Current use |
|---|---|---|---|
| Deterministic synthetic gold set | MIT, generated locally | configurable | M1 deterministic and negative-control tests |
| Azure Functions 2019 | CC-BY 4.0; citation required | 1 minute | Downloaded 2026-08-03; 142,968,140 bytes; SHA-256 `aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89` |
| Azure LLM/LMM inference | CC-BY 4.0 | request-level | planned inference lane |
| Bitbrains GWA-T-12 | terms not currently verifiable; canonical host unavailable | 5 minutes | optional multi-resource trace |
| Alibaba cluster traces | no explicit licence found as of 2026-08-03; data will not be redistributed | event/hourly | gated, optional extension |

The Azure evaluation cohort is selected from day 1 using the 20 highest-volume functions plus
two seeded samples from each remaining invocation-volume decile (`seed=20260802`). Selection is
therefore reproducible without claiming that the busiest functions represent the full trace.
Functions absent on a later day receive 1,440 explicit `is_imputed=true`, `quality=0` points.

Retrieval date, byte size, SHA-256, exact citation, and cohort-selection seed are added when a
source is fetched or selected. An unverified licence is a stop condition, not permission to omit
the record.
