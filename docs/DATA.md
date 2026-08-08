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

## Azure LLM inference traces 2024

| Field | Value |
|---|---|
| Source | `github.com/Azure/AzurePublicDataset`, release `dataset-llm-2024` |
| Licence | **CC-BY 4.0** (repo-root `LICENSE`, verified 2026-08-03) |
| Citation | Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, Esha Choukse. *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency*, HPCA 2025. |
| Retrieved | 2026-08-03 |
| Period | 2024-05-10 to 2024-05-19 |
| Schema | `TIMESTAMP` (absolute UTC), `ContextTokens`, `GeneratedTokens` |

| File | Size | Requests | SHA-256 |
|---|---:|---:|---|
| `AzureLLMInferenceTrace_code_1week.csv` | 660 MB | 16,803,695 | `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f` |
| `AzureLLMInferenceTrace_conv_1week.csv` | 1.1 GB | 27,303,999 | `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` |

Not redistributed; `scripts/` fetches from the release URL and `data/` is gitignored.
Timestamps are absolute UTC, so unlike the anonymized Azure Functions trace there is no
epoch assumption and day-of-week is a fact rather than a convention.
