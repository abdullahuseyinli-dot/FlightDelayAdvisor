# Census continuity and network-boundary audit

Status: passed create-only audit; results describe coverage and do not upgrade the
retrospective BTS schedule to an issue-time operational feed.

## Why this audit exists

The legacy application dataset is an approximately month-balanced row sample. Such a
sample is suitable for some flight-level model development, but it cannot be used to
reconstruct airport queues, schedule banks, or interairport propagation: unselected
neighbouring flights would look as if they did not exist.

FLARE-24 and CC-RTH therefore use a separate census track. Schedule and graph context
is materialized from every retained row first; only the supervised loss rows are
subsampled later for model fitting. The selected source columns are an intentional
T-24 projection, not random columns: actual movement times, realised outcomes,
target-year tail identifiers, and other future information are excluded by design.

## What was tested

`flightdelaybench-audit-census-context` performs two independent checks:

1. It reopens and hashes all 96 normalized monthly partitions from 2018 through 2025,
   aggregates every retained flight by date and airport endpoint, and compares the
   observed dates with the complete calendar.
2. It reopens the 24 raw BTS archives for 2024-2025 using only `FlightDate`, `Origin`,
   and `Dest`. Every raw row is classified into the induced top-100 network, a
   top-100-to-outside boundary, or the outside network. The induced count must exactly
   reproduce each normalized census partition.

The cohort rule is strict: both origin and destination must belong to the frozen set
of 100 airports. Thus “census” means the complete induced top-100 subgraph, not every
US domestic flight.

## Results

| Check | Result |
|---|---:|
| Census period | 2018-01-01 through 2025-12-31 |
| Monthly partitions | 96 of 96 |
| Expected / observed calendar dates | 2,922 / 2,922 |
| Missing calendar dates | 0 |
| Retained rows | 43,815,581 |
| Expected airport-days | 292,200 |
| Airport-days with no inbound or outbound retained flight | 141 |
| Raw 2024-2025 BTS rows | 14,080,680 |
| Both endpoints in top 100 | 11,759,279 |
| Exactly one endpoint in top 100 | 2,207,091 |
| Neither endpoint in top 100 | 114,310 |
| Boundary share of all flights touching the top 100 | 15.803% |

The induced counts match the CC-RTH graph exactly in every audited month. All 141
zero-activity airport-days belong to HPN in 2020; one additional HPN day has no
inbound retained flight. These are disclosed schedule observations. Without a
separate provider-completeness signal, the audit does not relabel a zero-service day
as corrupt or impute flights.

Boundary exposure is not uniform. In 2024-2025, the largest boundary endpoint shares
among the frozen airports include ANC (32.12%), DFW (28.48%), DEN (20.85%), MSP
(19.04%), CLT (18.54%), and ORD (18.46%). Consequently, the present CC-RTH features
under-count total load at some target airports even though their induced-cohort rows
are temporally complete.

## Scientific interpretation

The audit resolves two different concerns:

- **Random-row context loss:** not present in the reported CC-RTH experiment. All
  retained schedule rows contribute to demand and graph features before the 125,000
  rows-per-month training sample is drawn.
- **Network-boundary context loss:** present and material. Flights between a target
  airport and a smaller airport do not contribute to the current resource state.

The boundary does not invalidate same-cohort comparisons because FLARE-24 and every
CC-RTH candidate score the same flights under the same information rule. It does
limit mechanism and generalization claims, and it may attenuate airport-load and
rotation signals. The current results must therefore be described as an induced
top-100-network benchmark.

## Professional extension protocol

The next version should preserve the scored top-100 cohort while enlarging only the
context graph. It should be treated as a new preregistered experiment rather than a
silent replacement of the released evidence.

1. Normalize all raw flights that touch a target airport as context-only rows. Do not
   read their outcomes when constructing target-day state.
2. Refit prior-year airport/hour/season frontiers using the same expanded schedule
   boundary. Mixing an induced-cohort frontier with nationwide demand is invalid.
3. Include adjacent-day buffers across monthly boundaries and purge the graph radius
   at train/calibration/test boundaries. Keep evaluation split by whole operational
   date.
4. Score the unchanged target cohort and report an induced-context versus
   boundary-complete-context ablation. Do not compare unmatched rows.
5. Run deterministic context-thinning controls at 10%, 25%, and 50%, with repeated
   seeds. This estimates the direction and magnitude of the bias that the old random
   sample would have introduced.
6. Add rolling-origin tests, leave-airport/leave-metro-out tests, and route-disjoint
   tests. Date-cluster intervals remain primary; spatial holdouts answer a different
   generalization question.
7. Publish a context-completeness index per airport-date-hour and stratify effects by
   boundary exposure. A larger gain at high-exposure airports would support—but not
   prove—the missing-context mechanism.
8. Add authentic, issue-time-vintaged runway configuration, airport acceptance and
   departure rates, gate occupancy, and traffic-management constraints only when
   their timestamps and availability can be audited. Missing feeds remain missing.

Because the 2025 outcomes have already informed method development, any newly tested
2025 result is exploratory. The locked, unopened 2026 window is the appropriate
confirmation target after the expanded-context method is frozen.

## Reproduction

```bash
flightdelaybench-audit-census-context \
  --census-manifest manifests/census_top100_2018_2025_v2.json \
  --raw-manifest manifests/raw_bts_census_2024.json \
  --raw-manifest manifests/raw_bts_2025.json \
  --boundary-years 2024 2025 \
  --output reports/validation/census_top100_context_completeness_REPLICATION.json
```

The released evidence is
`reports/validation/census_top100_context_completeness_v1.json`, with file SHA-256
`da19af74dde21f212d83a54f0db807fbcb5bd15c72e10fbf4c21c79a8634ef96` and internal
self-hash `346bdbd7a1d8ae685b15caa1a4b90322c33a253129672ddbeda2fea42f7c5c84`.
