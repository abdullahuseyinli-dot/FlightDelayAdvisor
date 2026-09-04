# Corrected research data contract

This is a new input path. Existing date-aggregated feature caches must not be
renamed or assigned invented availability timestamps to satisfy it. Checksums
prove integrity, not the authenticity of historical timestamps; the source
review remains an explicit scientific responsibility.

## Observation histories

Run `python -m flightdelaybench.cutoff_dataset --targets TARGETS.parquet
--observations OBSERVATIONS.parquet --output-dir NEW_HISTORY_DIRECTORY` from the
FlightDelayAdvisor environment. Inputs are supplied separately; the command
does not download or fabricate missing operational evidence.

Targets require `sample_id`, `FlightDate`, `departure_time_utc`,
`cutoff_time_utc`, `Route`, `Reporting_Airline`, `Origin`, `Dest`, and
`ScheduledFlightId`. The cutoff is exactly scheduled departure minus 24 hours.

Each observation additionally requires `outcome` (`delay` or `cancel`), binary
`value`, `source_id`, `event_time_utc`, `source_published_at_utc`, and
`available_at_utc`. Duplicate sample/outcome revisions must be resolved with
evidence before construction. Delay observations are conditional on operation;
missing delay labels are not on-time observations. Event time is when the outcome
becomes observable, not necessarily when the flight was scheduled to end.

All event/publication/availability timestamps must be explicit and timezone-aware,
with event <= publication <= availability. Windows are availability-time windows
of 7, 28 and 90 days. Only earlier operating dates can contribute. A late report
enters when available, not retroactively on its operating date. Empty support
remains a missing rate and zero support. The builder exports 96 predictors across
global, route, airline, origin/destination inbound/outbound, and flight-ID views.

The implementation is tested on small batches. Full-census runtime, peak memory,
and partition-boundary equivalence still require a scale pilot before large builds.

## Matched experiment table and manifest

The assembled table contains the selected predictors and:

| Fields | Contract |
|---|---|
| `sample_id`, `FlightDate` | Unique scored identity and midnight operating date; this development runner accepts only 2024 |
| `departure_time_utc`, `cutoff_time_utc` | Exact T-24 relationship |
| `features_available_at_utc` | Audited maximum availability of the selected inputs, no later than cutoff |
| `cancel_label_available_at_utc`, `delay_label_available_at_utc` | Label availability used to exclude labels not yet known when fitting or stopping |
| `Cancelled`, `ArrDel15` | Binary observed task labels; unavailable arrival labels remain missing |
| `delay_label_observed`, `joint_label_observed`, `disruption_state` | Consistent hurdle semantics: 0 on time, 1 delayed, 2 cancelled |

A JSON dataset manifest has these required fields:

- `schema_version: 1`, `status: PASS_CUTOFF_DATASET`, `target_years: [2024]`.
- `evidence_class: REVIEWED_OBSERVATION_TIMESTAMPS`. Assumption tracks and
  synthetic fixtures cannot enter the research CLI as verified data.
- `source_evidence`: unique `source_id`, file `path`, `bytes`, `sha256`, and
  `timestamp_basis: observed_event_publication_and_availability` for every source.
- `features_by_context`: baseline, induced, and boundary predictor lists, with
  each retaining the preceding predictor set. These are column ablations on the
  same row cohort, not independently sampled files. Legacy recent, graph and
  rotation predictors are rejected; regenerated state needs a reviewed contract.
- `feature_evidence`: every selected feature maps to its source IDs.
- `label_evidence`: cancellation and delay availability each map to source IDs.
- `partitions`: explicit 2024 Parquet paths, byte sizes, hashes and row counts.
  Complete FlightDate min/max statistics are required before outcome access.
- `availability_audits`: paths, byte sizes and hashes of reviewed audits with
  `status: PASS_CUTOFF_AVAILABILITY_AUDIT`, `post_cutoff_values: 0`, the complete
  `features` list, `source_timestamps_synthesized: false`, and `partition_sha256`
  matching the exact dataset partitions, plus `label_availability_validated: true`.
  The audit must substantiate feature
  lineage and label timestamps, including any historical source revision policy.
- `manifest_sha256`: canonical JSON SHA-256 of all other manifest fields.

These status values describe requirements, not existing passed evidence.
No such real-data manifest is currently available. A self-authored PASS string
without source evidence is not an acceptable way to satisfy the contract.

## Execution and scope

`python -m flightdelaybench.cutoff_experiments --data-root DATA_DIRECTORY
--corrected-manifest DATASET_MANIFEST.json --output NEW_PREFLIGHT.json
--run-dir NEW_TRIAL_DIRECTORY` validates inputs and then runs the initial grid.
Without `--run-dir`, it records preflight only. Outputs are create-only. The
20-GiB reserve is a conservative campaign planning allocation, not a measured
minimum for every small trial.

The initial grid has 14 trials per seasonal fold: nested 125k/250k/full monthly
CatBoost hurdle samples across three contexts, and direct/hurdle model controls
for CatBoost, LightGBM and TabM on the boundary inputs. CatBoost categorical
interaction settings and sampling seeds are held fixed for context comparisons.
Training labels are censored at the first stopping cutoff; stopping labels are
censored at the first scoring cutoff. TabM now accepts the actual selected
feature matrix and uses training-only imputation, quantiles and categories.

Retuning, finalist repeats, incumbent stacking, independent confirmation, and
the conditional new architecture are later gates, not implemented research
results. The blend selector includes a literal zero-change incumbent option;
using it still requires honest forward incumbent predictions. The date and
seven-day-block comparison helpers do not by themselves select a winner.

Source-only tests and inference smoke tests never authorize release promotion.
