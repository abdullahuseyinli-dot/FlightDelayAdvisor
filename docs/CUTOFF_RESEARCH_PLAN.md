# Cutoff correction and bounded research plan

Revision: 2026-09-04. All existing data, models, reports, failures and release
archives are retained. The v3 release claim
is withdrawn in `manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json`.
2025 remains development-informed retrospective evidence; it cannot be called
untouched again. No 2026 outcome may be opened by this workflow.

## Sequential gates

1. Correct endpoint semantics and introduce timestamp-based outcome histories.
   Each contributing observation must have a source identity, event timestamp,
   source-publication timestamp and consumer-availability timestamp. All must be
   no later than the target cutoff; only earlier operating dates contribute.
   Missing timestamp evidence is rejected rather than synthesized from FlightDate.
   Frozen calendar-day histories and graph/rotation risks derived from them are
   excluded from the new model feature path.
2. Refit schedule/forecast and corrected-history baselines. Quantify old-history
   exposure and preserve both old and new generations. Exact historical source
   availability cannot be inferred from the current normalized BTS tables.
3. Compare nested 125k, 250k and full monthly estimator samples, keeping all context
   flights. Compare induced and boundary-complete context with identical samples,
   seeds, categorical-interaction limits and tuning budgets. Use at least three
   seasonal forward development folds and repeat finalists across three seeds.
4. Challenge current-feature CatBoost with LightGBM and TabM, including direct
   three-class and hurdle tasks. Fit preprocessing on training only. Regenerate
   forward incumbent predictions before stacking; do not reuse a later-selected
   meta-stack in an earlier fold. Include the incumbent as an exact no-change option.
5. Pilot issued weather and preannounced restrictions with demonstrated archive
   coverage. Separate hindsight diagnostics from forecast features. A live SWIM
   feed is not evidence of historical access. Do not fill an expired TAF with a
   later issue. Only then consider uncertain-capacity/recovery temporal modeling.

## Metrics and continuation

Accuracy gains are absolute. A +0.005 screening milestone with no material proper-
score regression is a continuation criterion, not the requested +0.05/+0.10 result.
Report class prevalence/recall, accuracy, joint log loss, Brier, and paired date-
cluster intervals. Check multi-day block sensitivity for serial correlation.
The final breakthrough reference is the corrected same-cohort incumbent, not an old
model using differently timed inputs. Retain all negative trials and disclose the
development search. Freeze the selected method before independent confirmation.

## Current constraints

The recorded preflight found insufficient campaign storage and missing
availability evidence. Full feature builds, new model sets and predictions require
a measured storage budget; raw evidence and prior runs must remain preserved.
The normalized tables lack actual arrival and source-availability timestamps.
An availability ledger or an explicitly labelled timing-assumption sensitivity
track is necessary; a D-2 shift alone does not establish an operational guarantee.

Completion means measured results and validation for the applicable gates. Code or
synthetic tests alone do not count as completion of a research experiment. A missing
data or storage prerequisite must remain visible and must not produce a PASS result.

## Implemented work and remaining gates

The endpoint argument-order correction, independently checked hurdle semantics,
timestamp-history builder, hash-bound data loader, chronological experiment runner,
matched sample grid, current-matrix TabM preprocessing, incumbent no-change blend
option, and paired date/week comparisons are implemented and tested on small data.
Historical release promotion is blocked and public-facing metadata marks the v3
withdrawal. See `CUTOFF_DATA_CONTRACT.md` for inputs and commands.

All twelve raw/normalized 2024 schemas were inspected, covering 5,929,613 normalized
rows by metadata. The raw archives retain actual departure/arrival clock and delay
fields; none of these event fields survive in the normalized partitions. Neither
layer contains the required publication/consumer/label-availability ledger. Event
times can therefore be investigated from raw data, but recovering them alone does
not establish historical receipt or revision validity. The schema audit read no
flight outcome values and is not a value-completeness audit.

A 16-case issued-TAF archive pilot is completed and preserved, including raw source
text. It establishes limited archive feasibility, not a predictive improvement or
an operational-availability guarantee. See `CUTOFF_WEATHER_PILOT.md`. Restrictions,
flight-weighted forecast coverage, new-information model ablations, and the
conditional capacity/recovery architecture remain uncompleted gates.

CPU Torch 2.9.1, TabM 0.0.3, numerical embeddings and the build backend were
installed and tested in the recorded correction environment.
The full-campaign 20-GiB reserve is explicitly a planning reserve,
not a measured minimum for a small fit. No full corrected feature build, real-data
model grid, retuning, finalist repeats, or corrected accuracy result is claimed.
2026 outcomes remain unopened. Nothing has been pushed, tagged, or deposited.
