# Cutoff-correction implementation record — 2026-09-04

This is the dated correction-generation record. See [project status](PROJECT_STATUS.md)
for the current research account and [artifacts](ARTIFACTS.md) for later checks.

**Software corrections validated; the full research campaign is not complete.**
No new real-data model accuracy, breakthrough, or publication-ready result is
established. The historical v3 release candidate remains withdrawn. No push,
tag, Zenodo deposit, evidence cleanup, or 2026 outcome access was performed.

## Implemented and checked

- Correct cancellation/conditional-delay argument order, with independent class
  semantics checks rather than simplex checks alone.
- Timestamp-validated 7/28/90-day histories, strict earlier-operating-date rules,
  cutoff-safe joins, missing-support handling, and source-evidence requirements.
- A hash-bound corrected-data entry point; legacy recent/graph/rotation caches
  cannot enter the new path merely by being relabelled.
- A sequential 42-trial development grid: matched monthly samples and contexts,
  seasonal forward folds, CatBoost/LightGBM/TabM direct and hurdle models,
  training-only preprocessing, and correctly censored training/stopping labels.
- Same-ID paired date and seven-day-block comparisons, absolute accuracy units,
  and a blend selector that can leave the incumbent unchanged.
- Create-only artifacts, preserved failures, explicit withdrawal/publication gates,
  source checks, archive inspection and isolated installed-wheel inference.

These are implemented capabilities, not 42 completed real-data experiments.
The small end-to-end estimator tests used synthetic fixtures. Optional CPU Torch,
TabM, numerical embeddings and the build backend were installed successfully.

## Validation evidence

| Check | Final observed result | Evidence |
|---|---|---|
| Source suite | 263 passed; 1 legacy integration module skipped; 2 script tests deselected | `reports/validation/flare24_source_checks_v14_cutoff_correction.json` |
| Compilation, lint, typing, lock, repository integrity | Passed; release still explicitly withdrawn | Same report |
| Fresh environment and installed-wheel inference | Passed on 128 historical 2024 inputs | `reports/validation/cutoff_clean_wheel_v1.json` and its inference report |
| Corrected source archive | Passed; all 66 pilot files byte-verified; wheel identical to clean-tested bytes | `reports/validation/cutoff_evidence_bundle_v5.json` |
| 2024 raw/normalized schema audit | 12 months; 5,929,613 normalized rows by metadata; availability evidence absent | `reports/validation/cutoff_source_schema_2024_v1.json` |
| Issued-weather archive pilot | 16 station/time cases audited; no flight labels used | `data/external/cutoff_taf_pilot_v1/report.json` |

The skipped module needs the legacy app integration environment and materialized
Git LFS artifacts. The two excluded legacy script tests execute full evaluation
and plot generation against those artifacts; they were not run over preserved
historical reports. Three non-failing LightGBM API deprecation warnings remain.
The corrected wheel smoke uses old model inputs to check packaging and endpoint
semantics only, not to establish corrected predictive performance.

Build/backend failures, a source-bundle omission, and an overly broad scan into
third-party environment files were recorded and corrected. Prior attempts remain
preserved. The final bundle is under `dist/cutoff_validation_v5/`; it is a local
validation artifact, not an external publication release.

## What is still blocked or uncompleted

The raw BTS archives retain actual event-clock fields that the normalized tables
discarded. Neither layer supplies the historical source-publication, consumer
receipt and label-availability ledger required for the strict corrected track.
Recovering actual times can support diagnostics, but cannot establish when our
forecasting system could have known a label or which revision was available.

The recorded campaign preflight found insufficient free storage. It uses a
20-GiB planning reserve, not a measured minimum for a small fit or a current
free-space measurement.
No data was deleted to manufacture space. Full feature builds, real-data refits,
retuning, finalist repeats and honest forward incumbent stacking remain undone.

The TAF pilot found T+24 coverage for all sampled JFK/O'Hare cases but none for
the sampled Des Moines/Central Wisconsin cases under its 15-minute latency
assumption. This supports investigating longer-range forecast complements; it
is not population coverage or an accuracy result. Planned-restriction coverage,
new-information model gains and the conditional capacity/recovery architecture
have not passed their gates.

The practical next choice is either to obtain auditable historical availability
evidence for the strict track, or define a separately labelled event-time and
latency-assumption sensitivity track using recoverable raw fields. The latter
can be scientifically useful, but must not be described as verified historical
T-24 operational performance. See `CUTOFF_RESEARCH_PLAN.md` and
`CUTOFF_DATA_CONTRACT.md` for the remaining gates.
