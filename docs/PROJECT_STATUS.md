# Project status

Status reviewed: 2026-09-04. Package version remains `0.1.0rc1`; no new release,
tag or Zenodo DOI is asserted.

## Current disposition

The previous local release candidate is **withdrawn pending cutoff correction
and new release validation**. The
[withdrawal record](../manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json)
binds the historical v3 ledger by hash. Old models, predictions, scores, failures
and distribution archives remain preserved.

| Workstream | State | What it establishes |
|---|---|---|
| FLARE-24 / CC-RTH / BC-POT-R experiments | Completed historical runs | Retrospective proxy scores and diagnostics |
| Census and boundary audits | Completed | Retained-date continuity and omitted boundary schedules |
| Cutoff/history and label-time validation | Implemented; small-fixture tests | Software behavior under explicit timestamp contracts |
| Matched 42-trial development grid | Implemented, not run on corrected real data | Experiment infrastructure only |
| Issued-weather archive pilot | 16 audited cases, no flight labels | Limited archive feasibility, not model improvement |
| Source/package checks | Recorded passes | Code and packaging integrity, not scientific acceptance |
| Corrected full-data model results | Not available | No corrected accuracy or breakthrough claim |
| Independent confirmation | Unopened | No 2026 outcome-based result |
| External publication | Not performed by this work | Metadata is a draft |

## What the audit changed

1. Prior-operating-day outcomes can occur or become available after a target
   flight's T−24 cutoff. Excluding the target date is insufficient. Derived
   history/graph/rotation risks require a reviewed availability contract and refits.
2. The old packaged inference smoke swapped cancellation and conditional-delay
   inputs. Its simplex check did not detect the semantic error. The smoke and
   semantic tests were corrected; main study callers already used the correct
   argument order, so this correction does not itself change historical scores.
3. The boundary study changed CatBoost `max_ctr_complexity` from 4 to 1 and did
   not offer the strongest meta-stack as an ensemble component. The study is not
   an isolated estimate of the value of small-airport data.

## Verification versus research

The presentation review's
[source check](../reports/validation/repository_presentation_source_checks_v1.json)
records 269 passed tests, one skipped legacy integration module, two deselected
fixed-output script tests, and passing compilation, lint, typing, lock, repository
and documentation checks. It verified 38 Markdown documents, 165 local links and
the displayed result tables at that revision. Later additions may change link
counts without changing the research results.

Its [fresh-wheel check](../reports/validation/repository_presentation_clean_wheel_v1.json)
installed the revised package in a new environment, verified isolated imports and
passed historical-model inference on 128 inputs. This is software validation, not
a new model evaluation.

The preceding correction-generation
[source check](../reports/validation/flare24_source_checks_v14_cutoff_correction.json)
records 263 passed tests, one skipped legacy integration module, two deselected
fixed-output script tests, and passing compilation, lint, typing, lock and repository
checks. It is a dated record, not a promise that every later checkout has that count.

The [clean-wheel check](../reports/validation/cutoff_clean_wheel_v1.json) installed
a wheel in a fresh environment and checked inference on 128 historical 2024
inputs. The [bundle audit](../reports/validation/cutoff_evidence_bundle_v5.json)
verified archive contents, including all 66 pilot files. These tests used existing
model inputs and do not establish corrected model quality. Subsequent presentation
checks are indexed in [artifacts](ARTIFACTS.md).

## Remaining gates

The [2024 schema audit](../reports/validation/cutoff_source_schema_2024_v1.json)
found actual event-clock fields in raw BTS files, but not in normalized tables.
Neither layer supplied the required historical source-publication, consumer-receipt
and label-availability ledger. Schema presence is not a value-completeness audit.
Reconstructing arrival times alone cannot establish when a forecasting system
could have known an outcome or which revision was available.

Before new performance claims:

1. Obtain reviewed availability evidence, or define a separate, explicitly labelled
   event-time/latency-assumption study. Do not manufacture receipt timestamps.
2. Pilot corrected feature construction at scale; record runtime and memory.
3. Refit matched baselines and context ablations with equal estimator settings.
4. Run sample-size, model-family, finalist-repeat and forward-stack comparisons.
5. Test genuinely new information only after archive and timing coverage is demonstrated.
6. Freeze the corrected method and confirm independently before stronger claims.

The existing preflight records storage and missing-data constraints; its 20-GiB
reserve is a planning allowance, not a measured minimum or a current free-space
report. No raw evidence should be deleted to satisfy it.
See the [sequential plan](CUTOFF_RESEARCH_PLAN.md) and [release checklist](RELEASE_CHECKLIST.md).
