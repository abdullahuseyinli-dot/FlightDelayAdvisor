# Current results

[Documentation index](README.md) · [Benchmark card](BENCHMARK_CARD.md)

## Evidence status

All model scores on this page precede the cutoff correction. They are preserved
retrospective proxy measurements, not corrected T−24 performance or blind
confirmation. The timing audit did not recompute these scores. Software checks and
archive-coverage pilots are not predictive experiments.

## Latest matched cohort

Source: [BC-POT-R recovered study](../reports/experiments/flare24_boundary_pot_v6_recovered.json),
fields `primary_evaluation`, `primary_argmax_decision_metrics` and
`primary_class_prevalence`. The scored population is 5,754,266 jointly observed
top-100-to-top-100 flights from January 3 through December 29, 2025. Cancellation
scoring separately includes 5,768,943 scheduled rows; conditional-delay scoring
includes 5,672,210 rows. Paired intervals use 2,000 resamples of 361 flight dates.

| Method | Accuracy (%) | Joint log loss | Multiclass Brier |
|---|---:|---:|---:|
| Schedule/history baseline | 76.7493 | 0.555921 | 0.337727 |
| FLARE-24 structural rotation | 77.3289 | 0.539577 | 0.327218 |
| CC-RTH gated model | 77.3283 | 0.539559 | 0.327217 |
| Regularized cancellation meta-stack | 77.3308 | 0.537377 | 0.326911 |
| Boundary-complete model | 77.3673 | 0.538733 | 0.326536 |
| Counterfactual-residual model | 77.3478 | 0.538168 | 0.326613 |
| Global boundary ensemble | 77.3522 | 0.538418 | 0.326665 |
| Q4-selected boundary ensemble | 77.3512 | 0.538449 | 0.326679 |

These values are checked against the stored report by
`python tools/validate_documentation.py`. The table does not mix the legacy sampled
application, the full-calendar-year parent audit or later model-selection rules.

### Selected boundary ensemble versus the prior meta-stack

| Measure | Difference | Paired 95% date-cluster interval | Interpretation |
|---|---:|---:|---|
| Accuracy, percentage points | +0.0203 | +0.0060 to +0.0349 | Very small gain |
| Joint log loss | +0.001072 | +0.000447 to +0.001760 | Worse |
| Multiclass Brier | -0.000231 | -0.000449 to -0.000012 | Small improvement |

A five-point gain means 75% to 80%, not a 5% relative reduction in another score.
Both +5 and +10 percentage-point accuracy gates were not met. The selected boundary
method is 0.6019 percentage points above the schedule/history baseline, but only
0.0203 points above the much stronger prior meta-stack.

The boundary-complete model has the highest standard accuracy point estimate in
this table. It was not the Q4-selected final method, and choosing it on the 2025
table would be additional retrospective selection. A separately Q4-selected
class-bias diagnostic reached 77.4922%; it is not the standard argmax endpoint.

### Why accuracy is insufficient

Joint-state prevalence is 76.5222% on time, 22.0518% delayed and 1.4260% cancelled.
Always predicting on time would therefore achieve 76.5222% accuracy on this cohort;
this is a descriptive reference, not a fitted model.

For the selected boundary ensemble, class recall is 98.6041% on time, 8.5994%
delayed and 0.0548% cancelled. Balanced accuracy is 35.7528%, versus 35.8780%
for the meta-stack. These are argmax decisions: a model can assign informative
cancellation probabilities while almost never making cancellation its most likely
class. Proper scores, calibration and task-specific ranking remain necessary.

## What the airport experiments established

The census/context audit distinguished missing dates from a restricted network
boundary. It verified all 2,922 dates in the 2018–2025 retained census. A separate
raw 2024–2025 scan found 2,207,091 target-to-outside flights absent from the induced
top-100 network: 15.803% of flights touching target airports.

BC-POT-R subsequently added 3,216,969 boundary schedule rows across its broader
context build, with 255 extra airports. Those are context-only flights, not new
scored targets. The two counts refer to different coverage periods and must not
be interchanged. See the [data card](DATA_CARD.md).

The small-airport comparison is **confounded**: the boundary campaign also changed
CatBoost `max_ctr_complexity` from 4 to 1 for memory control. Its ensemble did not
include the strongest prior meta-stack as a selectable component. Same target IDs
therefore do not make this an isolated feature-only ablation. Neither a large gain
nor proof that small airports are useless follows from these runs.

Severe boundary route-pressure residuals identify a +1.634 percentage-point
disruption-prevalence difference (95% interval +1.224 to +2.050 points). This is an
association, not a causal effect or demonstrated predictive improvement. The
any-boundary-change contrast is saturated and its temporal interval is
non-estimable; that failure remains recorded.

## Earlier FLARE-24 parent audit: a different cohort

Source: [full-year parent audit](../reports/experiments/flare24_2025_audit_v1_recovered.json)
and [nested ablations](../reports/experiments/flare24_2025_nested_ablation_v1.json).
This cohort contains 5,829,666 scheduled rows and 5,814,817 observed joint states
over all 365 dates of 2025. Do not compare these absolute scores directly with
the trimmed-cohort table above.

| Candidate | Joint log loss | Multiclass Brier |
|---|---:|---:|
| Schedule/history/graph baseline | 0.555476 | 0.337528 |
| Weather/aviation | 0.539607 | 0.327864 |
| Structural rotation | 0.539225 | 0.327076 |
| Propagated predecessor risk | 0.539817 | 0.327945 |

Structural rotation versus baseline: log-loss difference -0.016251
(95% interval -0.018948 to -0.013951), Brier difference -0.010452
(-0.011909 to -0.009189), using 2,000 paired date-cluster resamples.
Weather explains most of the historical gain. The incremental rotation log-loss
interval versus weather crosses zero. These old timing contracts require
correction before a strict forecast claim.

## Negative findings and remaining questions

- Propagated predecessor risk did not beat structural rotation.
- Nontrivial aggregate reconciliation did not survive selection; a better numerical
  solver is not evidence of improved predictions.
- The direct full resource hypergraph worsened the joint proper scores. Task
  factorization and cancellation stacking were more useful in the historical study.
- Counterfactual boundary residuals received zero ensemble weight.
- Frontier tabular models did not establish a large improvement in the earlier
  tracks; those tracks are not a current-feature, matched-budget corrected comparison.
- The 16-case issued-TAF pilot measured limited archive coverage only. It used no
  flight labels and establishes no accuracy gain.
- Corrected full-data refits, matched capacity settings, sample-size scaling,
  finalist repeats and new-information ablations remain uncompleted.

The [historical results ledger](RESULTS.md), [BC-POT-R interpretation](BCPOTR_RESULTS.md)
and [CC-RTH report](CCRTH_TECHNICAL_REPORT.md) retain the full experimental detail.
The [correction plan](CUTOFF_RESEARCH_PLAN.md) governs the next research generation.
2025 is development-informed and cannot become untouched again. The 2026 outcomes
in the confirmation dataset remain unopened, but aggregate 2026 statistics appeared
in unsolicited search snippets during a later source investigation. The
[incident](../manifests/failures/handoff_continuation_20260905_incidental_search_exposure_v1.json)
was recorded and the values were not used. Do not claim zero exposure. No corrected
real-data model was fitted before the [continuation stopped](RESEARCH_CONTINUATION_20260905.md).
