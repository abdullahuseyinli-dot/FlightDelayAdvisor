# Model card

Status: historical models and their scores are preserved, but the previous release
candidate is withdrawn following the cutoff audit. No corrected real-data model
performance has been established. This card records the parent model and its later
research extensions, not a production deployment recommendation.

## Model lineage

| Generation | Role | Current interpretation |
|---|---|---|
| Legacy app | Calibrated CatBoost delay and LightGBM cancellation | Separate sampled historical demo |
| FLARE-24 | Weather/aviation and structural-rotation CatBoost hurdle | Parent retrospective proxy model, described below |
| CC-RTH | Resource features, task factorization and cancellation meta-stack | Best historical joint log loss on the latest matched cohort |
| BC-POT-R | Boundary context and Q4-selected gated ensemble | Tiny accuracy/Brier gain, worse log loss than the meta-stack |
| Cutoff correction | Matched CatBoost, LightGBM and TabM direct/hurdle infrastructure | Implemented and fixture-tested; no corrected fitted model yet |

Use [current results](CURRENT_RESULTS.md) for comparable scores and
[artifacts](ARTIFACTS.md) for exact run identities. None of these results support
a universal state-of-the-art claim.

## Model summary

FLARE-24 estimates a coherent probability distribution over three mutually exclusive
states for each scheduled flight:

1. on time;
2. arrival delay of at least 15 minutes; or
3. cancellation.

The selected parent model is the `rotation_structural` candidate. It uses two CatBoost
classifiers in a hurdle construction: one predicts cancellation on all scheduled
cohort rows and one predicts delay conditional on operation. Their probabilities are
combined as `P(on_time)=(1-c)(1-d)`, `P(delayed)=(1-c)d`, and
`P(cancelled)=c`, where `c` is cancellation probability and `d` is delay
conditional on operation. Semantic endpoint checks are necessary in addition to
checking that rows sum to one. All selected parent calibration
families are identity. The selected convex ensemble places weight 1.0 on the
structural candidate, and the selected aggregate-alignment action is identity.

The name FLARE-24 describes the evaluated framework, including ensemble and aggregate
reconciliation hypotheses. It does not imply that a nontrivial blend or reconciliation
adjustment survived selection.

## Intended use

The model is intended for research, benchmarking, ablation studies, and educational
analysis of probabilistic US domestic flight disruption at a flight-specific 24-hour
horizon. Suitable uses include:

- comparing explicit prediction-time information sets;
- studying temporal shift with proper probability scores;
- evaluating fixed-vintage weather and schedule-only network hypotheses; and
- reproducing the frozen 2024-to-2025 benchmark protocol.

Outputs are risk estimates, not operational guarantees or instructions. A downstream
interface must show the forecast horizon, model version, coverage/missingness state,
and the fact that the evidence is retrospective.

## Population and targets

The cohort is the frozen union of 100 airports derived from the pre-existing 2024 BTS
cohort. The materialized 2024-2025 census has 11,759,279 schedule rows. The 2025 audit
contains 5,829,666 scheduled rows, of which 5,814,817 have a resolved joint outcome;
5,732,449 operated, non-diverted rows have an observed conditional-delay label.

Cancellation is scored on all scheduled rows with a binary label. A later diversion
is a non-cancellation for that task, but diverted or otherwise unresolved rows are not
silently recoded as on time for delay or joint scoring.

## Prediction-time inputs

The cutoff for flight `i` is scheduled departure in UTC minus 24 hours. Inputs include:

- retrospective schedule fields and calendar-day 7/28/90-day histories whose
  flight-specific availability needs correction;
- target-day schedule density and closed-left graph-pressure summaries;
- Open-Meteo GFS Previous Runs weather filtered by the implemented nominal issue
  contract, not proof of historical consumer receipt;
- FAA NASR runway-heading envelopes, without an active-runway claim;
- great-circle corridor proxies from nearby forecast nodes; and
- probabilistic predecessor structure inferred from schedule compatibility without
  reading target-year tail numbers.

Target outcomes, realised weather, target-year tail identity, actual movement times,
delay causes, and future/current-row label aggregates are forbidden predictors. The
feature registry contains 108 candidate FLARE fields; 23 zero-coverage fields were
excluded rather than represented as observed. The selected structural model used 83
usable FLARE extras in addition to the rich baseline.

## Training and selection

- January-August 2024: early-stopping training sample, 125,000 rows per month.
- September 2024: early-stopping validation, capped at 250,000 rows.
- January-September 2024: refit at the fixed selected iteration count.
- October-December 2024: expanding forward-only calibration, ensemble, and
  reconciliation-strength selection.
- Full 2025: frozen-method retrospective audit, with no model, calibrator, feature,
  weight, or threshold refit.

The delay models used 1,228,884 eligible refit rows; cancellation models used
1,250,000. The selected structural delay and cancellation fits used 397 and 562
iterations respectively. Four nested candidates were retained: baseline, weather,
structural rotation, and rotation with propagated predecessor risk.

## Historical parent performance

The following numbers precede the timing correction. They are not new corrected
forecast measurements and use a different cohort from the latest boundary table.

On 5,814,817 jointly scored 2025 flights, the selected structural model achieved:

| Measure | Baseline | Selected | Paired difference (selected - baseline) |
|---|---:|---:|---:|
| Joint log loss | 0.555476 | 0.539225 | -0.016251 (95% interval -0.018948 to -0.013951) |
| Multiclass Brier | 0.337528 | 0.327076 | -0.010452 (-0.011909 to -0.009189) |

These correspond to relative reductions of 2.93% and 3.10%. The joint log-loss
difference favored the selected model in all 12 months. Paired uncertainty uses 2,000
bootstrap resamples of all 365 flight dates as clusters.

Secondary selected-model results were:

| Endpoint | Log loss | Brier | AUROC | Average precision |
|---|---:|---:|---:|---:|
| Delay given operation | 0.478755 | 0.154754 | 0.717102 | 0.432120 |
| Cancellation | 0.067108 | 0.013565 | 0.789121 | 0.084899 |

## Ablation interpretation

Most of the gain comes from fixed-vintage weather and aviation transforms. Weather
versus baseline improved joint log loss by -0.015869 (95% interval -0.018587 to
-0.013663). Structural rotation versus weather improved joint Brier by -0.000788
(-0.001004 to -0.000588) and conditional-delay log loss by -0.001228 (-0.001518 to
-0.000958), but its incremental joint-log-loss interval crosses zero (-0.000858 to
+0.000144), and cancellation log loss is worse by +0.000824 (+0.000355 to +0.001345).

Propagated predecessor risk is worse than structural rotation on both joint proper
scores. The selected ensemble and reconciliation are exact identity operations. These
negative findings are part of the model card because they delimit what the evidence
supports.

## Calibration and reliability

Identity calibration was selected for all eight component models in the 2024 forward
folds. The release reports calibration intercept/slope, equal-mass ECE, prevalence,
top-decile lift/capture, monthly proper scores, major-origin diagnostics,
weather-severity strata, and missing-weather strata. These are diagnostic rather than
new selection criteria.

Weather covariates were present for 5,814,365 of 5,814,817 joint-scored 2025 rows.
Only 452 rows were missing both endpoint covariates, so their separate point estimate
is too small to support a benefit claim.

## Audit recovery disclosure

The first 2025 finalizer wrote all 12 monthly prediction and aggregate partitions,
then rejected stored float32 joint probabilities under an unnecessarily strict
`1e-8` row-sum tolerance. The maximum row-sum error was `4.470348e-08`; no row exceeded
the predeclared recovery bound of `1e-6`.

The recovery path reopened the immutable predictions, divided each row by its sum in
float64, and reran report assembly. The maximum probability adjustment was about
`3.33e-08`. It performed no refit, reprediction, reselection, or method change. The
failed run, diagnostic, recovery rule, original partitions, and recovered report all
remain separately visible.

## Limitations and prohibited interpretations

FLARE-24 does not establish:

- causal passenger benefit or a causal weather effect;
- airline or flight safety;
- production feed latency, uptime, or D-24 schedule snapshot fidelity;
- active runway state or actual target-year aircraft identity;
- demographic fairness;
- generalization beyond the observed top-100-airport US reporting cohort; or
- universal state-of-the-art performance.

The BTS schedule is a retrospective proxy. Weather uses archived model output, and
corridor conditions use nearby airport nodes rather than a gridded trajectory. The
FAA snapshot is static. Abrupt regime shifts can alter prevalence and calibration.

## Reproducibility and governance

The canonical protocol is `configs/flare24_v1.toml`. The frozen selection report,
method lock, monthly predictions, model and calibrator hashes, recovered audit,
nested ablations, and publication bundle have historical integrity validations.
Those checks do not override the cutoff withdrawal. Large data and
model artifacts remain external but are bound by path-independent SHA-256 values,
row counts, schemas, and manifests.

The historical January-June 2026 confirmation analysis was frozen in
`manifests/confirmation_lock_v1.json`; no 2026 outcome was acquired or read for this
release candidate. A cutoff-corrected method requires a new reviewed lock before
confirmation access; the old lock cannot validate changed inputs or models.
The later [source investigation](RESEARCH_CONTINUATION_20260905.md) recorded
incidental aggregate 2026 search exposure without opening the confirmation dataset.
The next lock must account for that incident rather than claim zero exposure.
Confirmation outcomes may not change the model, features,
calibration, ensemble, endpoints, bootstrap design, or success rule.
