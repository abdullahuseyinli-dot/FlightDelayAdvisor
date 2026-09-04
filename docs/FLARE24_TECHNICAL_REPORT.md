# FLARE-24 technical report

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

## A point-in-time benchmark for flight-level disruption probabilities

Release-candidate report, version 0.1.0-rc.1

Evidence state: frozen 2024 selection and full-year 2025 retrospective audit
Confirmation state: January-June 2026 protocol locked; outcomes unopened

## Abstract

Flight-delay studies are difficult to compare when they mix prediction horizons,
retrospective variables, sampled populations, binary endpoints, and random splits.
FLARE-24 addresses that problem with a flight-level, three-state benchmark whose
cutoff is scheduled departure minus 24 hours and whose primary evidence is evaluated
on a forward calendar year. The framework combines a rich schedule and closed-left
history baseline with forecast-vintage weather, aviation transformations, corridor
proxies, and schedule-only probabilistic rotation structure. It also tests convex
model blending and uncertainty-weighted alignment to independent aggregate forecasts.

All choices were made with 2024 data and written to a method lock before the full
2025 audit. On 5,814,817 flights with a resolved joint state, the selected structural-
rotation model reduced joint log loss from 0.555476 to 0.539225 and multiclass Brier
score from 0.337528 to 0.327076. The paired 365-date differences were -0.016251 (95%
interval -0.018948 to -0.013951) and -0.010452 (-0.011909 to -0.009189), relative
reductions of 2.93% and 3.10%. Joint log loss improved in every month.

The ablation is more informative than the headline. Fixed-vintage weather supplies
most of the gain. Structural rotation improves conditional-delay log loss and joint
Brier score beyond weather, but its incremental joint-log-loss interval crosses zero
and cancellation log loss worsens. Propagated predecessor risk is harmful, the
ensemble selects one model, and every nonzero reconciliation strength loses during
2024 selection. FLARE-24 is therefore presented as a strong leakage-audited benchmark
and a bounded applied-method contribution, not as universal state of the art.

## 1. Research question

At a flight-specific 24-hour cutoff, how much predictive information is added by:

1. archived weather that was actually issued before the cutoff;
2. aviation-aware transformations and route-corridor proxies;
3. uncertain schedule-only predecessor structure when target-year aircraft identity
   is unavailable;
4. propagated upstream disruption risk;
5. convex blending of nested models; and
6. soft alignment to independently forecast aviation margins?

The study evaluates probability quality, not a thresholded operational decision.
Primary measures are joint categorical log loss and multiclass Brier score.

## 2. Design and information boundary

For flight `i`, the prediction time is:

```text
cutoff_i = scheduled_departure_utc_i - 24 hours
```

The target state is on time, arrival delay of at least 15 minutes, or cancellation.
Cancellation is modeled on all scheduled rows with an observed binary cancellation
label. Delay is modeled only on operated, non-diverted rows with an observed
`ArrDel15` label. Rows without a resolved joint state are not imputed as on time.

The benchmark forbids target outcomes, realised weather, target-year tail number,
actual movement times, delay causes, and any current/future-label aggregate. Historical
rates use closed-left date windows. Open-Meteo Previous Runs values retain issue and
valid time separately and must satisfy the flight-specific cutoff. BTS schedule fields
are explicitly treated as a retrospective proxy, not as proof of a live D-24 schedule
snapshot.

## 3. Data

The population is the frozen union of 100 airports declared from the pre-existing
2024 cohort and rematerialized from official US DOT BTS Reporting Carrier On-Time
Performance files.

| Asset | 2024 rows | 2025 rows | Total |
|---|---:|---:|---:|
| Flight census | 5,929,613 | 5,829,666 | 11,759,279 |
| Weather airport-hours | 1,756,614 | 1,751,814 | 3,508,428 |

The weather, aviation/corridor, and rotation tables align to every census row. The
2025 joint endpoint has 5,814,817 resolved rows; conditional delay has 5,732,449.
Weather covariates are available on 5,814,365 joint-scored rows.

All raw and derived artifacts are create-only and manifest-bound by SHA-256, row
counts, schemas, projected columns, and outcome-access declarations. Failed and
superseded runs remain visible.

## 4. Candidate models

Four nested CatBoost hurdle candidates use identical row cohorts:

| Candidate | Additional information beyond rich baseline |
|---|---|
| Baseline | none |
| Weather | 76 usable forecast-vintage weather/aviation/corridor features |
| Structural rotation | weather plus 7 schedule-only latent-connection features |
| Rotation risk | structural rotation plus propagated closed-left predecessor risk |

The feature registry declares 108 FLARE fields. Twenty-three visibility, ceiling,
reflectivity, and freezing-level derivatives have zero coverage in the selected
archive and are excluded rather than fabricated or silently imputed.

Separate CatBoost models estimate cancellation `c_i` and delay given operation `d_i`.
The joint probability is:

```text
q_i = ((1 - c_i)(1 - d_i), (1 - c_i)d_i, c_i)
```

This prevents the impossible overlap created by treating binary delay and
cancellation probabilities as independent states.

## 5. Forecast-vintage and aviation features

For every flight, location, valid time, and variable, the selector uses the freshest
nonmissing archived forecast issued no later than the cutoff. Nominal day-2 values
provide a safe fallback and enable forecast-revision features when both snapshots
exist. Inputs cover origin departure, destination arrival, an origin time window, and
25/50/75-percent great-circle corridor proxies.

A dated FAA NASR snapshot supplies eligible runway headings. Wind is summarized over
those headings as a wind-optimal envelope; the benchmark does not claim to know the
active runway. Other transformations include endpoint crosswind/headwind, gust,
convection, icing-environment, snowfall, pressure/temperature changes, forecast age,
coverage, and weather-by-scheduled-bank-load interactions.

## 6. Capacitated latent rotations

The rotation kernel learns connection plausibility from prior-year examples with tail
supervision, but target-year tail identity is never read. Candidate predecessors must
match arrival/departure airport and carrier and have a 20-to-720-minute scheduled turn.
At most 12 candidates are retained. Nonnegative edge scores are converted to
probabilities under a predecessor-capacity constraint, with an explicit no-observed-
predecessor state.

The resulting features describe candidate count, connection mass, largest edge,
entropy, expected turn, tight-connection mass, and competition. They are probabilistic
schedule hypotheses, not recovered aircraft assignments.

## 7. Selection, calibration, ensemble, and alignment

January-August 2024 supplies the training sample, September controls early stopping,
and each model is refit through September at its fixed selected iteration count.
October-December is used in expanding forward folds for calibration and model-level
selection. No 2025 outcome enters these choices.

The selected structural model used 397 delay iterations and 562 cancellation
iterations. Identity calibration was selected for all eight candidate/task pairs. A
0.05-grid nonnegative ensemble selected weight 1.0 on structural rotation.

Independent origin-hour, destination-hour, carrier-day, and route-day state rates are
estimated by a recency-weighted hierarchical Dirichlet-multinomial model. Soft
reconciliation minimizes flight-level KL movement plus variance-weighted aggregate
deviation. An exact diagonal preconditioning of the dual coordinates reduced the
solver from an unscaled 1,000-iteration nonconvergence to 30 iterations in 0.278 s on
the recorded selection problem. However, every nonzero variance-strength candidate
worsened 2024 selection log loss, so identity was frozen. The solver result is a
numerical improvement, not a held-out predictive claim.

## 8. Evaluation protocol

The full 2025 census was scored without model or calibrator refitting. Primary
differences are paired on identical flights. Uncertainty is computed with 2,000
bootstrap resamples of all 365 flight dates as clusters, preserving within-day
dependence. Monthly scores, binary component metrics, calibration, severity,
missingness, and major-origin results are secondary diagnostics.

Because earlier repository work had accessed part of 2025, this is called a frozen-
method retrospective audit rather than blind confirmation. A separate January-June
2026 analysis and success rule have been locked without opening those outcomes.

## 9. Main results

| Candidate | Joint log loss | Multiclass Brier | Relative log-loss reduction vs baseline |
|---|---:|---:|---:|
| Baseline | 0.555476 | 0.337528 | - |
| Weather | 0.539607 | 0.327864 | 2.86% |
| Structural rotation | **0.539225** | **0.327076** | **2.93%** |
| Rotation risk | 0.539817 | 0.327945 | 2.82% |

The selected structural candidate improves baseline joint log loss in all 12 months.
Its conditional-delay log loss, Brier, AUROC, and average precision are 0.478755,
0.154754, 0.717102, and 0.432120. Cancellation values are 0.067108, 0.013565,
0.789121, and 0.084899.

The selected-minus-baseline difference grows with weather-severity stratum: joint log
loss differs by -0.006289 in Q1, -0.008767 in Q2, -0.015752 in Q3, and -0.037102 in
Q4. These strata are descriptive and associational.

## 10. Nested ablations and negative evidence

| Increment | Joint log-loss difference (95% interval) | Joint Brier difference (95% interval) |
|---|---:|---:|
| Weather - baseline | -0.015869 (-0.018587, -0.013663) | -0.009664 (interval excludes 0) |
| Structural rotation - weather | -0.000382 (-0.000858, +0.000144) | -0.000788 (-0.001004, -0.000588) |
| Rotation risk - structural rotation | +0.000591 (+0.000099, +0.001094) | +0.000869 (+0.000653, +0.001106) |
| Ensemble - structural rotation | 0 | 0 |
| Reconciled - ensemble | 0 | 0 |

Structural rotation improves delay-given-operation log loss by -0.001228 relative to
weather, while worsening cancellation log loss by +0.000824. Thus the defensible
interpretation is not that every FLARE component succeeds. The benchmark isolates a
large weather contribution, a smaller outcome-dependent structural effect, and clear
negative evidence for propagated risk and aggregate alignment.

## 11. Transparent recovery of the audit report

The original audit wrote all 12 prediction and 12 aggregate partitions before report
assembly rejected float32 probabilities with a maximum simplex error of
`4.470348e-08` under a strict `1e-8` tolerance. The recovery module verified every
persisted hash, required all errors to be at most `1e-6`, normalized rows in float64,
and reran evaluation only. The largest probability movement was about `3.33e-08`.
There was no refit, reprediction, reselection, or outcome-dependent change. The failed
run record and recovered report are both retained.

## 12. Contribution and novelty boundary

Nearby literature exists for graph delay modeling, aircraft-rotation features,
tail-assignment optimization, KL-regularized forecast reconciliation, and hybrid
boosting/temporal models. FLARE-24 does not claim to invent those broad categories.
Its supported contribution is the audited combination of:

- a flight-specific D-24 issue-time boundary;
- fixed-vintage weather and aviation/corridor transforms;
- schedule-only capacitated rotation hypotheses without target-year tail identity;
- coherent three-state hurdle probabilities;
- same-cohort, full-year, multi-million-flight proper-score evaluation; and
- selection rules that allow inventive components to fail and remain documented.

The exact combination was not found in the focused review, but formal systematic
database and citation-chain searches remain future scholarly work. No universal
novelty or state-of-the-art claim is made.

## 13. Limitations

- The airport restriction excludes the long tail of the US network.
- Retrospective BTS schedule fields are not a validated advance schedule feed.
- Archived weather does not validate live latency, uptime, or future availability.
- Corridor values are airport-node proxies, not a gridded route forecast.
- The FAA snapshot does not identify active runways.
- Latent connections are not observed target-year aircraft rotations.
- The 2025 audit is retrospective, not blind confirmation.
- Origin/carrier analyses are not demographic fairness studies.
- The benchmark does not establish causal benefit, operational safety, or thresholded
  decision value.

## 14. Evidence map

| Evidence | SHA-256 of file |
|---|---|
| 2024 selection report | `13fef0024e7a08139786ccdfc150a0c3db61208558d31be2a673f960b701e089` |
| Pre-audit method lock | `bac1cf84e9b58be00a4e5ccf511071b4b16697e84f36a0a62b1d4ee4906b8ecc` |
| Recovered 2025 audit | `8a4e88070d1a55cb30d3292629c1615057a92c661c2998df9045fc601d204171` |
| Nested paired ablation | `d6bc3cb00c226d1dfebb79c0a80dd8b29b91106c362090bda27a9e386fc88a9c` |
| Publication bundle v2 | `04e7ed7f6770c34c41af67549323c1e5c38a420643f7c96635f059d6408ec896` |
| Future confirmation lock | `daecde5d7c6690d21012031755dba1fbf72edd2fda6e8d395c2f1c42e252d037` |

The detailed machine-readable results are in
`reports/experiments/flare24_2025_audit_v1_recovered.json` and
`reports/experiments/flare24_2025_nested_ablation_v1.json`. The frozen method is in
`configs/flare24_v1.toml`; the data and model limitations are expanded in the data and
model cards.

## 15. Selected source boundary

The data and methodological boundary relies on the official
[BTS Reporting Carrier On-Time Performance table](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=%5D&gnoyr_VQ=FGJ),
[Open-Meteo Previous Runs documentation](https://open-meteo.com/en/docs/previous-runs-api),
and [FAA NASR archive](https://aeronav.faa.gov/aero_data/28DaySub/). The literature
map records the focused graph, tabular, rotation, and reconciliation sources and the
claim boundary derived from them.
