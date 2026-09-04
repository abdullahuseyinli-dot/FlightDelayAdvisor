# Verified development and retrospective results

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

Last updated: 2026-09-04
Status: validated retrospective evidence; no 2026 confirmation outcome has been accessed

This document reports only materialized, self-hashed artifacts. Lower log loss and
Brier score are better; higher AUROC and average precision (AP) are better. AP must
not be compared across cohorts with different prevalence. Confidence intervals use
2,000 paired bootstrap resamples of flight dates, preserving within-day dependence.

## FLARE-24: full-census three-state benchmark

FLARE-24 was fitted and selected using 2024 only. January-August supplied the
early-stopping training sample, September selected each fixed iteration count, the
models were refit through September, and expanding Q4 folds selected calibration,
ensemble weights, and whether to apply aggregate alignment. The exact artifacts were
then frozen in `manifests/flare24_method_lock_v1.json` before the full 2025 audit.
No flight model, calibrator, ensemble weight, or reconciliation choice was changed in
2025.

The audit includes 5,829,666 scheduled rows. Joint proper scores use the 5,814,817
rows with an observed three-state outcome; delay uses 5,732,449 operated,
non-diverted rows with an observed `ArrDel15`; cancellation uses all 5,829,666 rows.

| Candidate | Joint log loss | Multiclass Brier | Delay log loss | Cancellation log loss |
|---|---:|---:|---:|---:|
| Schedule/history/graph baseline | 0.555476 | 0.337528 | 0.492107 | 0.070186 |
| + forecast-vintage weather/aviation | 0.539607 | 0.327864 | 0.479983 | **0.066284** |
| **+ latent rotation structure** | **0.539225** | **0.327076** | **0.478755** | 0.067108 |
| + propagated predecessor risk | 0.539817 | 0.327945 | 0.480010 | 0.066465 |

Against the same-flight baseline, the selected structural-rotation candidate improves
joint log loss by 0.016251, or 2.93% relative, and multiclass Brier by 0.010452, or
3.10% relative. Both paired intervals exclude zero:

| Comparison | Proper score | Difference | Paired 95% date-cluster interval |
|---|---|---:|---:|
| Structural rotation - baseline | Joint log loss | -0.016251 | -0.018948 to -0.013951 |
| Structural rotation - baseline | Multiclass Brier | -0.010452 | -0.011909 to -0.009189 |
| Weather - baseline | Joint log loss | -0.015869 | -0.018587 to -0.013663 |
| Weather - baseline | Multiclass Brier | -0.009664 | -0.011006 to -0.008488 |

The structural candidate improves joint log loss in every month of 2025. The monthly
difference ranges from -0.033560 in January to -0.006376 in September; the benefit
does not depend on a single exceptional month.

### What the nested ablation does and does not establish

The supplemental paired analysis compares each layer to its immediate predecessor,
not only to baseline.

| Increment | Endpoint/score | Difference | Paired 95% interval | Interpretation |
|---|---|---:|---:|---|
| Weather - baseline | Joint log loss | -0.015869 | -0.018587 to -0.013663 | clear gain |
| Structural rotation - weather | Joint log loss | -0.000382 | -0.000858 to 0.000144 | interval crosses zero |
| Structural rotation - weather | Joint Brier | -0.000788 | -0.001004 to -0.000588 | clear gain |
| Structural rotation - weather | Delay log loss | -0.001228 | -0.001518 to -0.000958 | clear delay gain |
| Structural rotation - weather | Cancellation log loss | +0.000824 | 0.000355 to 0.001345 | cancellation tradeoff |
| Propagated risk - structural rotation | Joint log loss | +0.000591 | 0.000099 to 0.001094 | clear degradation |
| Propagated risk - structural rotation | Joint Brier | +0.000869 | 0.000653 to 0.001106 | clear degradation |

Thus the large overall advance is principally a forecast-vintage weather result. The
schedule-only latent rotation layer adds robust Brier and conditional-delay value,
but its incremental joint log-loss advantage is uncertain and it sacrifices some
cancellation performance. Propagated predecessor risk is a retained negative result.

All eight forward calibration choices were identity. The 0.05-grid convex ensemble
selected weight 1.0 on structural rotation and zero on the other candidates. The
aggregate reconciliation hypothesis also failed honestly: every nonzero alignment
strength worsened 2024 selection log loss, from 0.445076 at the weakest tested
alignment to 0.468340 at the strongest, versus 0.441953 for identity. The frozen 2025
output therefore performs no aggregate adjustment.

### Stability, coverage, and report recovery

The structural-minus-baseline joint-log-loss difference increases monotonically
across the frozen weather-severity quartiles: -0.006289 (Q1), -0.008767 (Q2),
-0.015752 (Q3), and -0.037102 (Q4). Only 452 joint-scored rows lack the weather
covariate; their descriptive difference is +0.000533, so no missing-weather benefit
is claimed.

The original audit wrote all 12 monthly prediction and aggregate partitions before
its final report assembly rejected ordinary float32 simplex round-off. Maximum row-sum
error was 4.47e-8 and maximum per-probability correction was 3.33e-8; no row exceeded
the declared 1e-6 recovery bound. A create-only finalizer normalized each saved row
in float64, without refitting or repredicting, and the independent validator reopened
all 5,829,666 predictions. The failure, deterministic reproduction, recovery rule,
and recovered report remain separately visible.

This is a substantial controlled improvement over the rich census baseline. It is
also much larger in relative magnitude than the earlier 2023 graph-only delay gain
of about 0.04%, but that contextual comparison spans different years and protocols.
PAFRA's 2025 binary relative gains (about 1.10% for delay and 6.07% for cancellation)
also use a different baseline and cannot be ranked directly against FLARE-24's joint
three-state gain.

## Main result: fixed-lead weather adds reproducible information

Prior-Anchored Forecast Residual Adaptation (PAFRA) starts from a frozen HMOP model
logit and learns an additive CatBoost correction from closed-left operational history,
calendar Fourier terms, and GFS forecasts archived at a fixed 24-hour lead. A matched
operational-only adapter controls for within-year adaptation without forecast weather.

The method was developed in January--August 2024, early-stopped in September, and
selected on October--December. Both tasks selected the forecast residual. The exact
artifacts were then frozen in `manifests/forecast24_pafra_method_lock_v1.json` before
the new method was scored on 2025. The 2025 run performs no model or calibrator refit.

### 2024 Q4 selection block

| Task | Candidate | Log loss | Brier | AUROC | AP |
|---|---:|---:|---:|---:|---:|
| Delay | Frozen HMOP | 0.421649 | 0.129886 | 0.657247 | 0.263239 |
| Delay | Operational residual | 0.421613 | 0.129876 | 0.657409 | 0.263341 |
| Delay | **PAFRA forecast residual** | **0.418967** | **0.129004** | **0.666532** | **0.277146** |
| Cancellation | Frozen HMOP | 0.035038 | 0.006223 | 0.763380 | 0.040229 |
| Cancellation | Operational residual | 0.035266 | 0.006225 | 0.748603 | 0.032027 |
| Cancellation | **PAFRA forecast residual** | **0.032188** | **0.005982** | **0.814446** | **0.140082** |

The paired forecast-minus-baseline log-loss differences are -0.002682 (95% interval
-0.004235 to -0.001192) for delay and -0.002850 (-0.004764 to -0.001288) for
cancellation. Forecast weather also beats the operational-only adapter on both tasks.

### Full-year 2025 retrospective audit

The audit covers 5,814,817 top-100-airport flights. Delay metrics use the 5,732,449
non-cancelled flights with an observed delay label; cancellation uses all rows.

| Task | Candidate | Log loss | Brier | Brier skill | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Delay | Frozen HMOP | 0.494872 | 0.160800 | 0.073719 | 0.682526 | 0.375384 |
| Delay | Operational residual | 0.494890 | 0.160801 | 0.073713 | 0.682458 | 0.375396 |
| Delay | **PAFRA forecast residual** | **0.489426** | **0.158790** | **0.085301** | **0.694328** | **0.392975** |
| Cancellation | Frozen HMOP | 0.071273 | 0.014202 | -0.017028 | 0.740113 | 0.046941 |
| Cancellation | Operational residual | 0.070822 | 0.014232 | -0.019123 | 0.745624 | 0.051111 |
| Cancellation | **PAFRA forecast residual** | **0.066945** | **0.013825** | **0.010015** | **0.786102** | **0.072950** |

Same-flight paired effects are the primary evidence:

| Task | Comparison | Log-loss difference (95% interval) | Brier difference (95% interval) |
|---|---|---:|---:|
| Delay | PAFRA - frozen HMOP | -0.005446 (-0.006432, -0.004559) | -0.002011 (-0.002375, -0.001666) |
| Delay | PAFRA - operational residual | -0.005464 (-0.006500, -0.004582) | -0.002012 (-0.002382, -0.001658) |
| Cancellation | PAFRA - frozen HMOP | -0.004328 (-0.005358, -0.003378) | -0.000378 (-0.000553, -0.000189) |
| Cancellation | PAFRA - operational residual | -0.003877 (-0.004689, -0.003079) | -0.000407 (-0.000557, -0.000266) |

Every interval favors the forecast adapter. This supports the bounded conclusion that
the fixed-lead forecast features add predictive information beyond both the frozen
historical model and an otherwise matched within-year adapter. It does not establish
a causal weather effect or validate a live production feed.

## Calibration selection and negative evidence

Five calibration families were compared in three expanding, forward-only folds within
2024 Q4. Cancellation selected identity. Delay selected an intercept adjustment by a
small pooled log-loss margin of 0.000240, but its paired interval (-0.000897,
0.000398) included zero. Raw PAFRA was therefore frozen as primary, with calibration
only a prelocked sensitivity analysis.

On 2025 the delay intercept adjustment worsened log loss from 0.489426 to 0.489822 and
Brier score from 0.158790 to 0.159004. This negative result is retained; the audit
cannot be used to replace the already locked primary method.

## Observed-to-Forecast Weather Transfer ablation

The separate OFWT experiment learns shared weather semantics from past realized daily
weather and substitutes archived 24-hour GFS values at inference. On the sampled
full-year 2024 selection cohort it improved delay log loss from 0.477532 to 0.469080
and cancellation log loss from 0.063635 to 0.059313. Delay AUROC was 0.709715 and
cancellation AUROC was 0.844685. Paired log-loss intervals excluded zero for both
tasks. This corroborates the forecast signal through a different transfer mechanism;
its absolute scores are not compared to the PAFRA Q4 block because the date ranges
differ.

## Rolling-origin operational baseline

The frozen task-specific HMOP configuration was evaluated on 200,000-row annual
samples. The year-specific results remain visible because prevalence and regime shift
make a simple grand mean misleading.

| Year | Delay prevalence | Delay log loss | Delay AUROC | Cancellation prevalence | Cancellation log loss | Cancellation AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.1929 | 0.460357 | 0.673757 | 0.0176 | 0.082284 | 0.744047 |
| 2020 | 0.0851 | 0.303227 | 0.609808 | 0.0586 | 0.234041 | 0.817551 |
| 2021 | 0.1675 | 0.415217 | 0.696829 | 0.0169 | 0.077278 | 0.773386 |
| 2022 | 0.2156 | 0.486681 | 0.679535 | 0.0281 | 0.114426 | 0.769359 |
| 2023 | 0.2083 | 0.475748 | 0.683329 | 0.0126 | 0.062627 | 0.752350 |

The 2020 fold exposes severe probability shift: delay Brier skill is -0.0488 and
cancellation Brier skill is -0.0043. It is retained as a stress regime, not removed to
improve an aggregate.

## Weather-severity and operational heterogeneity

Weather strata use q90/q99 airport-day thresholds learned from 2024 forecast
covariates without outcomes. A flight takes the maximum condition at its two
endpoints, with extreme overriding adverse.

| Task | Stratum | Rows | PAFRA - baseline log loss (95% interval) |
|---|---:|---:|---:|
| Delay | Typical | 3,319,769 | -0.003563 (-0.004450, -0.002671) |
| Delay | Adverse | 2,128,664 | -0.006291 (-0.007917, -0.004771) |
| Delay | Extreme | 284,016 | -0.021129 (-0.028258, -0.014008) |
| Cancellation | Typical | 3,344,156 | -0.005118 (-0.006002, -0.004242) |
| Cancellation | Adverse | 2,176,039 | -0.003529 (-0.005447, -0.001734) |
| Cancellation | Extreme | 294,622 | -0.001264 (-0.006601, 0.003548) |

The increasing delay benefit under more severe forecasts is consistent with a useful
weather mechanism, but remains associational. Cancellation's extreme-stratum
log-loss interval crosses zero and its Brier point estimate is worse, so a universal
extreme-weather cancellation improvement is not claimed.

Using minimum support of 10,000 rows and 50 positives, delay log loss improves for
13/14 airlines, 94/95 origins, and 94/95 destinations. Cancellation improves for
12/14 airlines, 91/97 origins, and 92/96 destinations. The complete group tables are
retained rather than reporting only winners. Hawaiian/Pacific and several smaller
airport groups are visible failure targets for future regional or partial-pooling
work; the present results are descriptive heterogeneity, not demographic fairness or
multiplicity-adjusted subgroup claims.

## Shift-recalibration negative result

Closed-Left Operational-Shift Recalibration (CLOSR) did not improve the rolling
baseline. Across 2020--2023, raw mean-year log loss was 0.420218 for delay and
0.122093 for cancellation. Expanding Platt calibration produced 0.422175 and
0.126604; the richer expanding shift model produced 0.426454 and 0.433934. The
last-year shift variant was also worse. All 24 artifacts are retained and verified.
This result argues against treating post-hoc annual recalibration as a reliable answer
to abrupt regime changes in this cohort.

## Boundary-complete small-airport experiment

BC-POT-R recomputes airport resource-time state from all schedule-visible flights that
touch a frozen target airport, including 3,216,969 flights to or from 255 context-only
airports. The scored population remains the identical top-100-to-top-100 cohort, and the
feature build uses complete daily schedule stacks rather than the estimator's
deterministic training sample.

On 5,754,266 primary-period flights, the Q4-selected residual-gated ensemble has
accuracy 0.773512 versus 0.773308 for the previous meta-stack: +0.000203 absolute
(95% paired date-cluster interval +0.000060 to +0.000349). This fails the predeclared
+0.05 and +0.10 breakthrough gates. Brier improves by -0.000231
(-0.000449 to -0.000012), while joint log loss worsens by +0.001072
(+0.000447 to +0.001760). Balanced accuracy is also lower.

The counterfactual-residual candidate receives effectively zero Q4 simplex weight.
Boundary pressure is still diagnostically meaningful: severe versus low route-shadow
residual is associated with +0.01634 disruption prevalence (+0.01224 to +0.02050).
Almost every flight has some nonzero boundary residual, so an any-change/no-change
interval is non-estimable rather than forced. Full tables, airport heterogeneity,
limitations, and the next issue-time state-assimilation design are in
`docs/BCPOTR_RESULTS.md`.

## Evidence identities

| Evidence | SHA-256 of file |
|---|---|
| FLARE-24 2024 selection report | `13fef0024e7a08139786ccdfc150a0c3db61208558d31be2a673f960b701e089` |
| FLARE-24 pre-audit method lock | `bac1cf84e9b58be00a4e5ccf511071b4b16697e84f36a0a62b1d4ee4906b8ecc` |
| FLARE-24 recovered 2025 audit report | `8a4e88070d1a55cb30d3292629c1615057a92c661c2998df9045fc601d204171` |
| FLARE-24 nested-ablation supplement | `d6bc3cb00c226d1dfebb79c0a80dd8b29b91106c362090bda27a9e386fc88a9c` |
| FLARE-24 publication bundle v2 | `04e7ed7f6770c34c41af67549323c1e5c38a420643f7c96635f059d6408ec896` |
| FLARE-24 unopened-2026 confirmation lock | `daecde5d7c6690d21012031755dba1fbf72edd2fda6e8d395c2f1c42e252d037` |
| FLARE-24 combined release asset validation | `0035f9dbf814346ad61b649d5df204d0c41995ef98a68ebaf6f23704a11e3517` |
| BC-POT-R recovered study report | `adaea6d8fb83d834f5a3b190f471990c7dab7fe218719dce39fadaf491c07279` |
| BC-POT-R independent study validation v2 | `f8a9d0398595bee3632aa642ccd3afb7469643923d4c6432c33608bdd9b1c028` |
| BC-POT-R publication manifest v2 | `b1f8d3fb2d5d2d09743c94228d0514150c475b807143b605b58cdf64f7cd9f83` |
| BC-POT-R publication validation v2 | `317c03cd91d5d3677140c03ad3736a62f1a9cf3dc2b2c28d5bcfffcdb33d7848` |
| PAFRA 2024 selection report | `c7ca2456022f2bf3e3b5bbfb510d6b9a806c332de8e5f7e515263502586aa2da` |
| PAFRA method lock | `b157d4bf5e8afba967c92922d15c438c8be44e377d8ef0d02fe50fedee54c051` |
| PAFRA 2025 retrospective report | `8fade492b1d32423b47dea58f0e8d93aeaaf4606376123e33582990c7d4b7404` |
| PAFRA 2025 diagnostic report | `4c601ab1fb8ac4e9ffd77eba555213c399d62d7632df8638cc9b7aa76c1543b5` |
| OFWT 2024 selection report | `7d6339541224434ef44d2374efb5d3f0832e29896cdf7f1a941b56d446f29e42` |
| HMOP rolling 2019--2023 report | `d9a814ceef9a5a0f8a9b702992abf04c8ec70a59f88c96bb8f29f46937859b44` |
| CLOSR negative-result report | `aecf209b164fbe5484df0cfbb2ec2c1ea2e93ef01dcd7525619fca61eb63043d` |

The FLARE-24 selection, method lock, audit, nested supplement, publication bundle,
confirmation lock, all 24 audit prediction/aggregate partitions, and every referenced
model/calibrator have independently validated checksums. The PAFRA audit report, its
duplicate run manifest, all 24 monthly prediction files, and all recorded source files
have likewise been independently checksum-verified.

## Comparison and data limitations

- The repository's older 2025 headline AUROCs were 0.6469 for delay and 0.6543 for
  cancellation. PAFRA is higher by about 0.047 and 0.132 respectively, but those are
  contextual differences, not a controlled improvement claim: the older experiment
  used a different sample and protocol.
- The current 2025 legacy-track cohort excludes diversions upstream. Its sample IDs
  remain unique and model-valid, but the inherited normalizer's post-first-chunk raw
  row locator component is incorrect. Labels, features, row membership, and metrics
  are unaffected; `reports/validation/SAMPLE_ID_LINEAGE_ERRATUM.md` records the issue.
- A corrected, independently validated official-census v2 track is being built for
  rich schedule, exact cohort-density, and closed-left flight-number history tests.
- The 2025 audit is explicitly retrospective because 2025 outcomes and older scores
  existed in the repository before PAFRA development. It is not described as a blind
  confirmation. The separate 2026 confirmation gate remains closed.

## Current promotion status

FLARE-24 has passed a 2024 forward selection, a pre-audit lock, a full 2025
retrospective same-cohort audit, paired nested ablations, and artifact validation.
The technical and publication bundle is suitable for a clearly labelled retrospective
GitHub/Zenodo research release once packaging and clean-environment checks pass. It is
not yet a blind-confirmatory result. The January-June 2026 analysis is frozen in
`manifests/confirmation_lock_v1.json`, but writing that lock did not authorize or open
2026 outcomes.
