# FLARE-24 method and evaluation protocol

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

Status: frozen 2024-selected method with an independently validated, full-year 2025
retrospective audit. The January-June 2026 confirmation protocol is locked and its
outcomes remain unopened.

FLARE-24 means **Flight-Level Aggregate-Reconciled Ensemble at 24 Hours**. It
predicts a coherent three-state distribution—on time, delayed at least 15 minutes,
or cancelled—for each scheduled flight. The method is designed to test whether
forecast-vintage weather, aviation-aware transformations, and uncertain latent
aircraft rotations add information to a strong schedule and closed-left history
baseline, and whether independent aggregate forecasts can safely improve the
flight-level probabilities.

The acronym names the full tested framework. Selection was allowed to reduce any
unsupported component to identity. In the frozen method, the ensemble is 100%
structural rotation and aggregate alignment is identity; neither a nontrivial blend
nor reconciliation is claimed as a predictive contribution.

The canonical machine-readable specification is
`configs/flare24_v1.toml`. This document explains that specification; it does not
supersede it.

## Prediction estimand and population

For flight (i), the prediction cutoff is

\[
\tau_i = t^{\mathrm{scheduled\ departure,UTC}}_i - 24\ \mathrm{hours}.
\]

The joint state is ordered as on-time, delayed, cancelled. A diverted flight or a
row without a resolved joint outcome is not included in the joint proper-score
evaluation. Cancellation uses every scheduled cohort row with a binary cancellation
label, including later-diverted flights as non-cancellations. Conditional delay uses
operated, non-diverted rows with an observed `ArrDel15` label. Missing
delay outcomes are never recoded as on-time.

The cohort is the frozen top-100-airport union derived from the existing 2024 BTS
cohort and materialized from the official Reporting Carrier On-Time Performance
files. BTS is a retrospective schedule proxy. No equivalence to an operational
D-24 schedule snapshot is claimed.

## Information boundary

Every model input appears in `contracts.py` with an availability horizon. Target
outcomes, realised weather, target-year tail numbers, actual movement times, delay
causes, and other post-departure fields are forbidden predictors.

Historical route, carrier, airport-direction, flight-number, and graph-pressure
features use closed-left windows: the target calendar date is excluded. Schedule
density is calculated from scheduled fields before outcome filtering. The weather,
rotation, and aggregate builders project explicit column lists and record those
lists in self-hashed manifests.

No code path in the FLARE-24 study accepts a 2026 census partition. The 2026
confirmation outcome remains behind a separate lock.

## Forecast-vintage weather

The weather cube preserves airport, valid time, issue time, lead time, and value.
For every target hour and every variable, the selector takes the freshest nonmissing
record satisfying

\[
t^{\mathrm{issue}} \leq \tau_i.
\]

This matters because a forecast valid at destination arrival can have been issued
after the departure-based D-24 cutoff even when it carries a nominal one-day lead.
Day-2 values provide a safe fallback and also permit forecast-revision features when
both a fresher safe value and its day-2 counterpart exist. Missingness and achieved
lead are retained as predictors; missing records are not replaced with realised
weather.

Features cover origin departure, destination scheduled arrival, a -6/-3/0-hour
origin window, and 25/50/75-percent great-circle corridor proxies. The corridor
values come from the nearest airport forecast nodes, so coverage and proxy distance
are retained and exact gridded-route weather is not claimed.

## Aviation transformations

A dated FAA NASR runway snapshot supplies eligible runway headings. Wind is resolved
against all eligible headings and summarized as a wind-optimal envelope. This is a
bounded geometry feature, not an active-runway reconstruction. Derived features
include headwind, crosswind, gust crosswind, visibility categories, convective and
icing indices, snowfall, endpoint maxima, and weather × scheduled-bank-load terms.

## Capacitated latent rotations

Target-year tail numbers are never read. A connection kernel is learned from
prior-year tail-supervised examples and applied to three-day target schedule windows.
Candidate predecessor (j\rightarrow i) edges must be airport-compatible, carrier-
compatible, and within the prescribed turn-time interval. A learned nonnegative
edge score (s_{ji}) is converted to connection probabilities using dual penalties
for predecessor capacity:

\[
p_{ji} = \frac{s_{ji}\exp(-\lambda_j)}
 {1 + \sum_{k\in\mathcal C_i}s_{ki}\exp(-\lambda_k)},
\qquad
\sum_i p_{ji} \leq 1.
\]

The extra denominator term is a no-observed-predecessor state. The resulting
features measure candidate count, matched mass, maximum edge probability, entropy,
expected turn time, tight-connection mass, and competition. A target-outcome-free
upstream risk is formed by logit-pooling the registered closed-left 7/28/90-day HMOP
rates and combining cancellation with delay given operation. Its expectation under
the predecessor-edge probabilities becomes the inbound-disruption feature. A
structural-only build is retained as an ablation rather than silently filling that
feature with target outcomes. A numerical feasibility repair changes only violated
nonnegative dual coordinates; every day records its
maximum inbound mass and the independent validator enforces the declared `1e-8`
floating-point tolerance.

These edges are probabilistic schedule hypotheses. They are not recovered aircraft
assignments and must not be described as operational tail tracking.

## Flight model and hurdle construction

Four nested CatBoost candidates are compared on identical flights:

1. rich schedule + HMOP + flight history + CL-SGMP graph pressure;
2. candidate 1 + cutoff-coherent weather and aviation features;
3. candidate 2 + structural latent-rotation features;
4. candidate 3 + propagated closed-left predecessor-disruption risk.

Each candidate fits cancellation probability (c_i) and delay conditional on
operation (d_i). They are combined without an impossible overlap:

\[
q_i = ((1-c_i)(1-d_i),\ (1-c_i)d_i,\ c_i).
\]

January–August 2024 trains the early-stopping fits and September selects each
candidate/task iteration count. Each model is then refit at that fixed count on
January–September before any Q4 prediction is made. This uses the validation month
without letting Q4 calibration or ensemble outcomes enter model fitting. The four
candidates are not permitted to use 2025 outcomes.

## Forward calibration and ensemble selection

Identity, intercept-only, Platt, beta, and isotonic calibration compete through
three expanding forward folds in 2024 Q4. A fold calibrates only on dates before its
validation dates. The selected family is then fit on Q4 2024 for frozen 2025 use.

The calibrated candidate joint probabilities are combined with nonnegative weights
on a 0.05 simplex grid. The weights minimize forward-fold joint log loss. The grid,
including losing weights, is retained in the selection report.

## Independent aggregates and soft marginal reconciliation

The aggregate forecaster does not sum the flight model. It independently estimates
three-state rates with a recency-weighted hierarchical Dirichlet-multinomial model
for origin-hour, destination-hour, carrier-day, and route-day groups. For group (g)
and state (c), it supplies a mean count \(\mu_{gc}\) and variance \(v_{gc}\).

For base flight probabilities (q_i), the reconciled marginals solve the strictly
convex objective

\[
\min_{p_i\in\Delta^2}
\sum_i \mathrm{KL}(p_i\|q_i)
+\frac12\sum_{g,c}
\frac{(\sum_{i\in g}p_{ic}-\mu_{gc})^2}{v_{gc}}.
\]

This is deliberately soft: an uncertain aggregate has less influence and no
unreliable total is forced as truth. The implementation solves the smooth dual with
a sparse group-by-flight incidence matrix. Its dual coordinates are diagonally
preconditioned by the square root of the Hessian diagonal at the base forecast; this
is an exact reparameterization that leaves the objective unchanged while handling
the orders-of-magnitude curvature range between major-airport and rare route/state
constraints. The implementation independently checks convergence, the primal
objective, the dual-gradient residual, and each probability simplex.

This component is best described as **uncertainty-weighted marginal alignment**.
It does not construct a coherent full joint distribution over all flight and count
outcomes. In particular, literature warnings about log-score comparisons of full
unreconciled and reconciled hierarchical distributions do not automatically justify
such a claim here. FLARE-24 evaluates ordinary per-flight categorical marginals with
proper log and Brier scores and reports reconciliation as useful only if it improves
the held-out dates.

The 2024 scale selection includes an explicit identity/no-alignment control in
addition to variance multipliers 0.5, 1, 2, 4, and 8. If every soft-alignment
candidate has worse forward-fold joint log loss, the frozen method leaves the
ensemble unchanged. Thus the reconciliation hypothesis is tested but cannot be
forced into the 2025 audit after a negative selection result.

## Chronological evidence stages

| Stage | Outcome access | Purpose | Claim level |
|---|---|---|---|
| 2024 Jan–Sep | 2024 only | model fit and early stopping | development |
| 2024 Q4 forward folds | earlier fold dates only | calibration, ensemble, reconciliation-scale selection | development |
| Pre-audit method lock | 2024 selection report only | hash and freeze every selected choice | no 2025 access |
| Full 2025 | frozen flight models/calibrators/weights/scale; 2024-only aggregate refit | same-cohort retrospective audit | retrospective, not blind confirmation |
| Jan-Jun 2026 | protocol locked; outcomes not acquired or read | future confirmation | unopened |

Primary evidence is joint log loss and multiclass Brier score. Uncertainty uses 2,000
paired bootstrap resamples of flight dates, preserving within-day dependence.
Cancellation and delay-given-operation metrics, monthly scores, missing-weather
strata, weather-severity strata, and major-origin strata are secondary diagnostics.

## Frozen selections and measured audit

The 2024 forward procedure selected identity calibration for all eight component
models, a 100% weight on the structural-rotation candidate, and identity/no aggregate
alignment. Every nonzero alignment strength worsened 2024 Q4 joint log loss. The
preconditioned dual solver made the alignment optimization numerically practical, but
that computational improvement did not convert into predictive selection.

The frozen method was then scored without refitting on the complete 2025 cohort. On
5,814,817 flights with a resolved joint state, structural rotation reduced joint log
loss from 0.555476 to 0.539225 and multiclass Brier score from 0.337528 to 0.327076.
The paired differences were -0.016251 (95% date-cluster interval -0.018948 to
-0.013951) and -0.010452 (-0.011909 to -0.009189), corresponding to relative
reductions of 2.93% and 3.10%. Joint log loss improved in every calendar month.

The nested ablation assigns most of the gain to fixed-vintage weather. Relative to
weather, structural rotation clearly improves conditional-delay log loss and joint
Brier score, but its joint-log-loss interval crosses zero and its cancellation log
loss is worse. Propagated predecessor risk is worse than structural rotation on both
joint proper scores. These results narrow the supported contribution rather than
being removed from the method account.

The original 2025 finalizer failed after all monthly inference artifacts had been
written because float32 row sums, whose maximum error was `4.470348e-08`, were tested
against a strict `1e-8` zero-relative tolerance. The transparent recovery reopened
the immutable predictions, normalized each row in float64, and reran only report
assembly. No row exceeded `1e-6`, the maximum probability adjustment was about
`3.33e-08`, and no model, calibrator, prediction, or selected choice changed.

## Prespecified ablations and failure policy

- baseline versus weather isolates forecast/aviation information;
- weather versus rotation isolates latent schedule connectivity;
- best single candidate versus convex ensemble isolates model diversity;
- ensemble versus reconciled isolates independent aggregate information;
- raw versus calibrated probabilities isolates probability correction.

All failures and negative ablations remain in versioned records. A component is not
promoted because it is novel-looking: it must improve the held-out proper score, and
the paired date-cluster interval must be reported. A null result can still support a
publishable benchmark or methods analysis.

## Reproduction and artifact integrity

Builders refuse to overwrite completed outputs and use `.part` files followed by an
atomic rename. Completion manifests include file hashes, row counts, projected input
columns, environment versions, code provenance, and explicit outcome-access flags.
The independent validator reopens the Parquet files and recomputes hashes, row counts,
schemas, finite-value checks, weather cutoffs, coverage, latent-capacity diagnostics,
and leakage declarations.

Raw BTS archives, Open-Meteo responses, FAA NASR input, failed builds, superseded
models, and prior reports are preserved. Large or license-restricted artifacts are
manifested locally and are not implied to be redistributable in a GitHub or Zenodo
source release.

## Claim limits

The validated 2025 audit supports comparative predictive improvement over the
prespecified same-cohort baseline on this retrospective BTS population. It does not
establish causal benefit, production latency, live schedule availability, active
runway state, recovered aircraft identity, operational safety, demographic fairness,
or universal state of the art. The 2025 result is not called blind confirmation
because earlier repository work had already accessed part of that calendar year.
