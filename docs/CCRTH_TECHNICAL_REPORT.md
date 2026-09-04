# CC-RTH and TF-CC-RTH: airport resource-time structure for T-24 flight disruption forecasting

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

Status: validated retrospective research evidence; prepared for a GitHub/Zenodo
software release; unopened-2026 confirmation remains pending.

## Abstract

This study asks whether flight disruption at scheduled departure minus 24 hours can
be improved by representing each flight as a user of shared, time-local airport
resources. The Capacity-Conditioned Resource-Time Flight Hypergraph (CC-RTH) adds
schedule pressure, static runway geometry, prior-year scheduling frontiers,
weather-conditioned capacity proxies, airport-local queues, metropolitan-system
pressure, route bottlenecks, and uncertain predecessor-aircraft messages to the
frozen FLARE-24 benchmark.

The direct 128-feature hypergraph did not improve joint prediction: on the primary
5,754,266-flight 2025 cohort, its joint log loss was 0.540214 versus 0.539577 for
FLARE-24, and its multiclass Brier score was 0.327804 versus 0.327218. However, it
improved cancellation ranking substantially. This negative-but-informative result
motivated Task-Factorized CC-RTH (TF-CC-RTH), which estimates cancellation and
delay-given-operation separately and recombines them as a coherent hurdle
distribution. The Q4-locked factorized method improved both joint log loss by
-0.000605 (95% paired date-cluster interval -0.000887 to -0.000345) and Brier by
-0.0000418 (-0.0000735 to -0.0000130).

A subsequent L2-regularized cancellation-logit stack combines the five CC-RTH
candidate cancellation forecasts while retaining FLARE-24 for conditional delay.
Its July 3-December 29 execution-held-forward result improved joint log loss by
-0.002103 (-0.003156 to -0.001186) and Brier by -0.000290 (-0.000471 to
-0.000132). Its full-primary descriptive changes were -0.002200 and -0.000307.
Because prior 2025 exploration informed this final architecture, these are strong
retrospective development results, not independent confirmation. Exact coefficients
and a success rule are locked for unopened 2026 evaluation.

## Research question and contribution

The central hypothesis was not merely that airport identity matters. FLARE-24
already contains airport, route, calendar, weather, history, and graph information.
The narrower question was whether explicit competition for shared resource-time
states adds information:

1. Which flights are scheduled immediately before and after a target flight at the
   same origin or destination?
2. How dense and imbalanced are arrival and departure banks at several time scales?
3. Is scheduled demand high relative to an airport's own season/hour frontier?
4. How does T-24 forecast weather interact with static runway geometry and the
   scheduling frontier?
5. Can local overload propagate through a plausible predecessor-aircraft chain or
   across the two endpoints of a route?

The principal methodological contribution is the empirical decomposition that
followed the negative monolithic result: airport-resource forecasts are useful much
more consistently for cancellation than for delay conditional on operation. A
task-factorized hurdle and a regularized logit-level evidence stack exploit that
asymmetry without changing the T-24 information boundary.

## Data and information boundary

The cohort is the frozen induced network of 100 airports used by FLARE-24: both
flight endpoints must be in the set. It contains 11,759,279 scheduled domestic
flights in 2024-2025. The primary CC-RTH evaluation
uses January 3-December 29, 2025; 5,754,266 flights have an observed joint state on
361 complete-context dates.

A separate census-context audit verifies all 2,922 calendar dates across 2018-2025
and exact raw-to-induced row reconstruction for every month of 2024-2025. It also
finds 2,207,091 flights with exactly one endpoint in the top 100, equal to 15.803% of
all flights touching the target airports. Those boundary flights are not nodes in
this v1 graph. The schedule is therefore temporally complete within its induced
cohort but not context-complete with respect to smaller airports.

For flight `i`, the cutoff is its scheduled departure in UTC minus 24 hours. Graph
construction uses only schedule fields, T-24 forecast-vintage weather, a static FAA
NASR runway catalogue, strictly prior-year scheduling frontiers, and a latent
rotation model inferred from schedule compatibility. It does not read target
outcomes, actual times, realized weather, target-year tail identifiers, delay causes,
or target-year outcome aggregates.

The BTS schedule is a retrospective proxy, not a proven D-24 operational snapshot.
No authentic issue-time gate assignment, active runway configuration, observed
queue, declared airport acceptance/departure rate, or traffic-management constraint
feed was available. Those quantities are not invented.

## Added airport and network variables

The executable registry declares 128 candidate resource features. Training-only
availability and variation filtering retained 120. The groups are:

| Group | Registered variables | Examples |
|---|---:|---|
| Static endpoint resources | 18 | runway and eligible-end counts, orientation families, parallel-runway proxy, ILS fraction, runway lengths |
| Scheduled demand | 38 | arrival/departure/movement counts over 15/30/60/120 minutes, leave-one-out load, preceding/following spacing, imbalance, burstiness, metro demand |
| Frontier and latent state | 60 | prior-year p50/p75/p90 service envelopes, scenario probabilities, utilization, slack, overload, queue, recovery, marginal overload, shadow price, metro state |
| Rotation messages | 5 | candidate-predecessor probability, message coverage, predecessor overload, queue, and shadow price |
| Route messages | 7 | two-endpoint overload, queue, slack, shadow-price, weather, metro, and bottleneck summaries |

Origin and destination annual-operations fields and optional operational-constraint
fields had no authentic usable coverage and were excluded. Runway counts describe
physical geometry, not the active or simultaneous runway configuration.

## Resource-time graph

Each scheduled flight connects to its origin-departure and destination-arrival
airport resources in deterministic 15-minute UTC buckets and, where registered, to
a metropolitan multi-airport resource. Candidate predecessor-successor edges are
probability weighted and constrained to total incoming mass at most one. The
materialized 2024-2025 graph contains:

| Object | Count |
|---|---:|
| Flights | 11,759,279 |
| Airport/metro resource-time nodes | 5,175,820 |
| Flight-resource incidence edges | 32,128,012 |
| Candidate rotation edges | 113,252,743 |

Same-airport demand is computed before training-row sampling. Queue proxies reset at
each airport-local operational-day boundary. A leave-one-flight-out calculation
prevents the focal flight from trivially supplying all of its own overload signal.
The later 125,000-row-per-month supervised training sample does not remove neighbouring
flights from these already-materialized resource states.

## Chronological design

The study does not use a random train/test split.

- January-August 2024 supplies training samples.
- August 30-31 are purged before September early stopping.
- Fixed-iteration models refit through September 30.
- October 1-2 separate refit from expanding Q4 calibration and selection folds.
- The parent CC-RTH choice is written before that formal runner opens 2025
  outcome-bearing partitions.
- Primary scoring is January 3-December 29, 2025, with paired bootstrap uncertainty
  clustered by all 361 flight dates.

The broad CC-RTH idea was developed after earlier 2025 FLARE results existed, so its
2025 audit is retrospective rather than blind confirmation. TF-CC-RTH was likewise
designed after aggregate CC-RTH results were known. The final meta-stack was designed
after exploratory 2025 inspection. To test temporal stability without overstating
independence, its coefficients are fitted on Q4 2024 predictions, regularization is
selected on January 3-June 30, 2025, July 1-2 are embargoed, and July 3-December 29 is
loaded only after the H2 execution lock. This held-forward ordering is not described
as epistemic blindness.

## Models and ablations

The four nested airport candidates add information to the exact persisted FLARE-24
probabilities:

1. `raw_demand`: static resource geometry plus schedule pressure;
2. `normalized_capacity`: demand plus prior-year frontier, weather/geometry
   scenarios, utilization, slack, and metro state;
3. `queue_shadow`: normalized state plus airport-local queue, recovery, marginal
   load, and shadow price;
4. `hypergraph`: all of the above plus route and predecessor-resource messages.

Each candidate uses matched CatBoost hurdle components. A Q4 convex simplex and a
shadow-price-regime simplex were also evaluated. TF-CC-RTH instead selects separate
convex simplexes for cancellation and delay-given-operation and recombines them as:

`P(cancelled)=c`, `P(delayed)=(1-c)d`, `P(on-time)=(1-c)(1-d)`.

The final development stack applies L2 logistic regression to the five clipped
cancellation logits. The selected `C=0.001` model has intercept 0.900465 and
coefficients 0.387346 (FLARE-24), -0.021546 (raw demand), 0.133640 (normalized
capacity), 0.393606 (queue/shadow), and 0.254491 (full hypergraph). Coefficients are
conditional on correlated predictions and are not causal feature effects.

## Results

### Direct airport-resource ablations

| Candidate | Joint log loss | Delta vs FLARE-24 | Brier | Delta vs FLARE-24 | Cancellation AUROC | Cancellation AP |
|---|---:|---:|---:|---:|---:|---:|
| FLARE-24 | 0.539577 | 0 | 0.327218 | 0 | 0.78934 | 0.08545 |
| Raw demand | 0.539183 | -0.000394 | 0.327703 | +0.000485 | 0.79574 | 0.08804 |
| Normalized capacity | 0.538815 | -0.000762 | 0.327557 | +0.000339 | **0.80120** | **0.09761** |
| Queue/shadow | 0.539836 | +0.000259 | 0.327761 | +0.000543 | 0.79472 | 0.08926 |
| Full hypergraph | 0.540214 | +0.000638 | 0.327804 | +0.000586 | 0.79871 | 0.09667 |
| Q4 capacity-gated blend | 0.539559 | -0.000017 | 0.327217 | -0.0000008 | 0.78956 | 0.08563 |

Normalized capacity has a favorable log-loss interval but an unfavorable Brier
interval. Raw demand's log-loss interval crosses zero and its Brier result is worse.
The full hypergraph is worse on both joint scores. The Q4-gated blend uses 91.8%
FLARE-24 and 8.2% queue/shadow only in the severe regime and is otherwise effectively
FLARE-24; its Brier interval crosses zero.

### Task factorization

The Q4 stress-gated TF-CC-RTH cancellation mixture uses resource-state forecasts,
while its conditional-delay mixture is effectively 100% FLARE-24. On 2025:

| Method | Joint log loss | Delta (95% interval) | Brier | Delta (95% interval) |
|---|---:|---:|---:|---:|
| FLARE-24 | 0.539577 | - | 0.327218 | - |
| TF-CC-RTH global, Q4 lock | 0.538969 | -0.000608 (-0.000895, -0.000347) | 0.327175 | -0.0000425 (-0.0000744, -0.0000133) |
| TF-CC-RTH stress-gated, Q4 lock | 0.538972 | -0.000605 (-0.000887, -0.000345) | 0.327176 | -0.0000418 (-0.0000735, -0.0000130) |

Both scores improve in 11 of 12 months and in all frozen capacity-stress regimes.
The separate 25-pair component grid found a larger cancellation-source result, but
that grid is explicitly post-hoc and its interval is not selection-adjusted.

### Regularized cancellation stack

| Period | Flights with joint outcome | FLARE-24 log loss | Stack log loss | Log-loss delta (95% interval) | Brier delta (95% interval) |
|---|---:|---:|---:|---:|---:|
| July 3-Dec 29 execution-held-forward | 2,881,998 | 0.544976 | 0.542873 | -0.002103 (-0.003156, -0.001186) | -0.000290 (-0.000471, -0.000132) |
| Jan 3-Dec 29 descriptive | 5,754,266 | 0.539577 | 0.537377 | -0.002200 (-0.002957, -0.001506) | -0.000307 (-0.000440, -0.000189) |

For the full primary period, cancellation log loss changes from 0.067497 to
0.065312, Brier from 0.013655 to 0.013475, AUROC from 0.78934 to 0.80015, AP from
0.08545 to 0.09710, and equal-mass ECE from 0.00636 to 0.00444. Joint log loss
improves in all 12 months and Brier in 11; the only Brier regression is approximately
+0.000003 in August. Both scores improve in low, elevated, and severe stress regimes.

Relative to the Q4-locked stress-gated TF-CC-RTH point estimate, the stack's full
retrospective improvement is about 3.6 times larger for log loss and 7.3 times larger
for Brier. Relative to the original capacity-gated CC-RTH log-loss change, it is
about 126 times larger. Ratios to near-zero effects are descriptive and are not used
as inferential claims.

## Interpretation

The experiment rejects the simplistic claim that more graph columns or a larger
monolithic model necessarily improve forecasting. The airport variables expose a
useful cancellation signal, but several variants harm conditional-delay probability
or calibration enough to worsen the joint forecast. Separating the hurdle tasks
allows the resource signal to affect the endpoint it helps while preserving the
strong FLARE-24 conditional-delay model. The regularized logit stack then combines
partly complementary cancellation forecasts.

This supports an incremental predictive-information claim under the retrospective
proxy boundary. It does not prove that physical runway capacity or queues cause a
cancellation, nor that the method is production-ready or state of the art.

## Limitations and next confirmation

- The final architecture and grid were informed by prior 2025 inspection; H2 is
  execution-held-forward but not independent confirmation.
- The model has no actual gates, active runways, AAR/ADR, observed queues, or
  issue-time traffic-management program feed.
- The single NASR snapshot is not a historical cycle panel.
- The schedule is a final BTS schedule proxy rather than an archived D-24 schedule.
- The induced top-100 graph excludes 15.803% of raw 2024-2025 flights that touch a
  target airport but connect to a smaller airport. A context-only outer-network
  ablation is required to quantify the effect.
- Airport-level diagnostics are descriptive and not multiplicity adjusted.
- Results cover the top-100-airport US cohort only.

The exact selected stack, coefficients, feature order, hurdle recombination,
reference, metrics, paired-date bootstrap, and joint success rule are frozen in
`manifests/flare24_ccrth_2026_confirmation_lock_v2.json`. Success requires the upper
95% paired date-cluster bounds for both joint log loss and multiclass Brier to be
below zero, with no 2026 refit or reselection. No 2026 outcome has been accessed.

## Reproducibility and evidence

The complete commands are in `docs/CCRTH_REPRODUCIBILITY.md`. Principal immutable
identities are:

| Artifact | File SHA-256 | Internal self-hash |
|---|---|---|
| Parent CC-RTH report v5 | `d9c6057508f5ceb2bcdf7c64054e9069d47df79ceeb311122f19e1354c802942` | `f09b60b5428b8abf38630fe489205968a3b9f56d89ffde29d6034551bb046572` |
| Parent validation v3 | `3bf12514f8c55aa0bf3e1b33331baa40764ebc82e3d356d73c7e9cd63ca4bbb0` | `ab0dac293b208d1a41ca6b03c448f0cdda4cdbcef82986344b4d4db0f066f897` |
| TF-CC-RTH report v1 | `02f1e2f1e800677e19d62fa6167675a29df6949c906303e36a0f0ee79632b767` | `7a8ed8d0113f8a645728668c62bf4d2951c94baa12eb738a32a70f1f73e6f6b0` |
| TF-CC-RTH validation v2 | `4f9b23841f0e8f99c3ced4acfaca5c9a3e3dd89deb0f08319d7e82d0417b2d8d` | `9eed473457a0becacebdf5b17d3014118d7465d2d20b27e8a884632ce42ebdb4` |
| Meta-stack report v1 | `ead6177a3a2ff6d82f7720044ee922c1f655b5095ff3b350106ed52af7378383` | `f0cf09d6166a654cc589f312b036722d975981070f72facdc6c837da5b58dd47` |
| Meta-stack validation v2 | `a2fffcd8da36363e078ec7bb49dfa440693b60a57c1835cb5589548986dd7fad` | `7e79a423ff1d94d83fb9e72631200a38d2277d8a45f82b138b748069b2acbe37` |
| Publication bundle v9 | `3db805b72fc7276db35d1769479fbc958d443df436329cfd52f82f78d6a4d41d` | `875d749e9261410ca4af034fb28e3f4be424d967f3b8abe902c9fb2c1dae4e8b` |
| Publication validation v1 | `31ca4c099645749bdb8a7db4f5f146e70f55e39c16daae1a98d648f7d571ea04` | `82c24b8d0b11df3e5a953652edfdf57aa77ff06b76950b08265d908ec0caeb81` |
| Census/context audit v1 | `da19af74dde21f212d83a54f0db807fbcb5bd15c72e10fbf4c21c79a8634ef96` | `346bdbd7a1d8ae685b15caa1a4b90322c33a253129672ddbeda2fea42f7c5c84` |

The independent meta-stack validator refits all seven candidate models, redoes H1
selection, regenerates all 12 monthly outputs, reproduces every primary and secondary
score table and both paired intervals, and matches stored probabilities within
`2.23e-8`. The v9 publication validator checks 17 tables and 13 figures, including
manual visual inspection of the six new figures and byte identity of the seven
previously validated figures.
