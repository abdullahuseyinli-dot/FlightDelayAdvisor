# BC-POT-R results and interpretation

## Status

The historical BC-POT-R runs are complete, but the release candidate is withdrawn
after the cutoff audit. Scores remain retrospective proxy results, not a corrected
forecast benchmark or a claimed accuracy breakthrough. Earlier aggregate 2025
results were known when the architecture was designed; 2026 remains unopened.
The [current results](CURRENT_RESULTS.md) and [project status](PROJECT_STATUS.md)
govern interpretation of this detailed historical report.

The context comparison also changed CatBoost `max_ctr_complexity` from 4 to 1
and did not include the strongest prior meta-stack in the ensemble candidate set.
It therefore does not isolate the effect of adding smaller airports.

The scored cohort never changed: 5,754,266 jointly observed top-100-to-top-100 flights
from 3 January through 29 December 2025. Small-airport flights are schedule-only context
at a target-airport endpoint; their outcomes are neither features nor scored labels.
The operations-twin features were built from complete daily schedule stacks, not the
model-training sample. Only the 2024 estimator fit was deterministically capped at
125,000 rows per month for resource control.

## Primary same-flight comparison

Lower log loss and Brier are better. Accuracy changes are absolute proportions.

| Method | Accuracy | Change vs prior meta | Joint log loss | Change vs prior meta | Multiclass Brier | Change vs prior meta |
|---|---:|---:|---:|---:|---:|---:|
| Schedule baseline | 0.767493 | -0.005816 | 0.555921 | +0.018544 | 0.337727 | +0.010817 |
| FLARE-24 | 0.773289 | -0.000019 | 0.539577 | +0.002200 | 0.327218 | +0.000307 |
| CC-RTH gated | 0.773283 | -0.000025 | 0.539559 | +0.002182 | 0.327217 | +0.000306 |
| Prior meta-stack | 0.773308 | 0 | **0.537377** | 0 | 0.326911 | 0 |
| Boundary-complete view | **0.773673** | **+0.000365** | 0.538733 | +0.001356 | **0.326536** | **-0.000375** |
| Counterfactual-residual view | 0.773478 | +0.000169 | 0.538168 | +0.000791 | 0.326613 | -0.000298 |
| Global boundary ensemble | 0.773522 | +0.000214 | 0.538418 | +0.001041 | 0.326665 | -0.000246 |
| **Q4-selected residual-gated ensemble** | **0.773512** | **+0.000203** | **0.538449** | **+0.001072** | **0.326679** | **-0.000231** |

For the selected method, paired whole-date 95% intervals versus the prior meta-stack
are [+0.000060, +0.000349] for accuracy, [+0.000447, +0.001760] for log loss, and
[-0.000449, -0.000012] for Brier. Thus it makes a very small standard-accuracy and
Brier improvement while significantly worsening log loss. Balanced accuracy falls from
0.358780 to 0.357528. The requested +0.05 and +0.10 accuracy gates both fail.

A separately labelled Q4-selected class-bias diagnostic reaches 0.774922 accuracy, only
+0.001614 over the prior meta-stack. It is not substituted for the standard argmax
gate.

## What the selection learned

The global Q4 simplex uses 50.77% boundary-complete view, 49.23% earlier CC-RTH gated
forecast, and effectively 0% counterfactual-residual view. Every residual-pressure
regime also assigns the residual candidate zero weight. The gated variant shifts only
the first two weights and is marginally better than the global mixture on Q4 log loss,
so it was selected before 2025. That Q4 advantage does not persist: the global mixture
is slightly better than the gated mixture on all three 2025 aggregate metrics.

Small-airport context changes the modeled resource state, and the full-view model
shows a modest classification difference. The changed learner settings prevent
attributing that difference solely to context. The signed induced-versus-complete
residual did not improve the selected forecast under this representation.

## Network findings

- The context artifact contains 3,216,969 boundary flights and 255 context-only airports
  in addition to the frozen 100 target airports.
- The full boundary feature build computes 1,031,784 context-only predecessor states,
  8,271,919 context-only incidence edges, and 113,999,341 total rotation edges.
- At least one boundary residual is nonzero for 5,768,939 of 5,768,943 primary-period
  schedule rows. With only four no-change rows, the any-change contrast is saturated;
  its temporal-cluster interval is correctly reported as non-estimable.
- A newly observed context-only predecessor exists for 32,511 primary-period rows.
  These rows have 0.0993 lower joint-disruption prevalence
  (95% date-cluster interval -0.1254 to -0.0676). This likely reflects which regional
  routes acquire a confident schedule predecessor; it is not a protective causal
  effect.
- Severe versus low route-shadow residual identifies a genuinely different risk
  regime: +0.01634 joint disruption [+0.01224, +0.02050], +0.00150 cancellation
  [+0.00030, +0.00277], and +0.01541 delay conditional on operation
  [+0.01155, +0.01906]. These are descriptive associations.

Airport heterogeneity is broad but tiny: accuracy improves at 65/100 origins and
60/100 destinations, whereas log loss improves at only 30/100 origins and 17/100
destinations. This reinforces the aggregate conclusion that boundary context changes
decisions slightly but does not produce better calibrated probabilities consistently.

## Why a five-point jump did not occur

The dominant features remain recent flight/airline delay history, scheduled departure
time, seasonality, and weather. Boundary-derived variables do not enter the top ten
delay features. The current twin observes planned schedule load and static runway
geometry, but not the operational state that turns load into disruption: active runway
configuration, acceptance rates, traffic-management initiatives, gate and aircraft
assignments, maintenance, crew legality, or schedule revisions known at issue time.

This aligns with the broader evidence boundary. Aeolus represents delay propagation
with aligned flight chains plus shared aircraft, crew, and airport-resource graphs and
uses temporal splits with leakage prevention
([NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/586fbdff064d506f5af3e3db82681f84-Abstract-Datasets_and_Benchmarks_Track.html)).
CausalNet reports that static traffic or distance graphs miss heterogeneous,
nonstationary inter-airport propagation, although its airport-level 1--3 hour task is
not directly comparable to this flight-level T-24 benchmark
([paper](https://arxiv.org/abs/2407.15185)). FAA documentation shows that TFMData can
begin around 24 hours before operation and that TFDM/AFIS carries airport configuration,
demand/delay, departure restriction, gate estimate, and surface-metering information
([feed comparison](https://www.faa.gov/media/19566),
[SWIM roadmap](https://www.faa.gov/media/19436)).

## Recommended next research track

The highest-value next experiment is an **issue-time latent airport-state assimilation
model**, not a larger static graph network:

1. First satisfy the [corrected timing contract](CUTOFF_DATA_CONTRACT.md), rebuild
   the matched baselines and preserve an unchanged scored cohort.
2. Build 15-minute airport sequences from full-year schedule context, but replace the
   saturated binary boundary flag with continuous exposure: context-only movement
   share, directional load, hub-feeder concentration, predecessor entropy, and distance
   to a learned capacity frontier.
3. Assimilate archived D-24 schedule revisions, TFM initiatives, AAR/ADR or runway
   configuration, and terminal forecast vintages when legitimately obtainable.
4. Pretrain a masked event/state model on all schedule-only airport sequences, then
   update a compact latent congestion state as each issue-time operational message
   arrives. Fuse that state with the strong tabular meta-stack through a residual
   correction head whose default is exactly zero.
5. Use nested rolling-origin selection in 2024. Any further 2025 evaluation is
   development-informed retrospective analysis, never untouched confirmation.
   Keep 2026 closed until the corrected method and analysis are frozen.
6. Add a separate external-generalization track that scores regional airports with
   airport-held-out and temporal tests. Do not mix that changed population into the
   existing same-flight benchmark.

This design directly attacks the missing state and the lack of exposure contrast.
No honest evidence currently supports promising a +0.05 gain; the experiment should
retain that threshold as a falsifiable gate rather than an expected outcome.

## Immutable evidence

| Artifact | File SHA-256 | Self-hash |
|---|---|---|
| Recovered study report | `adaea6d8fb83d834f5a3b190f471990c7dab7fe218719dce39fadaf491c07279` | `aa52b6dc5cc7c94ae5e59fb150b07ae777e516a355f046201bb3182468d81779` |
| Independent study validation v2 | `f8a9d0398595bee3632aa642ccd3afb7469643923d4c6432c33608bdd9b1c028` | `c8a9f885305018db3767a1c56800e75db62faf1c3bb6335dfd5c962baf551ce8` |
| Publication manifest v2 | `b1f8d3fb2d5d2d09743c94228d0514150c475b807143b605b58cdf64f7cd9f83` | `a980f0b50d99bdf998fe484337b5d011ab7aa06fe934ebffc14c88f370a57a60` |
| Publication validation v2 | `317c03cd91d5d3677140c03ad3736a62f1a9cf3dc2b2c28d5bcfffcdb33d7848` | `16673c5863baffd5ac72f305da6fbe7c94a58d08b81f5a18a1da61a9446c0c27` |

The report has an exact byte-identical copy at
`D:/FlightDelayAdvisorResearchData/runs/flare24_boundary_pot_v6_recovery_v1/run_manifest.json`.
The validated tables and figures are under
`reports/figures/bcpotr_v6_recovered_v2/`.
