# Literature map and candidate contribution

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

Last searched: 2026-09-03
Status: focused prior-art review completed for the release candidate; contribution
claims remain narrow and provisional rather than universal novelty or state of the art

## Positioning

Flight-delay papers often answer materially different questions: airport-level delay
forecasting over the next few hours, flight-level classification, delay propagation,
holding prediction, or retrospective explanation. Their scores are not directly
comparable when targets, populations, time horizons, information sets, sampling, or
splits differ. FlightDelayBench therefore treats proper scores on forward calendar
years as its primary evidence and uses headline accuracy/AUROC from prior work only as
context.

Recent graph work is especially relevant but does not make a graph architecture an
automatic improvement. Santos and Goncalves aggregate US flights to airport-hour
tensors and report that an LSTM has the lowest point error while their ST-GCN offers
network-aware interpretation and low complexity. DS-MGCSTNet fuses four static graphs,
a dynamic EMD graph, and an adaptive graph. TMTSC learns tree-structured multi-scale
temporal and spatial correlations. CausalNet estimates and self-corrects a Granger
causality graph. These methods target airport or network states rather than the present
24-hour-ahead flight-level probability task.

Strong tabular alternatives also require empirical testing under drift. TabM uses an
efficient ensemble-like MLP and is competitive on large tabular data. TabICLv2 reports
state-of-the-art tabular foundation-model results and greater scalability, but its
million-row setting still assumes roughly 50 GB of GPU memory. A July 2026 evaluation
finds systematic TFM degradation under real distribution shifts. Those findings
support testing frontier models, but not presuming that they beat native categorical
boosting on this temporally shifted, categorical-heavy cohort.

A focused September 2026 search found a close architectural antecedent that narrows
the novelty claim. Afrane, Xu, and Li's HPST model uses a boosting-based prior followed
by a GRU that adapts to recent spatiotemporal conditions. PAFRA therefore does not
claim to originate the broad idea of anchoring a dynamic model to a boosted prior.
The distinctions under test here are the additive CatBoost logit-offset formulation,
fixed 24-hour archived NWP rather than an ambiguously timed weather table, a matched
no-weather residual adapter, separate cancellation evaluation, paired proper-score
inference, and full-year multi-million-flight retrospective replication. The exact
combination was not found in the sources reviewed, but novelty remains provisional
until formal database and citation-chain searches are complete.

Lupescu and Aciobanitei's 2026 cost-sensitive CatBoost work is also a relevant large-
scale comparator: it reports chronological validation on roughly 240 million records
and proposes temporal undersampling without weather. Its balanced evaluation and F1
objective do not permit a numerical comparison to the present natural-prevalence
proper-score benchmark, but the census track should retain it as a scalability and
imbalance-handling baseline candidate.

Two late-2025 aviation studies sharpen the feature boundary. Lemay and Bastin identify
historical flight-number reliability, forecast weather, and traffic as important
airport predictors; the legacy balanced file cannot test flight-number history because
it discarded the identifier. The new census track therefore retains the published
flight number and computes its rates only over closed-left dates. Zhou's two-stage
delay-absorption model reports strong gains from upstream-rotation state, but those
variables describe an earlier leg and are not assumed available at this benchmark's
24-hour horizon. They are a useful upper-information comparator, not permission to use
post-schedule state.

## Candidate contribution under test

The contribution is intended to be empirical and protocol-level as well as
methodological:

1. **FlightDelayBench:** a leakage-audited, flight-level benchmark that distinguishes
   schedule/climatology, fixed archived forecast, and realised-weather oracle
   information; retains cancellations and missing delay outcomes correctly; and uses
   rolling calendar years, proper scores, and date-cluster uncertainty.
2. **Hierarchical Multi-Timescale Operational Priors (HMOP):** empirical-Bayes
   route, carrier, and directional-airport rates over closed-left 7/28/90-day windows,
   with momentum and acceleration contrasts. The target day is excluded.
3. **Directional Network Pressure (DNP):** inbound/outbound contrasts at both
   endpoints. This mechanism is retained only where it improves out-of-time evidence;
   its first delay screen was negative and its cancellation screen was positive for
   discrimination.
4. **Closed-Left Operational-Shift Recalibration (CLOSR):** a prospective
   recalibrator that combines a model logit with D-1 global, route, carrier, and
   endpoint shift signals. For year Y, coefficients are fit only on rolling
   predictions and labels before Y.
5. **Observed-to-Forecast Weather Transfer (OFWT):** learn comparable daily weather
   effects from past observed airport weather, then substitute GFS forecasts archived
   at a fixed 24-hour lead. Missing forecast periods fall back to strictly earlier-year
   airport-month climatology and remain flagged.
6. **Prior-Anchored Forecast Residual Adaptation (PAFRA):** initialize a second-stage
   booster at the frozen earlier-year HMOP logit and learn only additive corrections
   from 2024 closed-left operating signals and fixed-lead NWP. A matched
   operational-only residual model distinguishes adaptation gains from genuine
   forecast-weather gains. A fixed multi-harmonic annual Fourier embedding follows the
   temporal-representation lesson of Cai and Ye (2025). January--August trains the
   adapter, September controls early stopping, and untouched October--December selects
   the candidate before any 2025 retrospective audit.
7. **Official-census rich-schedule track:** retain all top-100 cohort schedule rows,
   including later diversions, and test published flight identity, scheduled block
   structure, and complete-cohort density through nested ablations. This track is kept
   separate from the legacy sample so a gain cannot be attributed silently to a changed
   cohort. Tail number and realised rotation state remain forbidden.

Names above are internal working names. A novelty statement requires improvement on
prespecified proper scores, mechanism-level ablation with paired date-cluster
intervals, and a formal database/citation-chain search. FLARE-24 has historical
proxy comparisons and a focused web prior-art review, not corrected T−24
validation. It has not had a systematic Scopus/Web of Science/TRID review.
Null and negative components remain relevant evidence; they do not establish
a new superior mechanism.

As of the current evidence, PAFRA has passed its 2024 Q4 selection ablation and a
locked full-year 2025 retrospective audit. Forecast residuals improve both log loss
and Brier score against the frozen and operational-only controls for both tasks, with
all global paired intervals excluding zero. Delay benefit increases across typical,
adverse, and extreme forecast strata. Extreme-weather cancellation remains uncertain,
and an additional delay intercept calibrator failed in 2025; both limitations are
retained in `docs/RESULTS.md`.

## FLARE-24 prior-art boundary

FLARE-24 extends the census track with flight-specific D-24 forecast-vintage
selection, aviation and corridor transformations, a schedule-only capacitated latent
rotation graph, a three-state hurdle distribution, and uncertainty-weighted alignment
to independently forecast airport, carrier, and route margins. Each ingredient has
nearby prior art, so the contribution is the precisely audited combination and its
same-cohort evidence—not a claim to have invented graph models, tail assignment, KL
regularization, or forecast reconciliation.

The weather distinction is substantive. Open-Meteo documents `previous_day1` and
`previous_day2` as values predicted 24 and 48 hours before valid time, whereas its
Historical Forecast API stitches the first hours of successive runs into a continuous
series. FLARE-24 preserves issue and valid time separately and selects against each
flight's departure-based cutoff. That avoids treating a good retrospective weather
analysis as if it had been available at prediction time.

FlightSense reports a large AUROC gain from tail-number rotation-chain features, but
those chains use observed aircraft identity and therefore answer a higher-information
question than D-24 inference without a target tail. The stochastic Air France work
and the Vueling tail-assignment case study likewise establish that aircraft routing is
a constrained assignment problem. FLARE-24 uses those lessons to infer bounded soft
connections from schedule compatibility while explicitly refusing an aircraft-
identity claim. Its testable question is whether uncertainty about rotation structure
is predictive when the actual target-year assignment is unavailable.

Forecast reconciliation is a mature field. KL-regularized soft reconciliation,
conditioning-based probabilistic reconciliation, score-optimized probabilistic
reconciliation, and discrete count reconciliation all predate this work. The closest
candidate distinction found in the focused search is a flight-level categorical
marginal projection against multiple overlapping, independently predicted aviation
count margins with uncertainty-dependent strength. This is narrower than a new theory
of probabilistic reconciliation and remains a provisional application/method claim.
The implementation does not produce a full coherent joint distribution over flights
and aggregate counts. That limitation is important because Panagiotelis et al. show
that the log score is improper for comparing full unreconciled and reconciled
hierarchical distributions. FLARE-24 instead evaluates per-flight categorical
marginals and retains multiclass Brier score and paired date-cluster comparisons.

The frozen 2024 selection and same-cohort 2025 audit are complete. FLARE-24 improves
joint log loss by 2.93% and multiclass Brier score by 3.10% against its prespecified
rich baseline, with both paired 365-date intervals excluding zero and a log-loss win
in all 12 months. The mechanism claim is narrower: fixed-vintage weather supplies
most of the gain. Structural rotations improve joint Brier and conditional-delay log
loss relative to weather, but their incremental joint-log-loss interval crosses zero
and cancellation log loss worsens. Propagated risk is worse, the ensemble collapses
to one candidate, and every nonzero reconciliation strength loses during selection.

The supported contribution is therefore a leakage-audited benchmark and an evidence-
backed combination of cutoff-coherent weather with schedule-only structural rotation,
not a claim that ensemble blending, propagated disruption, or reconciliation improves
prediction. The numerically preconditioned reconciliation solver is an implementation
advance within this codebase; broader algorithmic novelty is not asserted.

## CC-RTH prior-art and novelty boundary

CC-RTH must not be described as the first use of a hypergraph for flight-delay
prediction. Li et al.'s HAPFL represents flights as nodes and uses four hyperedge
families: common departure airport, common arrival airport, common origin-destination
pair, and common flight chain. It combines hypergraph attention, an O-D graph, and a
period-aware transformer. That is close prior art for the broad ideas of higher-order
flight relations, shared-resource competition, and sequential flight dependencies.

Aeolus is also directly relevant benchmark prior art. It aligns more than 50 million
flight records with tabular features, flight chains, and a graph containing shared
aircraft, crew, and airport-resource connections, and it explicitly emphasizes
temporal splits and leakage prevention. HyperIMTS is relevant architectural prior art
for learning temporal and cross-variable relationships with hypergraphs, although its
task is irregular multivariate time-series forecasting rather than flight-level
D-24 classification.

The narrower CC-RTH hypothesis is different from merely placing flights sharing an
airport into one hyperedge. It reifies a changing airport system as a resource-time
node and attaches a cutoff-valid state to that node. The distinctions being tested
are summarized below; they are candidate contribution boundaries, not proof of
universal novelty.

| Dimension | HAPFL / nearby graph work | CC-RTH under test |
|---|---|---|
| Relational unit | Flight-relation hyperedges or airport-network edges | Airport and metro resource-time nodes plus flight incidence |
| Time localization | Daily/multi-day flight hypergraphs and periodic fusion | Deterministic 15-minute resource buckets with 15/30/60/120-minute demand windows |
| Capacity state | Shared-resource relation learned from data | Prior-year scheduling frontiers conditioned by D-24 weather and cycle-dated runway geometry |
| Congestion mechanism | Learned attention/embedding | Explicit utilization, slack, queue, recovery, smooth shadow price, and leave-one-flight-out marginal load proxies |
| Rotation information | Flight-chain relation where available | Schedule-only probabilistic predecessor edges with bounded incoming mass and no target-year tail identity |
| Evaluation | Paper-specific datasets, horizons, labels, and classification metrics | Same-flight comparison with frozen FLARE-24 probabilities, chronological 2024 selection, 361-date 2025 retrospective proper scores, and date-cluster intervals |

FAA ASPM documentation confirms that the operational quantities approximated by the
current proxy layer are real and materially richer than BTS: quarter-hour demand,
airport-supplied AAR and ADR, runway configuration, and throughput distributions by
weather and runway configuration are defined products. NASA runway-configuration work
likewise identifies weather, traffic demand, current configuration, and airport
dynamics as important predictors. NASA ATD-2 surface scheduling further shows that
runway intent, demand/capacity imbalance, traffic restrictions, and runway/taxiway
outages affect operational queues and metering.

Those sources also expose the principal data limitation. The present graph uses
strictly prior-year *scheduled-count* frontiers, not observed physical capacity. Its
next scientifically useful extension is therefore not an unconstrained deeper neural
network first; it is a separately versioned operational-resource validation track
using issue-time-valid AAR/ADR, active or forecast runway configuration, demand,
traffic-management initiatives, runway/taxiway outages, and—where legitimately
available before D-24—gate or surface constraints. These data should calibrate or
replace the proxy service scenarios while the current BTS/NASR graph remains a fully
reproducible public baseline.

A targeted web search found no source implementing this exact combination of
resource-time nodes, prior-year empirical service scenarios, D-24 forecast-vintage
weather, static runway feasibility, focal-flight removal, queue/recovery/shadow
features, uncertain schedule-only rotation messages, and a same-cohort proper-score
ablation. This supports a focused combination/application claim only. It is not a
claim of firstness, state of the art, or a completed systematic-review search; Scopus,
Web of Science, TRID, and citation-chain review remain required before journal
submission.

### BC-POT-R empirical novelty boundary

BC-POT-R adds an explicit population intervention to that design: target flights are
held fixed while all top-100-touching smaller-airport schedules are introduced only as
resource and predecessor context. The induced and boundary-complete graphs are paired
flight by flight, and their signed difference is made observable to the learner. This
is narrower than a claim to have invented edge-based or spatiotemporal graph learning;
Aeolus, HAPFL, CausalNet, and newer edge-to-node GNN work are direct prior art for
flight-chain and network propagation.

The completed ablation is negative for the proposed counterfactual residual: it receives
zero Q4 ensemble weight and does not beat the previous meta-stack on joint log loss.
The full boundary view yields a small Brier/accuracy change and severe residual pressure
identifies a higher-risk regime. These are descriptive results, not yet an
isolated boundary-data effect: the campaign also changed CatBoost interaction
settings, and its timing contract requires correction. The network-coverage
analysis and negative experimental record remain useful. No firstness or
state-of-the-art claim is supported.

## Information-set guardrails

- HMOP and CLOSR use outcomes through the previous operating day only. BTS is a
  retrospective proxy for a live operations feed. The later audit found that this
  rule can include outcomes after a flight-specific T−24 cutoff; this affects
  benchmark validity, not merely deployment latency.
- Target-day schedule-density features are a supplementary cohort-density proxy. The
  2011--2024 legacy source is a research sample and the 2025 cohort excludes diverted
  flights, so these features cannot support a schedule-census or production claim
  without a genuine advance schedule snapshot.
- OFWT uses Open-Meteo `previous_day1`, documented as the forecast valid value made 24
  hours earlier. It does not use the seamless historical forecast as if it were a
  fixed-lead archive.
- Realised weather is an oracle-only upper-bound diagnostic.
- The January-June 2026 confirmation analysis is locked, but no 2026 flight outcome
  has been acquired or read for this release candidate.

## Completed frontier screens

On the sealed 2018 development fold, ChimeraBoost and two TabM variants did not beat
the tuned CatBoost proper score. These are retained as negative results. The realised
weather oracle reached AUROC 0.70635 and average precision 0.38851, exceeding the old
repository headline values, but it is explicitly non-deployable. HMOP direct reached
delay log loss 0.46121 before tuning, and the 12-trial search reached 0.46110. Rolling
years and the later fixed-lead retrospective audit supersede this screen as the main
evidence; the negative frontier results remain visible rather than being discarded.

## Primary sources

- Santos, L. F. F. M. and Goncalves, S. (2026), [Graph-Based Multi-Horizon
  Forecasting of Airport Delay Propagation in the U.S. Air Transportation
  Network](https://doi.org/10.3390/app16147110).
- [DS-MGCSTNet: Multi-Graph Convolution Spatial-Temporal Network
  with Dynamic and Static Graph Fusion for Flight Delay
  Prediction](https://doi.org/10.1016/j.jairtraman.2026.103047).
- [Airport flight delay prediction based on tree-structured multi-scale temporal and
  spatial correlation learning](https://www.sciencedirect.com/science/article/pii/S0969699726000955)
  (Journal of Air Transport Management, 2026).
- Zhu et al. (2024), [A Spatio-Temporal Approach with Self-Corrective Causal Inference
  for Flight Delay Prediction](https://arxiv.org/abs/2407.15185).
- Franco et al. (2025), [Graph machine learning for flight delay prediction due to
  holding manoeuvre](https://arxiv.org/abs/2502.04233).
- Gorishniy et al. (2024), [TabM: Advancing Tabular Deep Learning with Parameter-
  Efficient Ensembling](https://arxiv.org/abs/2410.24210) and the
  [official implementation](https://github.com/yandex-research/tabm).
- Qu et al. (2026), [TabICLv2: A better, faster, scalable, and open tabular foundation
  model](https://arxiv.org/abs/2602.11139).
- Loza et al. (2026), [Empirical Evaluation of Out-Of-Distribution Performance of
  Tabular Foundation Models](https://arxiv.org/abs/2607.26000).
- Han, Huang, and Wang (2024), [Model Assessment and Selection under Temporal
  Distribution Shift](https://proceedings.mlr.press/v235/han24b.html).
- Cai and Ye (2025), [Understanding the Limits of Deep Tabular Methods with Temporal
  Shift](https://proceedings.mlr.press/v267/cai25j.html).
- Zhou (2025), [Integrating Delay-Absorption Capability into Flight Departure Delay
  Prediction](https://arxiv.org/abs/2512.08197).
- Lemay and Bastin (2025), [Prediction of airport on-time
  performance](https://arxiv.org/abs/2601.00875).
- Afrane, Xu, and Li (2026), [A Hybrid Probabilistic Spatio-Temporal Model for Flight
  Delay Exceedance Prediction](https://doi.org/10.1109/AIRC69745.2026.11631438).
- Lupescu and Aciobanitei (2026), [Predicting Flight Delays Without Weather Data: A
  Cost-Sensitive CatBoost Approach with Temporal
  Undersampling](https://doi.org/10.1109/ECAI69016.2026.11613783).
- [Trends and gaps in machine learning based flight delay prediction in spatiotemporal
  and data-driven approaches: a systematic
  review](https://doi.org/10.1016/j.asoc.2026.115761).
- CatBoost, [baseline input documentation](https://catboost.ai/docs/en/concepts/input-data_baseline)
  and [`Pool.set_baseline`](https://catboost.ai/docs/en/concepts/python-reference_pool_set_baseline).
- Open-Meteo, [Previous Model Runs API](https://open-meteo.com/en/docs/previous-runs-api)
  and [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api).
- US DOT BTS, [Reporting Carrier On-Time Performance download
  table](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=%5D&gnoyr_VQ=FGJ).
- FAA Aeronautical Information Services, [28-day NASR subscription
  archive](https://aeronav.faa.gov/aero_data/28DaySub/).
- Li et al. (2026), [Hypergraph attention and periodic fusion learning for enhanced
  flight delay prediction](https://www.sciencedirect.com/science/article/pii/S1566253525011388).
- Xu et al. (2025), [Aeolus: A Multi-structural Flight Delay
  Dataset](https://papers.neurips.cc/paper_files/paper/2025/hash/586fbdff064d506f5af3e3db82681f84-Abstract-Datasets_and_Benchmarks_Track.html).
- AlKheder et al. (2026), [Edge-Based GNN for Network Delay Prediction Enhanced by
  Flight Connectivity](https://doi.org/10.3390/aerospace13020161).
- Li et al. (2025), [HyperIMTS: Hypergraph Neural Network for Irregular Multivariate
  Time Series Forecasting](https://proceedings.mlr.press/v267/li25bl.html).
- FAA ASPM, [Throughput Analysis definitions](https://www.aspm.faa.gov/aspmhelp/index/ASPM_Throughput_Analysis__Definitions_of_Variables.html),
  [Data Download definitions](https://www.aspm.faa.gov/aspmhelp/index/ASPM_Data_Download__Definitions_of_Variables.html),
  and [AERO definitions](https://www.aspm.faa.gov/aspmhelp/index/ASPM_AERO__Definitions_of_Variables.html).
- Puranik, Memarzadeh, and Kalyanam (2023), [Predicting airport runway configurations
  for decision-support using supervised learning](https://ntrs.nasa.gov/api/citations/20230009416/downloads/20230009416_Predicting_airport_runway_configurations_for_decision_support_using_supervised_learning_final.pdf).
- NASA ATD-2, [Surface Scheduling and
  Metering](https://ntrs.nasa.gov/api/citations/20170004722/downloads/20170004722.pdf)
  and [Phase 2 Technical Design
  Document](https://ntrs.nasa.gov/api/citations/20190031943/downloads/20190031943.pdf).
- Shelke, Shelke, and Kamerkar (2026), [FlightSense: an end-to-end MLOps
  platform for real-time flight-delay prediction via rotation-chain propagation
  features](https://arxiv.org/abs/2605.07364).
- Baty and Parmentier (2026), [Managing delay in tail assignment: from minimum
  turn time to stochastic routing at Air France](https://arxiv.org/abs/2602.10866).
- Pita et al. (2021), [The Tail Assignment Problem: a case study at Vueling
  Airlines](https://www.sciencedirect.com/science/article/pii/S2352146521000934).
- Zhang, Li, and Kang (2023), [Probabilistic Forecast Reconciliation with
  Kullback-Leibler Divergence Regularization](https://arxiv.org/abs/2311.12279).
- Panagiotelis et al. (2023), [Probabilistic forecast reconciliation: properties,
  evaluation and score optimisation](https://doi.org/10.1016/j.ejor.2022.07.040).
- Zambon, Azzimonti, and Corani (2024), [Efficient probabilistic reconciliation
  of forecasts for real-valued and count time
  series](https://link.springer.com/article/10.1007/s11222-023-10343-y).
- Zhang, Panagiotelis, and Kang (2024), [Discrete forecast
  reconciliation](https://doi.org/10.1016/j.ejor.2024.05.024).

## Search extensions before submission

- Search IEEE Xplore, Scopus, Web of Science, TRID, and Google Scholar for the final
  method names and their underlying mechanisms, including dynamic prior correction,
  hierarchical multi-window target encoding, and observed-to-forecast substitution.
- Build a target/horizon/split/information-set comparison table from papers whose full
  methods are accessible.
- Contact or inspect author repositories where a paper's split or forecast issuance
  time is ambiguous.
- Avoid “state of the art” wording unless identical cohort, endpoint, horizon, and
  temporal protocol comparisons can be reproduced.
