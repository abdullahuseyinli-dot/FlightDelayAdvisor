# BC-POT-R: Boundary-Complete Probabilistic Operations Twin

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

## Research question

Does schedule-visible traffic between the frozen top-100 airports and smaller airports
improve 24-hour-ahead disruption prediction for the **unchanged** top-100-to-top-100
flight population? The method tests this without adding small-airport flights to the
scored cohort and without reading their outcomes during feature construction.

BC-POT-R is a retrospective research extension of CC-RTH, not a claim of a production
airport digital twin. BTS final schedules proxy an advance schedule; the modeled
capacity, queue, recovery, and shadow-price fields are latent covariates rather than
observed FAA operational states.

## Why the earlier graph was incomplete

The original induced graph counted only top-100-to-top-100 flights. It therefore omitted
two schedule-visible mechanisms:

1. boundary flights consume departure or arrival slots at a top-100 endpoint; and
2. a flight arriving from a smaller airport can be the latent aircraft predecessor of a
   scored flight.

The second mechanism exposed an implementation limitation: rotation inference could
emit a boundary predecessor, but the predecessor's resource message was absent because
resource state had been computed only for scored flights. BC-POT-R computes a
weather-neutral resource state for those context-only predecessors while retaining the
cutoff-weather-conditioned state whenever the predecessor is itself a scored flight.

## Information boundary and populations

- Prediction time: scheduled departure minus 24 hours.
- Scored population: the frozen top-100-to-top-100 sample IDs, unchanged by construction.
- Context population: every BTS row with at least one frozen top-100 endpoint.
- Context fields: date, endpoints, carrier, flight number, scheduled departure, scheduled
  elapsed time, and distance.
- Forbidden feature inputs: outcomes, actual times, taxi values, cancellations, delays,
  diversions, and target-year tail identifiers.
- Weather: the existing cutoff-coherent FLARE-24 archive. Realized weather remains an
  oracle diagnostic only.
- Historical schedule frontier for target year Y: year Y-1 only.
- Confirmation outcomes: 2026 remains unopened.

The source artifact contains 20,761,297 schedule-only rows for 2023–2025: 17,544,328
induced rows and 3,216,969 boundary rows across 355 airports. Every monthly induced ID
set is checked against the frozen census before the artifact is accepted.

## Operations-twin construction

For each top-100 airport and 15-minute UTC resource bucket, the implementation projects
both endpoints of all context flights and retains only events physically incident on a
top-100 resource. This avoids inventing runway resources for the 255 context-only
airports while retaining their traffic contribution at the large-airport endpoint.

The model derives:

- multi-scale arrival, departure, and movement demand;
- a prior-year airport/direction/season/local-hour scheduling frontier;
- static cycle-dated NASR runway geometry;
- three weather-conditioned latent capacity scenarios;
- utilization, slack, overload, queue, recovery, marginal-load, and smooth shadow-price
  summaries;
- metro-area coupling; and
- probability-weighted one-hop latent rotation messages, including context-only inbound
  predecessors.

The expanded rotation variant also learns its transition and turn-time kernel from all
top-100-touching flights in Y-1, rather than learning only on the induced top-100 graph.
Historical tail identifiers are used only as prior-year supervision and are discarded
from the target-year feature path. The fixed annual kernel uses January-September of
Y-1 only, leaving at least 91 days before the earliest target cutoff; this avoids using
late-year actual tail assignments that would not yet have appeared in the public BTS
release. BTS requires Form 234 submissions after each reporting month, while its release
history shows that public releases can arrive materially later, so submission due dates
are not treated as public availability
([BTS reporting directive](https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-40-technical-directive-reporting-time),
[TranStats release history](https://www.transtats.bts.gov/releaseinfo.asp?6o=FGK&qv52ynB=4ryrn5r)).
Before a full second materialization is allowed, an
outcome-blind January smoke comparison must preserve every target ID and every
non-rotation feature while changing rotation-message features for at least 0.1% of
rows. This gate is frozen in `configs/flare24_boundary_rotation_v1.toml`.

FAA ASPM describes airport arrival/departure demand and rate data at quarter-hour
resolution, which motivates the resource-time representation but does not make the BTS
proxy an ASPM measurement ([ASPM airport quarter-hour dictionary](https://www.aspm.faa.gov/aspm/Dict_AirportQtr.pdf)).
The adapter boundary is designed for later archived TFM/CDM and schedule feeds; FAA's
feed comparison documents schedule, tail, gate, TMI, and roughly 24-hour TFMData content
([FAA data-feed comparison](https://www.faa.gov/media/19566)), while the CDM program
describes collaborative traffic-flow information exchange
([FAA CDM/TFM FAQ](https://cdm.fly.faa.gov/faq)). These authentic feeds are not silently
claimed present in this experiment.

## Paired residual architecture

The same flight is represented under two deterministic graphs:

- `induced`: top-100-to-top-100 context only;
- `boundary_complete`: all top-100-touching context, projected onto top-100 resources.

Their signed difference is the **boundary residual**. Missing numeric states are set to
zero only for subtraction, and explicit observation-change indicators disclose when a
rotation state becomes newly observed or is lost. The residual is not interpreted as a
causal treatment effect.

Two binary-hurdle CatBoost candidates are frozen:

1. `boundary_only`: FLARE-24 plus static resource fields and the boundary-complete dynamic
   state;
2. `counterfactual_residual`: FLARE-24 plus the induced graph and the signed boundary
   residual.

Cancellation and delay-given-operation heads are trained separately and recomposed into
the on-time/delayed/cancelled simplex. A non-negative Q4-2024 simplex may combine the
frozen earlier CC-RTH forecast with both new candidates. A second predeclared
mixture-of-experts fits separate simplex weights in low, elevated, and severe strata of
the signed route shadow-price residual; its cutpoints come from Q4 covariates and its
weights from the same purged forward Q4 selection rows. The global and residual-gated
ensembles are compared only on Q4, before the new 2025 predictions are produced. This is related in spirit to
recent network-aware aviation forecasting benchmarks, but BC-POT-R's paired
schedule-graph residual is a specific testable construction rather than a claim that the
architecture is uniquely novel ([Aeolus benchmark paper](https://papers.neurips.cc/paper_files/paper/2025/file/586fbdff064d506f5af3e3db82681f84-Paper-Datasets_and_Benchmarks_Track.pdf)).

## Temporal protocol

- Training: January–August 2024, deterministically sampled at 125,000 rows/month.
- Early stopping: September 2024, at most 250,000 rows, with a two-date graph-boundary
  purge from the pilot fit.
- Refit: through September 2024 using the early-stopped tree count.
- Calibration and ensemble selection: purged forward folds in Q4 2024.
- Primary retrospective evaluation: 3 January–29 December 2025; boundary dates are
  excluded because adjacent 2026 schedule context is deliberately unopened.
- Primary metrics: joint log loss and multiclass Brier score with paired date-cluster
  uncertainty.
- Secondary metrics: standard argmax accuracy, class recall, balanced accuracy, and a
  separately labeled Q4-selected class-bias diagnostic.

The original schedule-only baseline, FLARE-24, CC-RTH, the previous meta-stack, and the
new candidates are joined by immutable sample ID and rescored on the identical primary
rows. Accuracy changes are reported as absolute proportions and percentage points; no
relative-percentage substitution is permitted.

Earlier aggregate 2025 results were known when this extension was conceived. Therefore,
new 2025 results are **retrospective redevelopment evidence**, even though all model
fitting and numerical selection use 2024 only.

## Breakthrough criterion

The requested gain is interpreted literally: at least **+0.05 absolute accuracy** versus
the best earlier meta-stack on the same primary rows, with +0.10 as a stretch gate. A
change such as 0.773 to 0.778 is +0.005 and fails. Relative percentages, error reduction,
AUROC improvement, and proper-score improvement cannot be substituted for this gate.

## Empirical outcome

The locked experiment is complete. On 5,754,266 identical primary-period flights, the
Q4-selected residual-gated ensemble has accuracy 0.773512 versus 0.773308 for the prior
meta-stack, an absolute gain of only +0.000203. The predeclared +0.05 and +0.10 gates
both fail. The date-cluster interval for the small accuracy difference is
[+0.000060, +0.000349], but its magnitude is not practically close to the requested
breakthrough.

The selected method improves multiclass Brier by -0.000231
[-0.000449, -0.000012] but worsens joint log loss by +0.001072
[+0.000447, +0.001760] relative to the prior meta-stack. Its balanced accuracy is also
lower (0.357528 versus 0.358780). The result is therefore mixed rather than a new best
probabilistic forecast.

The Q4 simplex assigns effectively zero weight to the counterfactual-residual model.
It combines the boundary-complete view (50.77%) and earlier capacity-gated forecast
(49.23%); the residual gate changes those two weights by pressure regime. This is direct
negative evidence against the current residual architecture, not evidence that airport
network context is irrelevant.

Boundary context changes at least one residual for 5,768,939 of 5,768,943
primary-period schedule rows. Consequently, an any-change versus no-change association
has no credible temporal overlap and is explicitly marked non-estimable. More targeted
signals remain useful: severe versus low route-shadow residual is associated with
+0.01634 disruption prevalence, and 32,511 rows acquire a schedule-visible predecessor
state from the context-only network. These are descriptive associations, not causal
effects.

See `docs/BCPOTR_RESULTS.md` for the full interpretation and immutable evidence map.

## What richer data would complete the twin

The current adapters intentionally leave authentic issue-time operational constraints at
zero coverage. The highest-value next sources are archived runway configuration and
AAR/ADR distributions, TFM initiatives and airport acceptance constraints, true D-24
schedule snapshots, gate/tail rotations, and forecast-vintage terminal weather. NOAA
LAMP exposes terminal forecast elements useful for a deployable version
([NOAA LAMP elements](https://vlab.noaa.gov/web/mdl/lamp-elements)); NASA has also
published work on runway-configuration prediction
([NASA runway configuration study](https://ntrs.nasa.gov/citations/20230013042)).

The completed result sharpens that priority. More model complexity over the same BTS
proxy is unlikely by itself to create a five-point accuracy jump. Aeolus independently
emphasizes aligned flight chains and shared aircraft, crew, and airport-resource graphs
with temporal splits and leakage controls
([NeurIPS 2025 Aeolus](https://papers.neurips.cc/paper_files/paper/2025/hash/586fbdff064d506f5af3e3db82681f84-Abstract-Datasets_and_Benchmarks_Track.html)).
FAA feed documentation identifies the missing issue-time variables directly: TFMData
begins roughly 24 hours before scheduled operation, while TFDM/AFIS includes airport
configuration, demand/delay, departure restrictions, gate estimates, and surface
metering ([FAA feed comparison](https://www.faa.gov/media/19566),
[FAA SWIM roadmap](https://www.faa.gov/media/19436)).
