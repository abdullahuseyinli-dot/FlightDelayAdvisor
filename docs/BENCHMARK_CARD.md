# Benchmark card

[Documentation index](README.md) · [Current results](CURRENT_RESULTS.md)

## Question and status

How much predictive information do archived weather and airport-network context
add to a schedule/history baseline for US domestic flight disruption a day ahead?

The intended cutoff is scheduled departure in UTC minus 24 hours. Completed model
experiments use retrospective proxies whose strict availability was not established.
The previous release candidate is withdrawn. The corrected timestamp-aware
pipeline is implemented, but has no completed real-data benchmark yet.

## Evaluation contract

| Item | Definition |
|---|---|
| Unit | One scheduled flight with a stable sample ID |
| Target population | BTS reporting-carrier flights with both endpoints in the frozen 100-airport cohort |
| Joint classes | On time, arrival delay at least 15 minutes, cancelled |
| Cancellation task | All scheduled cohort rows with an observed binary cancellation label; later diversion is not cancellation |
| Conditional delay | Operated, non-diverted flights with an observed arrival-delay label |
| Missing outcomes | Excluded from the affected endpoint, never imputed as on time |
| Primary metrics | Joint log loss and unscaled multiclass Brier (sum of three squared probability errors) |
| Secondary metrics | Argmax and balanced accuracy, class recall, AUROC, average precision, calibration and strata |
| Historical uncertainty | Paired flight-date bootstrap, 2,000 resamples; 361 or 365 dates according to cohort |
| Corrected sensitivity | Date and seven-day-block comparisons; not yet real-data results |
| Large-improvement targets | +0.05 / +0.10 absolute accuracy, i.e. +5 / +10 percentage points |

## Cohorts must remain separate

| Track | Coverage | Role |
|---|---|---|
| Legacy app | Approximately month-balanced 2010–2024 row sample; separate sampled 2025 check | Historical application baseline |
| Census context | 43,815,581 retained rows, 2018–2025, all 2,922 dates | Context continuity, not all-US coverage |
| FLARE census | 11,759,279 scheduled rows in 2024–2025 | Feature alignment and parent experiments |
| Parent 2025 audit | 5,814,817 observed joint states, full calendar year | Historical parent comparison |
| Latest matched audit | 5,754,266 joint states, January 3–December 29, 2025 | Boundary/stack comparisons |
| Boundary context | 255 additional airports, context only | Network-boundary extension, unchanged scored targets |
| Corrected track | Planned 2024 development inputs with reviewed availability evidence | Not yet materialized |

Temporal completeness and network completeness are different. Complete daily
context was built before estimator subsampling; the old random application sample
was not used as a full airport-traffic census.

## Temporal selection

The parent model used January–August 2024 training, September stopping, a
January–September fixed-iteration refit, and forward Q4 calibration/selection.
Its choices were locked before its full-2025 audit. Later architectures were
informed by already-known 2025 evidence, so the year is development-informed even
where a particular model was selected on 2024.

The corrected runner defines three seasonal forward 2024 folds, separate stopping
and scoring periods, and explicit feature/label availability checks. There are
14 initial trials per fold (42 total), not 42 completed real-data experiments.
Preprocessing is fitted on training only; training/stopping labels must have
arrived before the corresponding later prediction cutoffs.

No random flight split is acceptable for the main temporal claim. No 2025 reuse
becomes blind confirmation. The 2026 confirmation dataset remains unopened, but
incidental aggregate search exposure was recorded in the
[later source investigation](RESEARCH_CONTINUATION_20260905.md). Any corrected
method needs a new reviewed lock that accounts for this incident before confirmation
access; complete absence of exposure must not be claimed.

## Controlled comparison requirements

Use identical target IDs, endpoint masks, folds, nested estimator samples, seeds,
CatBoost interaction settings and tuning budgets for feature/context ablations.
Build schedule context from all eligible flights, not just training samples.
Retain all failures and negative candidates. Select on development folds, not on
the displayed retrospective leaderboard.

The historical small-airport study kept target IDs fixed but changed an interaction
setting and ensemble candidate set. It therefore does not isolate a data-only effect.
See [limitations](LIMITATIONS.md) and the [correction contract](CUTOFF_DATA_CONTRACT.md).

## Contribution boundary

The defensible present contribution is a documented experimental record,
network-coverage analysis, timing audit and reusable validation software.
It does not establish a new state of the art, a novel mechanism's superiority,
causality or deployable forecast quality.
