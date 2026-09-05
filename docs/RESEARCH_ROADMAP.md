# Research roadmap

[Project status](PROJECT_STATUS.md) · [Sequential research plan](CUTOFF_RESEARCH_PLAN.md)

Further work is possible, but there is no evidence-backed promise of a five- or
ten-percentage-point improvement. Research is currently stopped at the source-
evidence gate. The following are conditional research opportunities, not running
jobs, completed implementations or claimed inventions.

| Priority | Question | Useful next deliverable | Go/no-go condition |
|---|---|---|---|
| 1. Information availability | Can we establish which schedule revision, forecast and operational label was genuinely available at each T−24 cutoff? | Reviewed source/revision/receipt contract and a small coverage audit | Authentic evidence; event clocks and HTTP retrieval dates alone are insufficient |
| 2. Dataset and scale | What do full context and additional estimator rows contribute under the corrected contract? | Matched 125k/250k/full learning curves on seasonal 2024 forward folds | Valid new input partitions; equal settings, targets and budgets |
| 3. Strong matched controls | Do current-feature CatBoost, LightGBM, TabM, direct and hurdle models improve the proper scores? | Complete initial grid, retuning, finalist repeats and forward incumbent predictions | Development-only selection; preserve zero-change fallback and all failures |
| 4. New information | Do issued weather scenarios, advance restrictions or schedule snapshots explain residual error? | Small matched source and incremental-value pilots | Auditable timing, licensing and flight-weighted coverage; uncovered flights stay visible |
| 5. Airport dynamics | Does an uncertain airport-capacity/recovery model add value beyond tabular controls? | Component ablations linking intra-airport queues and interairport predecessors | Preceding information-value evidence supports the added complexity |
| 6. Confirmation and release | Does a fixed corrected method generalize beyond its development evidence? | Reviewed lock, independent evaluation and exact-candidate archive | Account for prior outcome exposure; separate owner approval before access or publication |

## If historical availability cannot be recovered

A prospective collection of forecast issue/revision/receipt records and schedule
snapshots could establish a stronger future benchmark. Its prediction origins,
label-finalization policy and study period need to be fixed before evaluation.
This is new data collection, not a way to backfill unobserved receipts in 2024.

A separately labelled event-time/latency-assumption sensitivity study could also be
useful. It requires an explicit scope decision: the previous continuation stopped
without that authorization. Such a study must retain assumed versus observed
timestamps, quantify sensitivity, and never inherit strict historical T−24 claims.

## What the negative results suggest

- More complex models have not reliably displaced strong task-factorized controls.
  That does not prove architecture is exhausted: equal-budget, current-feature
  corrected comparisons remain unrun.
- The small-airport trial does not establish that those airports are irrelevant.
  It changed model complexity as well as context and needs a controlled ablation.
- Raw event times may improve event reconstruction, but cannot certify historical
  publication, receipt or schedule-vintage availability.
- A hindsight diagnostic may reveal possible information value. It is not a T−24
  model, an attainable operational gain, or an irreducible-error bound.
- Moving to T−6 or T−3 changes the prediction task and must have its own benchmark.

Every promoted finding needs paired same-flight metrics, class/season/airport and
coverage diagnostics, date/week uncertainty, runtime/memory measurements and an
entry in the [claim-to-evidence crosswalk](../paper/CLAIM_EVIDENCE_CROSSWALK.md).
