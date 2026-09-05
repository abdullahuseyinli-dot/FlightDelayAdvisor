# Result lineage and experiment history

[Current results](CURRENT_RESULTS.md) · [Complete experiment index](EXPERIMENT_INDEX.md)

This is the navigation layer for the work completed so far. Original reports,
manifests, failed runs and recovered outputs remain at their recorded paths.
The experiment index lists every JSON/Markdown experiment record, including
blocked trial registers; a listed artifact is not necessarily a successful fit.

## Research sequence

| Stage | Work retained | Evidence and interpretation |
|---|---|---|
| Legacy application | Sampled 2010–2024 delay/cancellation models and sampled 2025 backtest | [Legacy application](LEGACY_APPLICATION.md); different sampling and endpoints from the later census |
| Temporal baseline development | Expanding/recent windows, weighting, tuning, calibration, rolling evaluation and frontier tabular screens | [Historical results ledger](RESULTS.md), [experiment index](EXPERIMENT_INDEX.md); development comparisons, not a pooled leaderboard |
| Joint endpoint and census | Hurdle/direct joint models, full daily airport context, recent histories and graph pressure | [Benchmark card](BENCHMARK_CARD.md), [census audit](CENSUS_CONTEXT_AUDIT.md); context completeness is distinct from estimator sampling |
| Forecast adaptation | Fixed-lead GFS acquisition, operational controls, forecast residuals, calibration and 2025 diagnostics | [Forecast audit](../reports/experiments/forecast24_pafra_2025_audit_v1.json), [diagnostics](../reports/experiments/forecast24_pafra_2025_diagnostics_v1.json); historical binary-task results |
| FLARE-24 | Weather/aviation transforms, latent rotation structure, predecessor risk and aggregate reconciliation | [Technical report](FLARE24_TECHNICAL_REPORT.md), [nested ablation](../reports/experiments/flare24_2025_nested_ablation_v1.json); weather dominates; risk propagation and nonzero reconciliation are retained negative results |
| CC-RTH | Airport-resource/time context, task factorization and cancellation meta-stacking | [Technical report](CCRTH_TECHNICAL_REPORT.md); the full direct resource model did not improve joint proper scores |
| Small-airport boundary extension | Additional schedule context, operations-twin features, boundary rotations, counterfactual residuals and ensembles | [BC-POT-R results](BCPOTR_RESULTS.md); no large accuracy gain, and changed interaction settings confound the data-only contrast |
| Report-only recoveries | Parent float32 simplex correction and boundary saturated-contrast recovery | [Parent recovered report](../reports/experiments/flare24_2025_audit_v1_recovered.json), [boundary recovered report](../reports/experiments/flare24_boundary_pot_v6_recovered.json); preserved predictions, not new model fits |
| Cutoff audit and withdrawal | Inference argument correction; explicit event, publication, consumer and label availability | [Withdrawal](../manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json), [implementation record](CUTOFF_IMPLEMENTATION_STATE.md); old scores remain retrospective proxies |
| Source feasibility | Twelve-month 2024 schema audit, 16-case issued-TAF pilot and four fixed 2024 FAA/NWS requests | [Weather pilot](CUTOFF_WEATHER_PILOT.md), [continuation record](RESEARCH_CONTINUATION_20260905.md); source/issue metadata, not reviewed historical receipt coverage |
| Corrected development campaign | Initial grid and endpoint/history validation; 42 real-data trial slots unrun | [Stopped register](../reports/experiments/handoff_continuation_20260905_register_v1.json); no corrected fitted model or performance estimate |
| Cross-computer preservation | Complete identified local data/model/source transfer with byte verification | [Artifact guide](ARTIFACTS.md); restores bytes, not missing historical availability evidence |

## Comparability rules

The legacy sample, earlier rolling screens, parent full-year audit and latest
trimmed matched cohort answer different questions. Only the same-flight table in
[current results](CURRENT_RESULTS.md) supports the displayed accuracy contrasts.
Do not combine their sample sizes or rank their absolute metrics as one benchmark.

The latest selected boundary ensemble is 0.6019 percentage points above its
schedule/history baseline and 0.0203 points above the stronger cancellation
meta-stack. Its joint log loss is worse than the meta-stack's. The requested
five- and ten-point improvements were not achieved.

## Evidence that must remain visible

The availability gate stopped corrected real-data training. The 2026 confirmation
dataset was not opened, but unsolicited search snippets exposed aggregate 2026
statistics during the later source investigation. The
[incident record](../manifests/failures/handoff_continuation_20260905_incidental_search_exposure_v1.json)
is part of the evidence history; zero exposure must not be claimed. No new outcome
search or training is performed by this repository presentation work.

Older records describe the evidence state at their creation, including historical
unopened-outcome statements. This page and [project status](PROJECT_STATUS.md)
provide the later qualification without rewriting those records.
