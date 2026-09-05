# Complete experiment index

[Result lineage](RESULT_LINEAGE.md) · [Current results](CURRENT_RESULTS.md)

This index contains 26 JSON/Markdown experiment records. It includes
development screens, negative results, historical proxy studies and the stopped
corrected-campaign register. Presence here does not imply a completed fit or
corrected T−24 validity. The index does not read prediction rows or model files.

Exact bytes, SHA-256 and original top-level statuses are recorded in the
[machine-readable inventory](../manifests/research_experiment_index_v1.json). Original report names are stable
provenance identifiers. Use result lineage for readable study descriptions.

| Record | Evidence class |
|---|---|
| [catboost_all_expanding_screen_v1.json](../reports/experiments/catboost_all_expanding_screen_v1.json) | Historical development / screening |
| [catboost_all_recent5_screen_v1.json](../reports/experiments/catboost_all_recent5_screen_v1.json) | Historical development / screening |
| [catboost_all_weight4_screen_v1.json](../reports/experiments/catboost_all_weight4_screen_v1.json) | Historical development / screening |
| [catboost_decay2_expanding_screen_v1.json](../reports/experiments/catboost_decay2_expanding_screen_v1.json) | Historical development / screening |
| [census_2023_development_screen_v1.json](../reports/experiments/census_2023_development_screen_v1.json) | Historical development / screening |
| [closr_rolling_v1.json](../reports/experiments/closr_rolling_v1.json) | Historical development / screening |
| [flare24_2024_selection_v2.json](../reports/experiments/flare24_2024_selection_v2.json) | FLARE selection / retrospective proxy |
| [flare24_2025_audit_v1_recovered.json](../reports/experiments/flare24_2025_audit_v1_recovered.json) | FLARE selection / retrospective proxy |
| [flare24_2025_nested_ablation_v1.json](../reports/experiments/flare24_2025_nested_ablation_v1.json) | FLARE selection / retrospective proxy |
| [flare24_boundary_pot_v6_recovered.json](../reports/experiments/flare24_boundary_pot_v6_recovered.json) | Boundary retrospective proxy |
| [flare24_ccrth_2025_retrospective_v5.json](../reports/experiments/flare24_ccrth_2025_retrospective_v5.json) | Airport-resource retrospective proxy |
| [flare24_ccrth_metastack_v1.json](../reports/experiments/flare24_ccrth_metastack_v1.json) | Airport-resource retrospective proxy |
| [flare24_ccrth_task_factorized_v1.json](../reports/experiments/flare24_ccrth_task_factorized_v1.json) | Airport-resource retrospective proxy |
| [forecast24_calibration_2024_selection_v1.json](../reports/experiments/forecast24_calibration_2024_selection_v1.json) | Forecast adaptation / retrospective proxy |
| [forecast24_ofwt_2024_selection_v1.json](../reports/experiments/forecast24_ofwt_2024_selection_v1.json) | Forecast adaptation / retrospective proxy |
| [forecast24_pafra_2024_selection_v1.json](../reports/experiments/forecast24_pafra_2024_selection_v1.json) | Forecast adaptation / retrospective proxy |
| [forecast24_pafra_2025_audit_v1.json](../reports/experiments/forecast24_pafra_2025_audit_v1.json) | Forecast adaptation / retrospective proxy |
| [forecast24_pafra_2025_diagnostics_v1.json](../reports/experiments/forecast24_pafra_2025_diagnostics_v1.json) | Forecast adaptation / retrospective proxy |
| [handoff_continuation_20260905_register_v1.json](../reports/experiments/handoff_continuation_20260905_register_v1.json) | Stopped corrected-campaign register; no real-data trials |
| [hmop_joint_hurdle_analysis_2019_2023_v1.json](../reports/experiments/hmop_joint_hurdle_analysis_2019_2023_v1.json) | Historical development / screening |
| [hmop_task_specific_plus_joint_direct_v1.json](../reports/experiments/hmop_task_specific_plus_joint_direct_v1.json) | Historical development / screening |
| [hmop_task_specific_strict_rolling_v1.json](../reports/experiments/hmop_task_specific_strict_rolling_v1.json) | Historical development / screening |
| [lightgbm_all_expanding_screen_v1.json](../reports/experiments/lightgbm_all_expanding_screen_v1.json) | Historical development / screening |
| [prospective_ensemble_screen_v1.json](../reports/experiments/prospective_ensemble_screen_v1.json) | Historical development / screening |
| [recent_joint_2018_screen_v1.json](../reports/experiments/recent_joint_2018_screen_v1.json) | Historical development / screening |
| [SCHEDULE_CONTEXT_LIMITATION_NOTE.md](../reports/experiments/SCHEDULE_CONTEXT_LIMITATION_NOTE.md) | Historical limitation record |

## Verification and later additions

Run `python tools/build_experiment_index.py --check` to verify completeness,
bytes, hashes and the rendered index. No source report is modified.

Create a new inventory/index revision when adding experiment records; preserve
the previous inventory as evidence. Update the current references and checks
together. The generator refuses to overwrite existing outputs.

Source-access pilots, schema/coverage checks, package validation and failures
are separately indexed in [artifacts](ARTIFACTS.md) and
[result lineage](RESULT_LINEAGE.md). Private console logs, model/prediction
partitions and older source archives remain in their original run/transfer tiers.
