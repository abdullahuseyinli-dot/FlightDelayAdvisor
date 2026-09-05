# Claim-to-evidence crosswalk

[Manuscript evidence](README.md) · [Project status](../docs/PROJECT_STATUS.md)

| Statement | Evidence | Permitted interpretation |
|---|---|---|
| The retained 2018–2025 census has complete daily coverage. | [Context audit](../reports/validation/census_top100_context_completeness_v1.json) | All retained dates, not all-US network or operational information completeness |
| Weather accounts for most of the historical parent-model gain. | [Nested ablation](../reports/experiments/flare24_2025_nested_ablation_v1.json) | Same parent cohort under the old proxy timing contract; not corrected T−24 performance |
| Structural rotation modestly improves selected historical scores over weather. | [Nested ablation](../reports/experiments/flare24_2025_nested_ablation_v1.json) | Joint log-loss interval crosses zero; delay/Brier benefit and cancellation trade-off must remain attached |
| Task-factorized resource models and cancellation stacking were more useful than the direct full resource model. | [CC-RTH report](../docs/CCRTH_TECHNICAL_REPORT.md), [meta-stack report](../reports/experiments/flare24_ccrth_metastack_v1.json) | Historical comparisons, not architecture-wide superiority |
| Boundary extension added context involving 255 additional airports. | [Boundary context manifest](../manifests/flare24_boundary_context_v1.json) | Context-only rows; no expansion of the scored target population |
| The selected boundary ensemble gains 0.0203 accuracy percentage points over the prior meta-stack. | [Matched results](../docs/CURRENT_RESULTS.md), [recovered report](../reports/experiments/flare24_boundary_pot_v6_recovered.json) | Small gain with worse joint log loss; confounded feature-only comparison; neither five- nor ten-point target met |
| Earlier-operating-date filtering alone does not prove per-flight T−24 availability. | [Withdrawal](../manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json), [data contract](../docs/CUTOFF_DATA_CONTRACT.md) | Validity finding; preserve old scores but withdraw strict timing claims |
| The source pilot retrieved issued 2024 forecasts/advisories. | [TAF pilot](../data/external/cutoff_taf_pilot_v1/report.json), [bounded source probe](../reports/validation/handoff_continuation_20260905_source_probe_v1/report.json) | Source access and producer issue metadata; no historical receipt or population-coverage proof |
| The corrected experiment campaign completed 42 real-data trials. | [Stopped register](../reports/experiments/handoff_continuation_20260905_register_v1.json) | Unsupported: all initial real-data slots remain unrun and corrected metrics are null |
| Transfer integrity resolves historical availability. | [Transfer record](../manifests/transfer/evidence_transfer_20260905_v1.json) | Unsupported: hashes establish copied bytes, not missing timing evidence |
| Confirmation remains completely unexposed. | [Search-exposure incident](../manifests/failures/handoff_continuation_20260905_incidental_search_exposure_v1.json) | Unsupported: dataset access did not occur, but aggregate search exposure was recorded |
| FLARE-24 is a validated novel state-of-the-art operational forecaster. | No qualifying evidence | Unsupported: no corrected confirmation, operational validation or novelty determination |

Before submission, regenerate numerical statements from the appropriate immutable
predictions or independently validated report, retain cohort and evidence class,
and record the final table/figure location. Documentation and file-integrity checks
cannot promote a blocked scientific claim.
