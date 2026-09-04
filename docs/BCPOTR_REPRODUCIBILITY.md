# BC-POT-R reproducibility

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

Run commands from the repository root. All outputs are create-only; select a new version
suffix if any listed target already exists.

## 1. Build schedule-only boundary context

```powershell
uv run flightdelaybench-build-boundary-context `
  --raw-manifests manifests/raw_bts_census_2023.json manifests/raw_bts_census_2024.json manifests/raw_bts_2025.json `
  --airport-config configs/top100_airports_2024.json `
  --target-census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --output-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1 `
  --timezone-catalog D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1/airport_catalog.json `
  --manifest manifests/flare24_boundary_context_v1.json
```

## 2. Materialize the boundary-complete operations twin

```powershell
uv run flightdelaybench-build-ccrth `
  --census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --context-census-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1 `
  --context-manifest manifests/flare24_boundary_context_v1.json `
  --weather-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_features_v1 `
  --weather-feature-manifest manifests/flare24_features_v1.json `
  --airport-catalog data/external/forecast24_openmeteo_gfs_v1/airport_catalog.json `
  --context-airport-catalog D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1/airport_catalog.json `
  --resource-catalog data/external/flare24_ccrth_resources_nasr_20241003_v2.json `
  --resource-catalog-manifest manifests/flare24_ccrth_resources_nasr_20241003_v2.json `
  --rotation-manifest manifests/flare24_rotation_features_v5.json `
  --output-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_v1 `
  --graph-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_graph_v1 `
  --frontier-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_frontier_v1 `
  --manifest manifests/flare24_boundary_pot_v1.json `
  --target-years 2024 2025 `
  --artifact-mode features-only
```

`features-only` records the computed graph totals but does not duplicate sparse edge
tables. This is the predeclared storage-constrained research mode; full graph tables can
be regenerated under a new immutable run ID when sufficient archival space is available.

## 3. Validate features

```powershell
uv run flightdelaybench-validate-boundary-pot `
  --manifest manifests/flare24_boundary_pot_v1.json `
  --target-census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --output reports/validation/flare24_boundary_pot_v1.json
```

## 4. Fit and validate embargoed boundary rotation kernels

The frozen protocol uses only January-September tails in Y-1, leaving at least 91
days before the earliest target-year cutoff.

```powershell
uv run flightdelaybench-fit-boundary-rotations `
  --protocol configs/flare24_boundary_rotation_v1.toml `
  --raw-manifests manifests/raw_bts_census_2023.json manifests/raw_bts_census_2024.json `
  --airport-config configs/top100_airports_2024.json `
  --airport-catalog D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1/airport_catalog.json `
  --model-dir D:/FlightDelayAdvisorResearchData/models/flare24_boundary_rotations_v1 `
  --manifest manifests/flare24_boundary_rotations_v1.json

uv run flightdelaybench-validate-boundary-rotations `
  --manifest manifests/flare24_boundary_rotations_v1.json `
  --output reports/validation/flare24_boundary_rotations_v1.json
```

## 5. Run and validate the outcome-blind January rotation gate

Materialize January 2024 with the new kernels, using the same arguments as step 2
except for the paths and rotation manifest below:

```powershell
uv run flightdelaybench-build-ccrth `
  --census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --context-census-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1 `
  --context-manifest manifests/flare24_boundary_context_v1.json `
  --weather-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_features_v1 `
  --weather-feature-manifest manifests/flare24_features_v1.json `
  --airport-catalog data/external/forecast24_openmeteo_gfs_v1/airport_catalog.json `
  --context-airport-catalog D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_context_v1/airport_catalog.json `
  --resource-catalog data/external/flare24_ccrth_resources_nasr_20241003_v2.json `
  --resource-catalog-manifest manifests/flare24_ccrth_resources_nasr_20241003_v2.json `
  --rotation-manifest manifests/flare24_boundary_rotations_v1.json `
  --output-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_rotation_smoke_v1 `
  --graph-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_rotation_smoke_graph_v1 `
  --frontier-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_rotation_smoke_frontier_v1 `
  --manifest manifests/flare24_boundary_rotation_smoke_v1.json `
  --target-years 2024 --target-months 1 --artifact-mode features-only

uv run flightdelaybench-validate-boundary-pot `
  --manifest manifests/flare24_boundary_rotation_smoke_v1.json `
  --target-census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --allow-partial `
  --output reports/validation/flare24_boundary_rotation_feature_smoke_v1.json

uv run flightdelaybench-validate-boundary-rotation-smoke `
  --protocol configs/flare24_boundary_rotation_v1.toml `
  --rotation-validation reports/validation/flare24_boundary_rotations_v1.json `
  --reference-manifest manifests/flare24_boundary_pot_v1.json `
  --candidate-manifest manifests/flare24_boundary_rotation_smoke_v1.json `
  --year 2024 --month 1 `
  --output reports/validation/flare24_boundary_rotation_smoke_v1.json
```

Do not proceed unless the last report has status
`PASS_BOUNDARY_ROTATION_OUTCOME_BLIND_MATERIALITY_GATE`.

## 6. Materialize and validate the final boundary-rotation features

Repeat step 2 with `manifests/flare24_boundary_rotations_v1.json` as the rotation
manifest and these immutable output paths:

```text
D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_rotation_v1
D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_rotation_graph_v1
D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_rotation_frontier_v1
manifests/flare24_boundary_pot_rotation_v1.json
```

Then validate:

```powershell
uv run flightdelaybench-validate-boundary-pot `
  --manifest manifests/flare24_boundary_pot_rotation_v1.json `
  --target-census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --output reports/validation/flare24_boundary_pot_rotation_v1.json
```

## 7. Run the locked retrospective study

```powershell
uv run flightdelaybench-run-boundary-pot `
  --protocol configs/flare24_boundary_pot_v1.toml `
  --boundary-manifest manifests/flare24_boundary_pot_rotation_v1.json `
  --boundary-validation reports/validation/flare24_boundary_pot_rotation_v1.json `
  --boundary-rotation-gate reports/validation/flare24_boundary_rotation_smoke_v1.json `
  --parent-report reports/experiments/flare24_ccrth_2025_retrospective_v5.json `
  --meta-report reports/experiments/flare24_ccrth_metastack_v1.json `
  --baseline-selection-report reports/experiments/flare24_2024_selection_v2.json `
  --baseline-audit-report reports/experiments/flare24_2025_audit_v1_recovered.json `
  --census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 `
  --recent-dir D:/FlightDelayAdvisorResearchData/derived/census_recent_prior_day_v2 `
  --flight-recent-dir D:/FlightDelayAdvisorResearchData/derived/census_flight_recent_v2_duckdb `
  --graph-dir D:/FlightDelayAdvisorResearchData/derived/census_graph_clsgmp_v1 `
  --weather-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_features_v1 `
  --rotation-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_rotation_features_v5 `
  --capacity-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_ccrth_v3 `
  --boundary-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_boundary_pot_rotation_v1 `
  --run-dir D:/FlightDelayAdvisorResearchData/runs/flare24_boundary_pot_v1 `
  --output reports/experiments/flare24_boundary_pot_v1.json
```

The study writes `method_lock.json` before loading new training/evaluation outcomes and
records that earlier aggregate 2025 evidence was already known.

## 8. Validate, report, and validate publication assets

```powershell
uv run flightdelaybench-validate-boundary-pot-study `
  --report reports/experiments/flare24_boundary_pot_v1.json `
  --output reports/validation/flare24_boundary_pot_study_v1.json

uv run flightdelaybench-report-boundary-pot `
  --report reports/experiments/flare24_boundary_pot_v1.json `
  --validation reports/validation/flare24_boundary_pot_study_v1.json `
  --output-dir reports/figures/bcpotr_v1 `
  --manifest manifests/flare24_boundary_publication_assets_v1.json

uv run flightdelaybench-validate-boundary-pot-publication `
  --manifest manifests/flare24_boundary_publication_assets_v1.json `
  --output reports/validation/flare24_boundary_publication_assets_v1.json
```

## 9. Audited recovery used for the retained v6 run

The actual v6 run completed every predictive artifact but failed while bootstrapping a
descriptive contrast whose unexposed group contained only four rows. Do not rerun or
overwrite it. The following command verifies the exact 25-file inventory, replays
purged-forward calibration and ensemble selection, checks persisted probabilities, and
recomputes the report without fitting a model or regenerating a prediction:

```powershell
uv run flightdelaybench-recover-boundary-pot-study `
  --source-run-dir D:/FlightDelayAdvisorResearchData/runs/flare24_boundary_pot_v6 `
  --failure-manifest manifests/failures/flare24_boundary_pot_study_v6_saturated_contrast.json `
  --recovery-run-dir D:/FlightDelayAdvisorResearchData/runs/flare24_boundary_pot_v6_recovery_v1 `
  --output reports/experiments/flare24_boundary_pot_v6_recovered.json `
  --bootstrap-repetitions 2000 --seed 20260903
```

Validate and render that immutable report with:

```powershell
uv run flightdelaybench-validate-boundary-pot-study `
  --report reports/experiments/flare24_boundary_pot_v6_recovered.json `
  --output reports/validation/flare24_boundary_pot_study_v6_recovered_v2.json

uv run flightdelaybench-report-boundary-pot `
  --report reports/experiments/flare24_boundary_pot_v6_recovered.json `
  --validation reports/validation/flare24_boundary_pot_study_v6_recovered_v2.json `
  --output-dir reports/figures/bcpotr_v6_recovered_v2 `
  --manifest manifests/flare24_boundary_publication_assets_v6_recovered_v2.json

uv run flightdelaybench-validate-boundary-pot-publication `
  --manifest manifests/flare24_boundary_publication_assets_v6_recovered_v2.json `
  --output reports/validation/flare24_boundary_publication_assets_v6_recovered_v2.json
```

The recovered report and the recovery run manifest are byte-identical. The study
validator additionally verifies that no predictive artifact was recreated and that the
maximum annual ensemble reproduction error is below `3.5e-8`.
