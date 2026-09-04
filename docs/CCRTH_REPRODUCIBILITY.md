# CC-RTH reproducibility guide

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

This guide reproduces the Capacity-Conditioned Resource-Time Flight Hypergraph extension without opening the locked 2026 confirmation outcomes. Run commands from the repository root in Python 3.11 or 3.12.

## Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,models,weather,plots]"
```

The measured run uses CatBoost GPU training. Set `task_type="CPU"` only for a non-comparable diagnostic run and give it a new run identifier. Never overwrite the released run.

## Required parent assets

CC-RTH consumes the already checksummed FLARE-24 census, closed-left historical features, CL-SGMP features, cutoff-coherent weather features, and latent-rotation features. Their authoritative manifests remain:

- `manifests/census_top100_2018_2025_v2.json`
- `manifests/census_recent_prior_day_v2.json`
- `manifests/census_flight_recent_v2_duckdb.json`
- `manifests/census_graph_clsgmp_v1.json`
- `manifests/flare24_features_v1.json`
- `manifests/flare24_rotation_features_v5.json`

The full derived tables are intentionally outside Git. Preserve the absolute or remapped roots in the run manifest.

## 1. Build the static resource catalogue

Use the FAA NASR archive already bound to the FLARE-24 release. Never infer a missing runway or annual-operations value.

```bash
flightdelaybench-build-ccrth-resources \
  --archive D:/FlightDelayAdvisorResearchData/raw_faa_nasr/effective_2024-10-03/28DaySubscription_Effective_2024-10-03.zip \
  --airport-catalog data/external/forecast24_openmeteo_gfs_v1/airport_catalog.json \
  --output data/external/flare24_ccrth_resources_nasr_20241003_v2.json \
  --manifest manifests/flare24_ccrth_resources_nasr_20241003_v2.json
```

The released catalogue contains physical runway records and runway ends. `max_parallel_runways` is a geometry proxy, not an operationally simultaneous capacity claim.

## 2. Materialize features and sparse hypergraphs

Use a new output suffix if any listed path already exists.

```bash
flightdelaybench-build-ccrth \
  --census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 \
  --weather-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_features_v1 \
  --weather-feature-manifest manifests/flare24_features_v1.json \
  --airport-catalog data/external/forecast24_openmeteo_gfs_v1/airport_catalog.json \
  --resource-catalog data/external/flare24_ccrth_resources_nasr_20241003_v2.json \
  --resource-catalog-manifest manifests/flare24_ccrth_resources_nasr_20241003_v2.json \
  --rotation-manifest manifests/flare24_rotation_features_v5.json \
  --output-dir D:/FlightDelayAdvisorResearchData/derived/flare24_ccrth_v3 \
  --graph-dir D:/FlightDelayAdvisorResearchData/derived/flare24_ccrth_graph_v3 \
  --frontier-dir D:/FlightDelayAdvisorResearchData/derived/flare24_ccrth_frontier_v3 \
  --manifest manifests/flare24_ccrth_features_v3.json \
  --target-years 2024 2025
```

This stage projects every BTS partition to schedule-only columns before computation. The manifest must report no outcome columns, no target tail number, prior-year frontier lineage, source checksums, per-period graph counts, and source-code provenance.

Validate it before modeling:

```bash
flightdelaybench-validate-ccrth \
  --capacity-manifest manifests/flare24_ccrth_features_v3.json \
  --output reports/validation/flare24_ccrth_features_v3.json
```

## 3. Run the nested chronological study

The exact old FLARE-24 final probabilities are reused rather than approximately retrained. This makes every reported increment a same-flight comparison to the previously published result.

```bash
flightdelaybench-run-ccrth \
  --protocol configs/flare24_ccrth_v1.toml \
  --baseline-selection-report reports/experiments/flare24_2024_selection_v2.json \
  --baseline-audit-report reports/experiments/flare24_2025_audit_v1_recovered.json \
  --capacity-manifest manifests/flare24_ccrth_features_v3.json \
  --capacity-validation reports/validation/flare24_ccrth_features_v3.json \
  --census-dir D:/FlightDelayAdvisorResearchData/normalized_top100_census_v2 \
  --recent-dir D:/FlightDelayAdvisorResearchData/derived/census_recent_prior_day_v2 \
  --flight-recent-dir D:/FlightDelayAdvisorResearchData/derived/census_flight_recent_v2_duckdb \
  --graph-dir D:/FlightDelayAdvisorResearchData/derived/census_graph_clsgmp_v1 \
  --weather-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_features_v1 \
  --rotation-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_rotation_features_v5 \
  --capacity-feature-dir D:/FlightDelayAdvisorResearchData/derived/flare24_ccrth_v3 \
  --run-dir D:/FlightDelayAdvisorResearchData/runs/flare24_ccrth_v1 \
  --output reports/experiments/flare24_ccrth_2025_retrospective_v1.json \
  --bootstrap-repetitions 2000 \
  --seed 20260903
```

The runner trains only augmented candidates. It reads the frozen 2024 FLARE selection for the exact reference, uses January–August 2024 for model development, embargoes August 30–31 when selecting the iteration count against September, and refits fixed-iteration models on all January–September rows. October 1–2 form the model-to-calibration embargo. Calibration and blend selection then use forward Q4 folds with a two-date embargo at every boundary; final calibrators fit through December 31. The runner writes `method_lock.json` before it opens the pre-existing 2025 audit probabilities or 2025 outcome-bearing feature frames. Primary scoring spans January 3–December 29: the first two dates form the year-boundary embargo and the final two are excluded because their +2-day graph context would require unopened 2026 schedules. The 361 retained complete-context dates form the uncertainty clusters; complete-year point values are descriptive comparators only. The report explicitly labels 2025 retrospective rather than blind confirmation.

If execution fails after all eight final 2024 models are durably written but before
the method lock or any 2025 outcome is opened, a later run may reuse those models
with `--recovery-model-record`. Recovery is accepted only when a retained failure
record binds the unchanged protocol and capacity manifest, declares the closed
2025/2026 boundary, contains all eight candidate/task artifacts, and every byte
count, SHA-256 digest, model feature contract, task, and CatBoost tree count is
reverified. The new run still reconstructs the frozen training sample and resumes
the full Q4 selection, method-lock, and 2025 evaluation sequence. This is execution
recovery, not model selection or a new result; both the source failed run and the
recovery record are included in the final provenance.

Create that record by re-inspecting the failed run rather than transcribing hashes:

```bash
flightdelaybench-record-ccrth-recovery \
  --failed-run-dir D:/FlightDelayAdvisorResearchData/runs/flare24_ccrth_v2 \
  --protocol configs/flare24_ccrth_v1.toml \
  --capacity-manifest manifests/flare24_ccrth_features_v3.json \
  --expected-study-report reports/experiments/flare24_ccrth_2025_retrospective_v2.json \
  --output manifests/failures/flare24_ccrth_study_v2_storage_disconnect.json \
  --failure-summary "Storage disconnected during a 2024 selection-prediction write."
```

The recorder refuses a method lock, completed report, calibrator, or any 2025-named
artifact; reloads and checks every model; verifies task, nested feature contract,
tree count, and frozen CatBoost parameters; hashes retained partials; and self-hashes
the evidence record. Supply the resulting path to a fresh versioned run with
`--recovery-model-record`; never reuse the failed run directory itself.

Validate the completed evidence package:

```bash
flightdelaybench-validate-ccrth \
  --study-report reports/experiments/flare24_ccrth_2025_retrospective_v1.json \
  --output reports/validation/flare24_ccrth_2025_retrospective_v1.json
```

Generate checksum-bound CSV tables and publication figures in a new directory:

```bash
flightdelaybench-report-ccrth \
  --report reports/experiments/flare24_ccrth_2025_retrospective_v1.json \
  --output-dir reports/figures/ccrth_v1 \
  --manifest manifests/flare24_ccrth_publication_assets_v1.json
```

## 4. Source checks

```bash
python -m compileall -q src/flightdelaybench tests tools
python -m ruff check src/flightdelaybench tests
python -m mypy src/flightdelaybench
python -m pytest -q -m "not integration and not confirmation"
python -m build
```

The repository also contains older standalone scripts under `src/` with legacy lint debt. Release lint is scoped to the packaged `src/flightdelaybench` tree and tests; no legacy file is silently reformatted as part of CC-RTH.

## Failure and rerun policy

Never delete a failed or partial graph run. Record its exception, completed artifacts, scientific impact, and remediation in `manifests/failures/`; choose a fresh versioned directory; rerun; and admit only a complete checksum-valid manifest to modeling. Partial model sets are evidence only. A complete eight-model 2024-only set may be recovered solely under the audited contract above; any checksum, protocol, feature-contract, task, iteration, temporal-boundary, or completeness mismatch forces a fresh fit. Missing weather, static resources, or optional constraints must remain explicit missingness or documented fallback—not invented measurements.
