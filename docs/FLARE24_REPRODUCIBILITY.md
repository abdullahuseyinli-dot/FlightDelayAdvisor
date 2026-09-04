# FLARE-24 reproducibility guide

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

This guide reproduces the release candidate without treating local absolute paths as
portable identifiers. SHA-256 values, row counts, schemas, and frozen choices in the
manifests are authoritative. All builders are create-only: choose a new output or run
identifier, and never remove a prior run to make a command succeed.

## 1. Supported environment

- Python 3.11 or 3.12
- Windows, Linux, or macOS for source-only checks
- A large external workspace for the BTS, forecast, feature, prediction, and model
  artifacts; use the manifest `bytes` fields to budget storage
- CatBoost for the FLARE models and Matplotlib/Seaborn for publication figures

The measured audit environment used Python 3.11.9, CatBoost 1.2.10, NumPy 2.4.6,
Pandas 2.3.3, SciPy 1.17.1, and Joblib 1.6.0. The recorded machine had an Intel
Core i9-13900HX (24 cores/32 logical processors), 32 GiB RAM, and an RTX 4060 Laptop
GPU; the CatBoost study code does not claim GPU training.

## 2. Install and run the source-only gate

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,models,plots]"

.\.venv\Scripts\python.exe -m compileall -q app.py src tools tests
.\.venv\Scripts\python.exe -m ruff check src/flightdelaybench tests
.\.venv\Scripts\python.exe -m mypy src/flightdelaybench
.\.venv\Scripts\python.exe -m pytest -q -m "not integration and not slow and not confirmation"
.\.venv\Scripts\python.exe tools/validate_repository.py
```

The source-only gate does not open external Parquet/model artifacts or any 2026
outcome.

## 3. External artifact layout

The commands below use `$DataRoot` as an operator-selected directory. The successful
run used the following logical layout; another machine may use different absolute
paths.

```powershell
$DataRoot = "D:\FlightDelayAdvisorResearchData"
$Census = "$DataRoot\normalized_top100_census_v2"
$Recent = "$DataRoot\derived\census_recent_prior_day_v2"
$FlightRecent = "$DataRoot\derived\census_flight_recent_v2_duckdb"
$Graph = "$DataRoot\derived\census_graph_clsgmp_v1"
$WeatherCube = "$DataRoot\derived\flare24_weather_cube_v2"
$WeatherFeatures = "$DataRoot\derived\flare24_features_v1"
$RotationFeatures = "$DataRoot\derived\flare24_rotation_features_v5"
```

The repository manifests map every external artifact to its SHA-256 and size. A path
may be remapped, but a hash mismatch is not a path-remapping issue and must fail.

## 4. Acquire and build the point-in-time feature assets

The raw BTS census, closed-left history, and graph builders are documented by their
own CLI help and manifests. The FLARE-specific stages are:

```powershell
flightdelaybench-acquire-flare24-weather `
  --airport-catalog configs/top100_airports_2024.json `
  --output-dir "$DataRoot\raw_openmeteo_previous_runs\flare24_day2_v1" `
  --manifest manifests/flare24_openmeteo_gfs_day2_NEW.json `
  --lead-days 2 --start-year 2024 --end-year 2025 --workers 4 --retries 5 --resume

flightdelaybench-build-flare24-weather `
  --acquisition-manifest manifests/flare24_openmeteo_gfs_day2_NEW.json `
  --output-dir "$DataRoot\derived\flare24_weather_cube_NEW" `
  --manifest manifests/flare24_weather_cube_NEW.json

flightdelaybench-build-flare24-runways `
  --archive "$DataRoot\raw_faa_nasr\28DaySubscription_Effective_2024-10-03.zip" `
  --airport-catalog configs/top100_airports_2024.json `
  --output "$DataRoot\derived\flare24_runways_NEW.json" `
  --manifest manifests/flare24_runways_NEW.json `
  --minimum-runway-length-ft 4000

flightdelaybench-build-flare24-features `
  --census-dir $Census `
  --weather-manifest manifests/flare24_weather_cube_NEW.json `
  --airport-catalog configs/top100_airports_2024.json `
  --runway-catalog "$DataRoot\derived\flare24_runways_NEW.json" `
  --output-dir "$DataRoot\derived\flare24_features_NEW" `
  --manifest manifests/flare24_features_NEW.json `
  --years 2024 2025

flightdelaybench-build-flare24-rotations `
  --census-dir $Census `
  --airport-catalog configs/top100_airports_2024.json `
  --recent-dir $Recent `
  --recent-manifest manifests/census_recent_prior_day_v2.json `
  --output-dir "$DataRoot\derived\flare24_rotation_features_NEW" `
  --model-dir "$DataRoot\runs\flare24_rotation_NEW\models" `
  --manifest manifests/flare24_rotation_features_NEW.json `
  --target-years 2024 2025
```

`NEW` is intentional. Do not point these commands at the completed v1/v2/v5 outputs.
Acquisition may resume a named incomplete download only when its manifest records the
partial state.

## 5. Validate materialized assets

The successful release assets can be reopened with:

```powershell
flightdelaybench-validate-flare24 `
  --weather-manifest manifests/flare24_weather_cube_v2.json `
  --feature-manifest manifests/flare24_features_v1.json `
  --rotation-manifest manifests/flare24_rotation_features_v5.json `
  --output reports/validation/flare24_assets_RECHECK.json
```

The validator checks self-hashes, external file hashes, row counts, period alignment,
schemas, finite values, issue-time cutoffs, coverage, latent predecessor capacity,
and explicit leakage declarations. A source-only Zenodo checkout cannot pass this
external-data gate until the manifest-bound artifacts have been restored.

## 6. Reproduce 2024 selection

Use a new run directory and report path:

```powershell
$SelectionRun = "$DataRoot\runs\flare24_2024_selection_REPLICATION"

flightdelaybench-run-flare24 selection `
  --census-dir $Census `
  --recent-dir $Recent `
  --flight-recent-dir $FlightRecent `
  --graph-dir $Graph `
  --weather-feature-dir $WeatherFeatures `
  --rotation-feature-dir $RotationFeatures `
  --run-dir $SelectionRun `
  --output reports/experiments/flare24_2024_selection_REPLICATION.json `
  --census-manifest manifests/census_top100_2018_2025_v2.json `
  --protocol configs/flare24_v1.toml `
  --recent-manifest manifests/census_recent_prior_day_v2.json `
  --flight-recent-manifest manifests/census_flight_recent_v2_duckdb.json `
  --graph-manifest manifests/census_graph_clsgmp_v1.json `
  --weather-feature-manifest manifests/flare24_features_v1.json `
  --rotation-feature-manifest manifests/flare24_rotation_features_v5.json `
  --rows-per-train-month 125000 `
  --validation-limit 250000 `
  --bootstrap-repetitions 2000 `
  --seed 20260903
```

The recorded selection took 2,321.81 seconds on the measured machine. Model training
may vary numerically across library/platform versions; any replication must publish
its own hashes rather than overwrite or impersonate the frozen run.

To lock a replicated selection, use a new lock path:

```powershell
flightdelaybench-run-flare24 lock `
  --selection-report reports/experiments/flare24_2024_selection_REPLICATION.json `
  --output manifests/flare24_method_lock_REPLICATION.json
```

The canonical release lock remains `manifests/flare24_method_lock_v1.json`.

## 7. Reproduce the frozen 2025 audit

```powershell
$AuditRun = "$DataRoot\runs\flare24_2025_audit_REPLICATION"

flightdelaybench-run-flare24 audit `
  --method-lock manifests/flare24_method_lock_v1.json `
  --census-dir $Census `
  --recent-dir $Recent `
  --flight-recent-dir $FlightRecent `
  --graph-dir $Graph `
  --weather-feature-dir $WeatherFeatures `
  --rotation-feature-dir $RotationFeatures `
  --run-dir $AuditRun `
  --output reports/experiments/flare24_2025_audit_REPLICATION.json `
  --bootstrap-repetitions 2000 `
  --seed 20260903
```

Do not use the recovery command on an ordinary successful audit. It exists only for
the retained v1 finalizer failure where all monthly predictions were complete and
hash-verified. For that exact run, the create-only recovery was:

```powershell
flightdelaybench-recover-flare24-audit `
  --method-lock manifests/flare24_method_lock_v1.json `
  --failed-run-dir "$DataRoot\runs\flare24_2025_audit_v1" `
  --census-dir $Census `
  --failure-record manifests/failures/flare24_2025_audit_v1_float32_simplex_tolerance.json `
  --output reports/experiments/flare24_2025_audit_v1_recovered.json `
  --bootstrap-repetitions 2000 `
  --seed 20260903
```

That output already exists; the displayed command documents provenance and will
correctly refuse to overwrite it.

## 8. Rebuild analysis and figures

Choose new paths for a replication:

```powershell
flightdelaybench-analyse-flare24-ablation `
  --audit-report reports/experiments/flare24_2025_audit_v1_recovered.json `
  --output reports/experiments/flare24_2025_nested_ablation_REPLICATION.json `
  --table reports/figures/flare24_REPLICATION/flare24_2025_nested_paired_intervals.csv

flightdelaybench-report-flare24 `
  --selection-report reports/experiments/flare24_2024_selection_v2.json `
  --audit-report reports/experiments/flare24_2025_audit_v1_recovered.json `
  --output-dir reports/figures/flare24_REPLICATION `
  --manifest manifests/flare24_publication_bundle_REPLICATION.json
```

The canonical v2 bundle supersedes v1 because the first visualization duplicated
curves and used an ambiguous alignment label. The supersession record is preserved.

## 9. Verify the complete lightweight evidence chain

```powershell
flightdelaybench-validate-flare24 `
  --method-lock manifests/flare24_method_lock_v1.json `
  --study-report reports/experiments/flare24_2025_audit_v1_recovered.json `
  --publication-bundle manifests/flare24_publication_bundle_v2.json `
  --confirmation-lock manifests/confirmation_lock_v1.json `
  --nested-ablation reports/experiments/flare24_2025_nested_ablation_v1.json `
  --output reports/validation/flare24_release_RECHECK.json
```

The checked-in canonical validation records are under `reports/validation/`. The
confirmation lock freezes analysis; it does not itself authorize or report an outcome.

## 10. Build curated distributions

Use a new output directory so prior distributions remain intact:

```powershell
python -m build --outdir dist/flare24-0.1.0rc1
python -m zipfile -l dist/flare24-0.1.0rc1/flightdelaybench-0.1.0rc1-py3-none-any.whl
tar -tf dist/flare24-0.1.0rc1/flightdelaybench-0.1.0rc1.tar.gz
```

The wheel must contain all `flightdelaybench.flare_*` modules. The source archive is
curated to code, tests, configurations, documentation, lightweight manifests/reports,
and publication figures. It must not contain raw BTS/weather archives, derived
Parquet, external run directories, or frozen model binaries.

## 11. Fresh-wheel inference smoke

Create a fresh environment outside the repository, install only the built wheel and
its model extra, and invoke the installed console command. The frozen selected model
hashes are shown explicitly:

```powershell
$FreshEnv = "$DataRoot\fresh_envs\flare24_0_1_0rc1"
python -m venv $FreshEnv
& "$FreshEnv\Scripts\python.exe" -m pip install `
  "dist/flare24-0.1.0rc1/flightdelaybench-0.1.0rc1-py3-none-any.whl[models]"

$FrozenModels = "$DataRoot\runs\flare24_2024_selection_v2\models"
& "$FreshEnv\Scripts\flightdelaybench-smoke-flare24.exe" `
  --census-dir $Census `
  --recent-dir $Recent `
  --flight-recent-dir $FlightRecent `
  --graph-dir $Graph `
  --weather-feature-dir $WeatherFeatures `
  --rotation-feature-dir $RotationFeatures `
  --delay-model "$FrozenModels\rotation_structural_delay.joblib" `
  --delay-model-sha256 8b5ba70969a45e5ac1359d837fdaab459357e6b2b828f2873941a3b9e16cae3d `
  --cancellation-model "$FrozenModels\rotation_structural_cancellation.joblib" `
  --cancellation-model-sha256 81a0ce449053549be7fbd4b3959744e9e41134e58b964482eaaf5e83a17a2ab4 `
  --year 2024 --month 10 --limit 128 `
  --output reports/validation/flare24_fresh_wheel_smoke_REPLICATION.json
```

This smoke test verifies installed-package imports, frozen model hashes, feature/model
compatibility, finite probability bounds, and the three-state simplex. It does not
score outcomes or open 2026.

## 12. Reproduction expectations

A valid replication must report its environment, wall-clock times, new run IDs,
output hashes, and any numerical differences. It must preserve failures and must not
substitute another cohort, weather vintage, horizon, endpoint, or random split while
using the FLARE-24 name. Exact release evidence identities are listed in
`docs/RESULTS.md` and the release manifest.
