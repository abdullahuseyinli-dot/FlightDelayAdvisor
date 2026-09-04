# FlightDelayAdvisor

Probabilistic flight-disruption research: what can schedules, weather forecasts and
airport-network context tell us about a flight a day before departure?

[![CI](https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor/actions/workflows/tests.yml/badge.svg)](https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor/actions/workflows/tests.yml)

> **Research status:** the previous release candidate is withdrawn following a
> prediction-cutoff audit. The scores below are historical retrospective proxy
> results, not validated T−24 operational performance. Corrected software has been
> tested; corrected real-data experiments have not been completed.
> [Status and correction](docs/PROJECT_STATUS.md).

The repository contains a three-state benchmark (on time, arrival delay of at least
15 minutes, cancelled), preserved experiments and negative results, a timestamp-aware
evaluation pipeline, and a separate legacy Streamlit application. It is intended for
reproducibility and research, not live travel advice.

## Main findings

Weather supplied most of the historical improvement over the schedule/history
baseline. Airport-resource models helped cancellation probability estimates, but
adding small-airport schedule context did not deliver a large accuracy gain.
The later audit identified a more fundamental problem: excluding the target
operating day does not ensure that a historical outcome was available at each
flight's scheduled-departure-minus-24-hours cutoff.

The latest matched comparison covers **5,754,266 flights**, January 3–December 29,
2025, with 361 date clusters. All rows below use the same observed joint outcomes.
Accuracy is standard three-class argmax; lower log loss and Brier are better.

| Method | Accuracy (%) | Joint log loss | Multiclass Brier |
|---|---:|---:|---:|
| Schedule/history baseline | 76.7493 | 0.555921 | 0.337727 |
| FLARE-24 structural rotation | 77.3289 | 0.539577 | 0.327218 |
| Regularized cancellation meta-stack | 77.3308 | 0.537377 | 0.326911 |
| Boundary-complete model | 77.3673 | 0.538733 | 0.326536 |
| Q4-selected boundary ensemble | 77.3512 | 0.538449 | 0.326679 |

The selected boundary ensemble gains **0.0203 percentage points** over the
meta-stack (95% date-cluster interval: 0.0060 to 0.0349 points), while log loss
worsens. Neither the +5 nor +10 percentage-point target was met. The boundary
comparison also changed a CatBoost interaction setting, so it does not isolate
the benefit of adding small airports. The best 2025 point estimate is not
substituted for the method selected on 2024.

On-time flights account for 76.5222% of this cohort; accuracy alone obscures weak
minority-class recall. See [current results](docs/CURRENT_RESULTS.md) for class
metrics, uncertainty, negative findings and exact source reports.

## What is included

- **FLARE-24:** schedule/history baselines, fixed-lead archived weather, aviation
  transforms, latent schedule-compatible rotations and reconciliation ablations.
- **CC-RTH:** capacity-conditioned airport resource-time features, task-factorized
  models and a regularized cancellation stack.
- **BC-POT-R:** boundary-complete schedule context from 255 additional airports,
  probabilistic operations-twin features and residual/ensemble experiments.
- **Cutoff correction:** explicit event, source-publication, feature and label
  availability checks; matched temporal experiment infrastructure; a small issued-TAF
  archive feasibility pilot. The real-data study remains pending.

Method names describe investigated frameworks, not established novel or
state-of-the-art inventions. Null, harmful, failed and superseded runs are retained.

## Quick start

Python 3.11 or 3.12. From a source checkout:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,models,plots]"

python -m pytest -q -m "not integration and not slow and not confirmation"
python tools/validate_repository.py
```

These checks do not download research data or retrain the benchmark. Model inference
and full reconstruction require external artifacts. Some legacy integration scripts
write fixed report paths and must not be run over the preserved evidence.

See [usage](docs/USAGE.md) for lint, typing, optional models, packaging, inference and
safe reproduction scopes. See [legacy application](docs/LEGACY_APPLICATION.md) for
Git LFS setup, the Streamlit demo and its separate sampled results.

## Documentation

| Start here | Contents |
|---|---|
| [Benchmark card](docs/BENCHMARK_CARD.md) | Population, endpoints, temporal splits and comparison rules |
| [Current results](docs/CURRENT_RESULTS.md) | Matched scores, absolute changes and interpretation |
| [Project status](docs/PROJECT_STATUS.md) | Completed work, withdrawn claims and outstanding gates |
| [Data card](docs/DATA_CARD.md) / [model card](docs/MODEL_CARD.md) | Inputs, coverage, model lineage and intended use |
| [Artifacts](docs/ARTIFACTS.md) | Evidence locations, external dependencies and reconstruction limits |
| [Limitations](docs/LIMITATIONS.md) | Timing, missing operational state, confounding and scope |
| [Documentation index](docs/README.md) | Technical reports, historical protocols and correction plan |

## Repository layout

```text
src/flightdelaybench/   Research pipelines, models and independent validators
configs/               Versioned experiment protocols and frozen airport cohort
manifests/             Source/derivative hashes, method locks and failure records
reports/               Historical scores, diagnostics, figures and validation evidence
data/external/         Curated source metadata and small issued-weather pilot
docs/                  Benchmark documentation, methods and research status
tests/                 Synthetic/unit tests and opt-in artifact-backed checks
tools/                 Source, documentation, package and evidence validation
app.py + config/       Legacy Streamlit application and artifact paths
```

Large datasets and research models are not bundled as ordinary Git files. An archive
containing code and reports is not a self-contained full-data reproduction package.

## Citation and licensing

Citation metadata is in [CITATION.cff](CITATION.cff). No Zenodo DOI is assigned here;
the [Zenodo metadata](.zenodo.json) is a draft, not evidence of a deposit. A new release
requires the [release checklist](docs/RELEASE_CHECKLIST.md).

Software is [MIT licensed](LICENSE). Upstream data and dependencies retain their own
terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Contributions should follow
[CONTRIBUTING.md](CONTRIBUTING.md) and the [research standards](docs/RESEARCH_STANDARDS.md).
