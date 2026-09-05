# Architecture and extension contract

[Documentation index](README.md) · [Research roadmap](RESEARCH_ROADMAP.md)

FlightDelayAdvisor has two application boundaries: the research package under
`src/flightdelaybench/`, and the preserved Streamlit application at the repository
root. The legacy app is not the interface to the corrected research pipeline.
Existing module and artifact locations remain stable so historical commands and
hash-bound records continue to resolve.

## Data and evidence flow

```text
Raw BTS / forecast responses / dated airport resources
                  |
          source hashes and lineage
                  |
          +-------+-----------------------+
          |                               |
  Historical proxy track          Corrected timestamp track
  census and feature generations  reviewed event/publication/receipt evidence
          |                               |
  FLARE / CC-RTH / BC-POT-R        cutoff dataset and label-availability checks
          |                               |
  preserved models/predictions    seasonal forward development experiments
          |                               |
  independent result validation  blocked until authentic inputs exist
          |
  reports, uncertainty, failures and qualified public claims
```

The two tracks share modelling utilities, not evidence eligibility. A historical
feature cache cannot become a corrected input by being renamed or rehashed.

## Package map

| Boundary | Representative modules | Responsibility |
|---|---|---|
| Acquisition and normalization | `acquisition`, `bts`, `census_normalization`, `forecast_acquisition`, `flare_weather_acquisition` | Create source/partition records; retain original bytes |
| Historical context | `recent`, `census_recent`, `census_graph`, `boundary_context` | Earlier-date and schedule-network features under their recorded proxy contracts |
| Weather and latent rotations | `flare_weather`, `flare_features`, `flare_rotation_features` | Weather transforms and uncertain schedule-compatible predecessors |
| Airport resources and boundary models | `flare_capacity_features`, `flare_capacity_study`, `flare_capacity_factorized`, `flare_capacity_metastack`, `flare_boundary_study` | Named historical experiment families and their ablations |
| Corrected input boundary | `cutoff_history`, `cutoff_dataset` | Timestamp order, as-of eligibility, endpoint consistency and hash-bound dataset admission |
| Corrected experiment infrastructure | `cutoff_experiments` | Matched samples, seasonal folds, model trials and paired comparisons; not a completed full campaign |
| Independent checks and reporting | Generation-specific `*_validation` and `*_reporting` modules; `tools/validate_documentation.py` | Reconstruct recorded contracts and keep narrative tables tied to evidence |
| Repository-facing navigation | `tools/build_experiment_index.py`, `docs/RESULT_LINEAGE.md`, `paper/` | Discover results and their claim limits without loading large data or models |

The entry points and optional dependencies are declared in
[pyproject.toml](../pyproject.toml). Use the appropriate CLI's `--help`; old absolute
paths are not a portable configuration contract.

## Adding a component

1. Define the source, prediction horizon, target population and required timing
   evidence before implementing a feature. Follow the [data contract](CUTOFF_DATA_CONTRACT.md).
2. Register its availability and keep hindsight-only variables in an explicitly
   separate diagnostic track. Publication, revision and receipt semantics are not
   interchangeable with event time.
3. Add fixture tests for row identity, endpoint masks, missing support, chronology,
   partition edges and behaviour under contradictory labels or late arrivals.
4. Compare on identical flight IDs, folds, estimator settings and budgets. Keep
   full eligible airport context even when the estimator uses a nested row sample.
5. Write into a new run directory with predictions, resolved configuration,
   hashes, resource measurements and failure records. Never overwrite an old run.
6. Add independent validation and an experiment-index entry before promoting a
   result into the current narrative. Keep a no-change incumbent option.

Prospective receipt logging, additional forecast sources, learning curves, later
finalist repeats and conditional capacity/recovery modelling remain distinct future
work. Their design or partial infrastructure is not an implemented research result.

## Trust and execution boundaries

Downloaded data must be verified before use. Serialized model formats such as
Joblib/Pickle can execute code; load only trusted artifacts matching their manifests.
Source-only tests use synthetic fixtures and small reports. Full evidence checks,
model inference, data acquisition and training are separate opt-in operations.
No routine documentation or source check may open confirmation outcomes.
