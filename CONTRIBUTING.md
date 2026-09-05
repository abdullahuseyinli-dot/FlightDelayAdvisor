# Contributing

FlightDelayBench accepts bug fixes, validation improvements, additional temporal
baselines, and new prediction-horizon adapters. Changes that improve a headline
metric by weakening the information cutoff, dropping difficult negatives, or using a
random split are not comparable contributions.

## Development workflow

Follow the [research standards](docs/RESEARCH_STANDARDS.md),
[current benchmark contract](docs/BENCHMARK_CARD.md) and
[cutoff-correction plan](docs/CUTOFF_RESEARCH_PLAN.md). The previous release
candidate is withdrawn; source checks alone cannot restore its timing claims.
Use the [architecture and extension contract](docs/ARCHITECTURE.md) to place new
work and the [roadmap](docs/RESEARCH_ROADMAP.md) to identify prerequisites.

1. Create a focused branch and install the `dev` extra.
2. Register every candidate predictor in `flightdelaybench.contracts` with its
   earliest valid horizon and source.
3. Add tests for row alignment, temporal cutoff, missingness, and outcome invariance
   where applicable.
4. Use a new run identifier. Never overwrite or delete completed evidence.
5. Run unit tests, Ruff, mypy, manifest validation, and the clean-environment smoke
   test before requesting review.
6. Report negative and failed experiments alongside improvements.

Use [USAGE.md](docs/USAGE.md) for non-destructive commands. Current result tables
are checked against immutable reports by `tools/validate_documentation.py`.
Documentation changes must preserve distinctions between historical scores,
software validation and uncompleted real-data experiments. Report accuracy
changes in absolute percentage points; do not substitute relative changes.

Every new experiment report must appear in the generated
[experiment index](docs/EXPERIMENT_INDEX.md). Check it with
`python tools/build_experiment_index.py --check`; when adding research records,
create a new versioned inventory/index and update their current references together.
Do not rewrite an older immutable index or conceal failed/blocked records.

Do not commit restricted upstream data. Derived public-release artifacts must carry a
source description, construction code, row counts, checksums, and applicable licence
or redistribution constraints.

## Claims

Pull requests must distinguish development, selection, retrospective, and
confirmatory results. Do not claim causality, airline safety, demographic fairness,
real-time feed validity, or state of the art without evidence designed for that claim.
