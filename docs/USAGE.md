# Usage and reproducibility

[Documentation index](README.md) · [Artifact inventory](ARTIFACTS.md)

## Choose the appropriate scope

| Scope | Requirements | What it checks |
|---|---|---|
| Source checkout | Python 3.11/3.12, source and small evidence files | Unit tests, documentation and stored report integrity |
| Historical inference | Research models and exact external feature artifacts | Packaging, model loading and class-probability semantics |
| Historical reconstruction | Raw data, provider archives, storage and compute | Reproduction of a named historical proxy experiment |
| Corrected research | Reviewed feature/label availability ledger and new run paths | New timing-valid development results; not yet available |
| Legacy application | Git LFS data/models and app dependencies | Historical Streamlit demo, not the FLARE research pipeline |

## Install for source development

For PowerShell, clone current main without downloading large LFS objects:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --branch main --single-branch https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd FlightDelayAdvisor
```

For a POSIX shell, prefix that clone command with `GIT_LFS_SKIP_SMUDGE=1` for that
process only. The default `main` branch contains the current research work.
Skip cloning when using an existing checkout; preserve its local changes.
See [versioning](VERSIONING.md) for the preserved legacy branch and older handoffs.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,models,plots]"
```

`pyproject.toml` defines dependency ranges; `uv.lock` records a resolved environment.
The pip command above uses those ranges, not the exact lock. For lock-based source
development, use an installed `uv` and a separate project environment:

```bash
uv sync --locked --extra dev --extra models --extra plots
uv run --locked python tools/validate_repository.py
```

For an existing development checkout, `uv lock --check --offline` checks lock
consistency without changing it. Do not silently update the lock to reproduce
a historical environment: use the versions recorded in that run's report.

The `frontier` extra adds TabM, Torch, numerical embeddings and ChimeraBoost;
`weather` adds acquisition/array tools; `app` adds Streamlit.
Install only the extras needed for the intended workflow. Neural-model dependencies
are substantial and are not needed to read results.

## Non-destructive source checks

```bash
python -m compileall -q app.py src tools tests
python -m ruff check src/flightdelaybench tests tools
python -m mypy src/flightdelaybench
python -m pytest -q -m "not integration and not slow and not confirmation"
python tools/validate_repository.py
python tools/validate_documentation.py
python tools/build_experiment_index.py --check
```

These commands use synthetic fixtures and small checked-in evidence. Optional app
or frontier checks may skip when their dependencies are absent; report the actual
test summary. A passing source suite does not establish new model performance.

For a create-only combined record, choose a previously unused output:

```bash
python tools/run_flare24_release_checks.py --output reports/local/source_checks_my_run.json
```

Despite its historical filename, the runner records the withdrawal and does not
promote a release. The two fixed-output legacy evaluation/plot tests are excluded
by the marker expression above because they would overwrite preserved reports.

## Build and inspect a local package

Use a new output directory for every build:

```bash
python -m build --no-isolation --outdir dist/my_validation_run
```

`--no-isolation` requires the declared build backend to be installed in the active
environment. The source distribution includes code, protocols, curated reports,
documentation and the small weather pilot, not the large research tables/models.
The wheel installs the research package, not the legacy application dataset.
The `paper/` claim crosswalk and complete experiment index belong to the source
archive. Archive audits must include their exact bytes along with the current
documentation, not just the executable wheel.

For a full clean-wheel audit on the recorded Windows artifact layout, consult
`python tools/validate_cutoff_clean_wheel.py --help`. It requires new environment
and output paths plus the external research data/model paths. It checks legacy-model
inference only, not corrected predictive performance. Old manifest machine paths may
need explicit operator mapping; a different location is acceptable only when the
recorded bytes and hashes match.

## Historical reconstruction

Use the generation-specific guides for
[FLARE-24](FLARE24_REPRODUCIBILITY.md),
[CC-RTH](CCRTH_REPRODUCIBILITY.md) and
[BC-POT-R](BCPOTR_REPRODUCIBILITY.md). They document the old proxy contracts.
Keep those outputs separate from any cutoff-corrected experiment; rerunning the old
builder does not fix the timing limitation.

Download and derive into a dedicated data workspace. Verify manifest hashes before
inference or analysis. Do not run commands against completed output directories.
Retain failed attempts, provider responses and recovered runs.

## Corrected experiments

The [data contract](CUTOFF_DATA_CONTRACT.md) specifies the input schemas and
create-only commands for `flightdelaybench.cutoff_dataset` and
`flightdelaybench.cutoff_experiments`. The latter can record preflight without
training; supplying a new run directory enables the initial grid only after
validated inputs exist.

There is no supplied real-data manifest satisfying the new availability contract.
Do not fabricate timestamps or relabel legacy caches to bypass that gate.
2026 outcomes are not part of these source checks or development commands.
The later [source investigation stopped](RESEARCH_CONTINUATION_20260905.md) without
authorizing an assumption-based alternative. The [roadmap](RESEARCH_ROADMAP.md)
describes conditional options, not permission to resume that campaign.
