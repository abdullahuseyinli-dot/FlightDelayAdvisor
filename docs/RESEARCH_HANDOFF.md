# Research handoff: continuing on a new computer

Originally prepared 2026-09-05 for `research/point-in-time-flightdelaybench`;
current development now uses `main`. See [versioning](VERSIONING.md) for the
preserved legacy baseline and the unchanged historical transfer snapshot.
Later status: the [source investigation stopped](RESEARCH_CONTINUATION_20260905.md)
without corrected training or an authorized assumption study. Read the updated
[project status](PROJECT_STATUS.md), including its aggregate exposure disclosure,
before following the conditional plan below. The [restoration guide](DATA_ACQUISITION_RUNBOOK.md)
describes the verified private transfer and its earlier source-snapshot boundary.

This is a source-and-evidence handoff, not a scientific release or a restored
T−24 validity claim. Read [project status](PROJECT_STATUS.md),
[research standards](RESEARCH_STANDARDS.md), [current results](CURRENT_RESULTS.md),
[the correction plan](CUTOFF_RESEARCH_PLAN.md) and
[the corrected data contract](CUTOFF_DATA_CONTRACT.md) before training.

## 1. Check out current main

```bash
git clone --branch main --single-branch https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor.git
cd FlightDelayAdvisor
git status --short
git log -1 --oneline
```

Match the commit to the accompanying handoff message. Older transfer manifests
correctly identify their earlier research-branch snapshot; they are not instructions
to replace newer main files. Preserve existing local changes when using an existing
checkout. Main promotion is not a release tag or Zenodo deposit.

Git LFS can fetch substantial legacy data during checkout. To inspect source first,
set `GIT_LFS_SKIP_SMUDGE=1` only for the clone/checkout process, then remove that
temporary setting. This leaves LFS pointers, not materialized data. Avoid an
unqualified `git lfs pull` until the intended artifact scope is understood.

## 2. What transfers through Git, and what does not

| Material | Handoff status |
|---|---|
| Research package, tests, protocols and documentation | Versioned source |
| Historical scores, figures, manifests, method locks and failure records | Small versioned evidence; old claims remain qualified |
| Issued-TAF pilot | 66 small files with source text, responses and hashes |
| Legacy sampled Parquet, two deployed app models, twelve raw 2025 BTS ZIPs | Git LFS; pointer files alone are not their contents |
| Raw BTS 2018–2024, full census/derived tables, forecast archives, research models and monthly prediction partitions | External; reacquire, rebuild or transfer separately |
| Corrected dataset with reviewed availability/label timestamps | Not yet available; additional storage does not create this evidence |
| Virtual environments, caches and local distribution archives | Not transferred through Git; rebuild locally |

The old external workspace was `D:/FlightDelayAdvisorResearchData`. Configure
a suitable root on the new computer instead of assuming that path exists.
Inspect [artifacts](ARTIFACTS.md), source manifests and each CLI's `--help`.
Historical guides are recipes for named old generations, not a complete automated
zero-artifact reconstruction of every run.

Preserve immutable manifest paths and hashes. Record relocation in a separate map
or explicit command arguments. A provider file with different bytes is a new source
version, not the old artifact with an inconvenient hash.

## 3. Establish the environment and verify source

Inspect CPU, GPU, RAM, available storage and operating system. Approximately 1.5 TB
of disk space is an available resource, not evidence of sufficient memory for
parallel full-census fits.

Use Python 3.11 or 3.12 in a dedicated environment and follow [usage](USAGE.md).
The `dev,models,plots` extras support the main source checks; `frontier` adds
the neural/alternative estimators. Report skipped optional tests accurately.
The original correction environment used CPU Torch and TabM; GPU use is not a
prerequisite and must not be silently claimed.

```bash
python -m compileall -q app.py src tools tests
python -m ruff check src/flightdelaybench tests
python -m mypy src/flightdelaybench
python -m pytest -q -m "not integration and not slow and not confirmation"
python tools/validate_repository.py
python tools/validate_documentation.py
```

Run these after installation. They do not require the large research artifacts.
Legacy fixed-output evaluation/plot scripts are excluded because they can overwrite
preserved reports. Artifact-backed inference requires the exact models and input
tables or an explicitly separate rebuild. Validation tools' `--help` documents
path overrides; do not edit immutable reports to point to a different drive.

The `.gitattributes` rules preserve checksum-bound research evidence and module
bytes across newline conventions. Do not apply bulk line-ending normalization to
those paths.

## 4. Known findings and unresolved questions

The latest historical matched cohort has 5,754,266 observed joint outcomes
(January 3–December 29, 2025). The cancellation meta-stack had accuracy 77.3308%,
joint log loss 0.537377 and multiclass Brier 0.326911. The selected boundary
ensemble had 77.3512%, 0.538449 and 0.326679: only +0.0203 percentage points of
accuracy and worse log loss. Neither +5 nor +10 percentage-point targets were met.

These are retrospective proxy results preceding the cutoff correction. Historical
calendar-day windows can include outcomes after a flight-specific T−24 cutoff.
The corrected software has been fixture-tested, but no corrected real-data
performance has been established.

Weather contributed most of the historical gain. Task factorization and stacking
also mattered; architecture is not proven exhausted. Complete context was built
before estimator sampling, but the controlled 125k/250k/full learning curves remain
unrun. The small-airport study changed CatBoost `max_ctr_complexity` from 4 to 1
and omitted the strongest incumbent from its ensemble; it is not an isolated
estimate of the value of boundary data.

2025 is development-informed and cannot become untouched again. Do not acquire
or inspect 2026 or later flight outcomes during development. Old confirmation
locks cannot validate a changed model or input contract.

## 5. Sequential implementation and evaluation

### A. Source and timing gate

Inventory what exists, can be reacquired, must be rebuilt or cannot be obtained.
Raw actual event times can support reconstruction, but are not source-publication
or consumer-receipt timestamps. Never invent availability or relabel old history,
graph or rotation caches as corrected.

Follow the timestamp/label contract in `CUTOFF_DATA_CONTRACT.md`. If authentic
historical availability cannot be established, request a decision between another
source and a separately labelled event-time/latency-assumption study. Keep the
strict track blocked; continue useful read-only feasibility work.

Source checks, self-authored PASS fields and ample disk space cannot satisfy this
scientific gate.

### B. Matched baseline and learning curves

When valid inputs exist, use the initial 42-trial development grid: 14 trials on
each of three seasonal 2024 forward folds. Compare nested 125k/250k/full monthly
estimator samples while retaining full context. Compare baseline, induced and
boundary features with identical target IDs, endpoint masks, settings and budgets.

Evaluate current-feature CatBoost, LightGBM and TabM with direct and hurdle
formulations. Fit preprocessing on training only; censor training and stopping
labels using their availability times. Inspect rather than assume implementation
of the later retuning, finalist repeats and incumbent stacking. Complete those
steps, repeat finalists across at least three seeds, and generate honest forward
incumbent predictions before fitting a stack. Keep an exact no-change option.

### C. Additional information, first in small matched pilots

Candidate sources and primary references:

- [FAA advisory archive](https://www.fly.faa.gov/adv/advAdvisoryForm):
  restrictions and planned traffic-management initiatives.
- [FAA ASPM](https://www.aspm.faa.gov/): airport rates, configurations and traffic.
  Access/revision constraints matter; historical measurements are not automatically
  prediction-time inputs.
- [NWS terminal forecasts](https://aviationweather.gov/help/data/) and
  [IEM API](https://mesonet.agron.iastate.edu/api/1/docs): issued airport weather.
- [NOAA GEFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast):
  longer-range forecast scenarios and uncertainty.
- [OAG snapshot documentation](https://knowledge.oag.com/docs/what-is-a-schedule-snapshot-file-date):
  the schedule actually known at an earlier date, subject to access and licensing.

Verify issue/receipt semantics, historical coverage, amendments, licensing and
flight-weighted coverage before full acquisition. No purchase or access request
on the owner's behalf is implied. Keep hindsight values isolated. The existing
16-case TAF pilot used no flight labels; it is not a population-coverage or
accuracy result, and its latency values are assumptions.

Test simple additions and small residual heads on unchanged scored populations
before designing a large architecture. Do not drop uncovered or difficult flights
silently. Another BTS-derived dataset is not automatically new independent evidence.

### D. Information-value diagnostic

Optionally compare the corrected baseline, genuinely pre-cutoff extra inputs, and
a separately labelled hindsight diagnostic using independently measured weather
or operational conditions. Do not use direct target-label shortcuts.

A large hindsight gain suggests possible information headroom, not T−24 access.
A small gain is not a mathematical ceiling or proof that the task is irreducible.

### E. Conditional airport capacity/recovery model

Proceed when preceding evidence supports it. Predict a distribution of airport
capacity and recovery conditions from eligible forecasts, demand, advance
restrictions and history. Combine it with intra-airport queues and uncertain
interairport predecessor links; evaluate its incremental value for flight outcomes.

Retain uncertainty and the incumbent fallback. Ablate each component against
simple tabular controls. This is a research hypothesis, not an established
invention or promised improvement. A T−6/T−3 extension is a separate task and
must not be presented as a gain on T−24.

## 6. Evaluation, stopping and deliverables

Use chronological splits and paired same-flight comparisons. Report accuracy in
percentage points, joint log loss, Brier, balanced accuracy, class recall,
calibration/ranking, date-cluster intervals and multi-day-block sensitivity.
Include airport/season/severity/coverage diagnostics, seeds, runtime, memory and
storage. Retain all failed and negative runs with unique output directories.

Use parallel agents for independent reviews where helpful; schedule memory-heavy
training according to measured resources. Keep long jobs resumable and report
their actual completion state.

Continue through feasible gates, but do not replace unavailable source evidence
with invented values. Do not promise a five- or ten-point gain or choose a winner
on 2025 and call it prespecified.

Deliver reproducible code/configuration, an artifact/acquisition inventory, an
experiment register, matched results with uncertainty, updated cards and an
evidence-backed recommendation to continue, change scope or stop.

Run source tests, lint, typing, documentation checks, relevant manifest/run
validation and clean-package checks before commits. External publication,
purchase, confirmation access and release tags require separate owner approval.
State exactly what ran, improved, failed and remains blocked.
