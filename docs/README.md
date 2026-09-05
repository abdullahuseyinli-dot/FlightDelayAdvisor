# Documentation

Start with the [repository overview](../README.md), then choose a reading path.
Current status pages govern interpretation of historical protocols and reports.

## Current research account

Moving to another machine? Start with the [research handoff](RESEARCH_HANDOFF.md).

| Document | Purpose |
|---|---|
| [Benchmark card](BENCHMARK_CARD.md) | Task, populations, endpoints, splits and metrics |
| [Current results](CURRENT_RESULTS.md) | Same-cohort comparisons, uncertainty and negative evidence |
| [Project status](PROJECT_STATUS.md) | Completed capabilities and unresolved research gates |
| [Data card](DATA_CARD.md) | Source coverage, missingness, timing and network boundaries |
| [Model card](MODEL_CARD.md) | Model families, selection history and intended use |
| [Limitations](LIMITATIONS.md) | What the available evidence cannot establish |
| [Usage](USAGE.md) | Installation, verification and safe execution scopes |
| [Artifacts](ARTIFACTS.md) | Evidence navigation, portability and external data |
| [Architecture](ARCHITECTURE.md) | Source boundaries, timing trust boundaries and extension requirements |
| [Data acquisition and restoration](DATA_ACQUISITION_RUNBOOK.md) | Source/LFS/external tiers and verified transfer scope |
| [Result lineage](RESULT_LINEAGE.md) | Complete sequence of studies, recoveries and stopped work |
| [Experiment index](EXPERIMENT_INDEX.md) | Every retained experiment report with a byte-bound inventory |
| [Research roadmap](RESEARCH_ROADMAP.md) | Conditional improvements, prerequisites and stop criteria |
| [Versioning and branches](VERSIONING.md) | Current main, legacy snapshot and publication boundaries |
| [Manuscript evidence](../paper/README.md) | Claim-to-evidence mapping and publication scope |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Attribution and redistribution boundaries |
| [Research standards](RESEARCH_STANDARDS.md) | Evidence preservation and contribution rules |
| [Release checklist](RELEASE_CHECKLIST.md) | Conditions for a new, accurately scoped release |

## Cutoff-correction work

- [Research plan](CUTOFF_RESEARCH_PLAN.md): sequential correction and experiment gates.
- [Data contract](CUTOFF_DATA_CONTRACT.md): explicit timestamps, lineage and label availability.
- [Implementation state](CUTOFF_IMPLEMENTATION_STATE.md): correction-generation checks.
- [Issued-weather pilot](CUTOFF_WEATHER_PILOT.md): small archive study and its limits.
- [Stopped continuation](RESEARCH_CONTINUATION_20260905.md): later source investigation,
  validation amendments, unrun trial register and aggregate search-exposure disclosure.

## Historical methods and experiments

These documents describe earlier generations. Their original validations checked
the contracts implemented at the time; they do not supersede the later cutoff
withdrawal or establish corrected T−24 validity.

| Track | Method / protocol | Results | Reproduction |
|---|---|---|---|
| FLARE-24 | [Method](FLARE24_METHOD.md) | [Technical report](FLARE24_TECHNICAL_REPORT.md) | [Guide](FLARE24_REPRODUCIBILITY.md) |
| CC-RTH | [Method](CCRTH_METHOD.md), [gates](CCRTH_ACCEPTANCE_GATES.md) | [Technical report](CCRTH_TECHNICAL_REPORT.md) | [Guide](CCRTH_REPRODUCIBILITY.md) |
| BC-POT-R | [Method](BCPOTR_METHOD.md), [gates](BCPOTR_ACCEPTANCE_GATES.md) | [Results](BCPOTR_RESULTS.md) | [Guide](BCPOTR_REPRODUCIBILITY.md) |

Supporting records: [census/context audit](CENSUS_CONTEXT_AUDIT.md),
[historical results ledger](RESULTS.md), [literature map](LITERATURE_AND_NOVELTY.md),
[original research protocol](RESEARCH_PROTOCOL.md),
[original feature contract](POINT_IN_TIME_FEATURES.md), and
[withdrawn candidate release notes](RELEASE_NOTES_FLARE24_RC1.md).

The [legacy application guide](LEGACY_APPLICATION.md) retains installation, historical
sampled scores and script locations. Its [original design document](FlightDelayAdvisor_Documentation.md)
is background, not documentation of the current research package.
