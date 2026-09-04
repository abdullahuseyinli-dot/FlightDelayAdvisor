# Data card

Status: census counts and historical artifact checks are preserved. Strict
flight-specific T−24 availability has not been established for the old feature
generation, and its release candidate is withdrawn. This card distinguishes
structural integrity from prediction-time validity. See [project status](PROJECT_STATUS.md).

## Sources and scope

FlightDelayBench uses US DOT Bureau of Transportation Statistics (BTS) Reporting
Carrier On-Time Performance records. FLARE-24 restricts the official census files to
the frozen union of 100 airports declared from the pre-existing 2024 cohort. Its
materialized 2024-2025 census contains 11,759,279 schedule rows in 24 monthly
partitions: 5,929,613 in 2024 and 5,829,666 in 2025.

The older 2010-2024 application track uses an approximately month-balanced row
sample. It is not interchangeable with the FLARE-24 census and must not be used to
construct airport-resource or interairport context. In FLARE-24/CC-RTH, schedule
features are constructed from every retained census row before supervised model-row
sampling; absolute scores across the tracks are not controlled comparisons.

Additional covariates are:

- Open-Meteo GFS Previous Runs at nominal 24- and 48-hour fixed leads for 100
  airports, retaining issue and valid time separately;
- an FAA NASR runway snapshot effective 2024-10-03, used as a runway-heading envelope
  rather than an active-runway record;
- closed-left 7/28/90-day route, carrier, flight-number, and directional-airport
  outcome summaries;
- schedule-density and CL-SGMP graph-pressure features; and
- schedule-only latent predecessor probabilities learned with prior-year tail
  supervision but applied without reading target-year tail numbers.

The weather cube contains 1,756,614 airport-hour rows for 2024 and 1,751,814 for
2025. The joined weather/aviation/corridor and rotation datasets each align exactly
to all 11,759,279 census rows.

## Endpoints and retained rows

- Joint state: on time, arrival delay of at least 15 minutes, or cancellation when a
  unique state is observed. The 2025 audit scores 5,814,817 rows.
- Conditional delay: `ArrDel15` among operated, non-diverted flights with an observed
  delay label. The 2025 audit scores 5,732,449 rows.
- Cancellation: binary `Cancelled` on every scheduled cohort row with that label.
  A later diversion remains a non-cancellation. The 2025 audit scores all 5,829,666
  rows.

Missing delay or joint outcomes are never imputed as on time. Cancellation rows and
rows without a delay outcome remain available to the cancellation task.

## Prediction-time boundary

The intended cutoff is scheduled departure in UTC minus 24 hours. A weather value
must have a defensible issue/availability time no later than that cutoff.
Historical features used `[D-window, D)` and excluded the target day. This does
not establish T−24 validity: an earlier operating day's outcome may still occur
or arrive after the target cutoff. Target outcomes,
realised weather, actual movement times, delay causes, target-year tail identity, and
other post-departure fields are forbidden predictors.

BTS schedule fields are a retrospective proxy for a D-24 schedule snapshot. BTS
publication timing does not establish that the closed-left outcome summaries would
be available from a live production feed. The corrected contract requires explicit
event, source-publication and consumer-availability times, plus label availability
for fitting and stopping. No real-data dataset satisfying that contract is currently
available. See [the data contract](CUTOFF_DATA_CONTRACT.md).

The 12-month 2024 schema audit found actual event-clock fields in raw archives but
not normalized tables. Neither layer supplied historical publication/receipt and
label-availability evidence. Header presence is not a completeness or timing audit
of the underlying outcome values.

## Coverage and missingness

The create-only context audit verifies all 96 monthly partitions and all 2,922 dates
from 2018-01-01 through 2025-12-31. The 43,815,581-row census has 141 of 292,200
airport-days with no retained inbound or outbound service, all at HPN in 2020; zero
service is disclosed rather than automatically imputed as a missing file.

The frozen cohort is an induced network: both endpoints must be in the top 100. An
independent scan of all 14,080,680 raw BTS rows in 2024-2025 exactly reconstructs the
11,759,279 retained rows and identifies 2,207,091 additional top-100-to-outside
flights. Thus 15.803% of flights touching a target airport are absent from the v1
context graph. Temporal completeness and network-boundary completeness are separate
properties; only the former is complete here. See `docs/CENSUS_CONTEXT_AUDIT.md`.

The FLARE feature schema contains 108 registered weather/aviation/corridor columns.
Seventy-six passed training-time usability checks. Twenty-three registered fields
have zero coverage in this fixed-lead archive—principally visibility, ceiling,
reflectivity, and freezing-level derivatives—and were automatically excluded rather
than imputed or represented as measured. Structural and risk rotation candidates use
83 and 84 extra features respectively.

In the 2025 audit, 5,814,365 of 5,814,817 joint-scored rows have both endpoint weather
covariates; 452 do not. Missing-weather results are reported separately and do not
support a benefit claim.

The CC-RTH extension aligns 128 registered resource features to all 11,759,279
2024-2025 flights and exports 5,175,820 resource-time nodes, 32,128,012 incidence
edges, and 113,252,743 uncertain rotation edges. Across 1,505,187,712 numeric feature
cells, 3.2355% are missing. Four registered fields have zero coverage: origin and
destination annual-operations metadata, and origin and destination optional-constraint
overload probabilities. The NASR cycle did not supply a defensible annual-operations
value for this transform, and no authentic issue-time-vintaged operational-constraint
feed was supplied; these fields remain missing and are excluded by the training-only
usability rule. Runway-feasibility and weather-stress coverage is approximately
97.6%, while predecessor-resource message values are present for 99.24% of flights.

The graph audit also records that the origin overload-probability proxy has a median
of one in all 24 partitions. It is therefore not treated as a stand-alone physical
capacity measurement. Continuous utilization, slack, queue, recovery, and shadow-price
features remain available, and the nested normalized-capacity ablation is required to
show whether this proxy construction adds predictive information.

The BC-POT-R boundary-complete extension retains the same 11,759,279 target rows but
builds their resource context from 20,761,297 schedule-only rows: 17,544,328 induced
rows plus 3,216,969 boundary rows across 355 airports. The 255 additional airports are
context-only. Their outcomes are not feature inputs and their flights are not silently
added to the scored cohort. The final boundary-rotation build computes 1,031,784
context-only predecessor states and 8,271,919 context-only incidence edges. Exact
target sample IDs are independently validated in all 24 partitions.

## Integrity and lineage

Each raw or derived asset has a create-only manifest recording source paths and
hashes, projected columns, row counts, output hashes, versions, and explicit outcome
access. Independent validators reopen Parquet partitions and check period alignment,
the cutoffs implemented at the time, schemas, probability bounds, latent inbound
capacity, and self-hashes. A historical PASS does not validate a stronger timing
contract introduced later.

Raw provider responses, failed/superseded builds, and the successful artifacts are
all retained. No 2026 outcome was acquired or read for this release candidate.

## Known limitations

- The scored cohort remains top-100-to-top-100. BC-POT-R now includes the previously
  omitted top-100-touching long-tail schedules as context, but does not establish label
  performance or generalization for those 255 context-only airports.
- A retrospective BTS schedule is not proof of D-24 production schedule availability.
- The FAA runway transform does not know the active runway.
- The single FAA NASR snapshot is effective 2024-10-03 and is used only as static
  reference geometry. It is not a cycle-correct historical metadata panel for flights
  earlier in 2024; any CC-RTH result depending on runway geometry requires a
  cycle-vintage sensitivity analysis before a strict real-time claim.
- Corridor weather uses nearby airport forecast nodes, not a full gridded trajectory.
- Latent rotations are hypotheses, not recovered aircraft assignments.
- Weather archives do not establish live API uptime, latency, or future availability.
- No demographic attributes are present; origin/carrier diagnostics are not
  demographic fairness analyses.

## Redistribution

Code, configuration, small reports, validation records, and manifests are suitable
for GitHub/Zenodo. Large BTS-derived tables, provider responses, and model artifacts
may remain external and are referenced by exact hashes and acquisition instructions.
Redistributors must review provider and archive-host terms; Git LFS is an artifact
transport mechanism, not a data license. The MIT software license does not
relicense upstream data. See [third-party notices](../THIRD_PARTY_NOTICES.md).
