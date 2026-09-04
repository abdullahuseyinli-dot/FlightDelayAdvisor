# CC-RTH acceptance gates

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

These gates separate artifact integrity, scientific validity, predictive evidence, and release readiness. A negative model result does not erase a valid benchmark; it changes the claim that may pass.

## Gate 1 — Information contract

Pass only when:

- every CC-RTH predictor is registered at the 24-hour horizon;
- graph construction projects BTS inputs to schedule-only columns;
- target outcomes, actual movement times, realised weather, and target-year tail identifiers are absent;
- the scheduling frontier uses target year minus one only;
- optional constraints reject records issued after the flight cutoff; and
- no 2026 outcome is accessed.

Evidence: feature registry tests, construction manifest, source provenance, and graph validation report.

## Gate 2 — Graph and numerical integrity

Pass only when:

- all 24 monthly partitions for 2024–2025 exist and match their checksums;
- sample, flight-node, and resource-node identifiers are unique;
- every target flight has origin and destination resource incidence;
- incidence and rotation successors have no dangling target endpoints;
- incoming latent-rotation probability is bounded by one;
- feature values contain no infinity and bounded fields remain in range;
- proxy queues reset at operational-day boundaries; and
- every schedule-time repair is outcome-blind, counted, and reproducible.

Evidence: `flightdelaybench-validate-ccrth --capacity-manifest ...`. Modeling cannot start without a checksum-bound PASS record.

The v1 graph passes this gate for its declared induced top-100 cohort. The separate
census/context audit must also pass and the network boundary must be reported. This
gate does not permit “complete nationwide graph” language: 2,207,091 raw 2024-2025
flights cross between the cohort and smaller airports and are outside v1 context.

## Gate 3 — Temporal protocol

Pass only when:

- iteration selection trains no later than 2024-08-29, embargoes August 30–31, and validates in September;
- fixed-iteration models refit only through 2024-09-30;
- October 1–2 separate model refit from calibration training;
- every calibration fold uses a two-date embargo;
- calibration and blend choices use 2024 only;
- any cross-run model recovery is admitted only from a self-hashed failure record
  proving all eight model checksums/contracts and a pre-lock, 2024-only boundary;
- a self-hashed method lock is written before the run opens a 2025 outcome-bearing artifact; and
- primary 2025 scoring spans January 3–December 29, excluding January 1–2 for the left-boundary embargo and December 30–31 because complete +2-day context would require unopened 2026 schedules, and clusters uncertainty by all 361 retained `FlightDate` values.

Evidence: frozen TOML protocol, run model/calibrator records, method lock, and study validator.

## Gate 4 — Predictive claim

The benchmark/evaluation claim passes when the exact FLARE-24 baseline is reproduced, all nested candidates are scored on identical eligible rows, proper scores and paired date-cluster intervals are reported, and negative increments are retained.

An **improvement** claim is allowed only for a frozen 2024-selected method whose 2025 primary proper-score difference is negative. A stronger “robust improvement” statement additionally requires the paired 95% interval to exclude zero on both joint log loss and multiclass Brier score. No state-of-the-art claim follows without a reproduced comparator using the same cohort, horizon, information set, endpoint, and split.

## Gate 5 — Mechanism interpretation

Pass only when raw demand, normalized capacity, queue/shadow-price, and graph-message increments are shown separately; missingness and stress-regime behavior are reported; and feature importance is labeled descriptive. Queue, overload, recovery, scenario probability, marginal load, and shadow price remain proxies unless independently validated against operational airport data.

## Gate 6 — Release

Pass only when:

- package tests, research-package lint, strict typing, compilation, and builds pass;
- graph and study validators pass from immutable manifests;
- tables and figures are generated from the self-hashed report and themselves checksummed;
- README, method, data card, model card, results, limitations, changelog, citation, and Zenodo metadata agree with measured values;
- large artifacts remain external but checksum-addressable; and
- calendar continuity, induced-cohort reconstruction, and outer-network omissions
  are quantified in the census/context audit;
- failed/partial runs and their remediation records remain visible.

No Git tag or Zenodo deposit should be described as complete before all six gates have evidence. The 2026 confirmation gate remains separately closed.
