# Research standards

These rules govern contributions, experiments and release review. The
[cutoff data contract](CUTOFF_DATA_CONTRACT.md) additionally requires explicit
event, publication, consumer and label availability; earlier operating dates
alone are insufficient to establish T−24 validity.

- Preserve every tracked historical result, raw archive, model, and failure record.
- Never overwrite a completed run. New runs use a new run identifier and output directory.
- Do not use 2026 outcomes until the confirmation gate is locked and recorded.
- Every predictive feature must be present in the feature-availability registry and must
  be available at the declared prediction horizon.
- Realised weather is an oracle diagnostic, never a deployable input.
- Route, carrier, airport, and congestion statistics must be computed from strictly
  earlier dates. Current-row and future labels are forbidden.
- Fit preprocessing, model selection, calibration, thresholds, and ensembles without
  confirmation outcomes.
- Primary comparisons use proper scoring rules and paired date-cluster uncertainty.
  AUROC, AUPRC, threshold metrics, and decision replays are secondary.
- Retain negative results and distinguish development, retrospective, and confirmatory
  evidence in every report.
- Do not claim causal passenger benefit, operational safety, demographic fairness, or
  real-time forecast validity from retrospective BTS data.
- Run tests, lint, manifest validation, archive inspection, and a clean-environment
  smoke test before a release or tag.
