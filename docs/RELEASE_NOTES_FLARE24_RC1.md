# FLARE-24 + BC-POT-R 0.1.0-rc.1 release notes

Status: WITHDRAWN following the cutoff audit on 2026-09-04. These notes describe the
historical candidate, not an approved current release. No external commit, tag,
GitHub release, Zenodo deposit or DOI is claimed. See
[project status](PROJECT_STATUS.md) and the [current checklist](RELEASE_CHECKLIST.md).

## Headline result

FLARE-24 was selected using 2024 only, frozen before evaluation, and audited without
refitting on the complete 2025 top-100-airport census. Among 5,814,817 flights with a
resolved three-state outcome, the selected structural-rotation candidate achieved:

| Measure | Rich baseline | FLARE-24 selected | Relative reduction |
|---|---:|---:|---:|
| Joint log loss | 0.555476 | 0.539225 | 2.93% |
| Multiclass Brier | 0.337528 | 0.327076 | 3.10% |

The paired 365-date differences are -0.016251 (95% interval -0.018948 to -0.013951)
and -0.010452 (-0.011909 to -0.009189). Joint log loss improves in all 12 months.
The result is a frozen-method retrospective audit, not blind confirmation.

## Boundary-complete extension result

BC-POT-R adds 3,216,969 schedules connecting the target system with 255 smaller
airports as context while keeping the scored top-100-to-top-100 sample IDs unchanged.
Its Q4-selected residual-gated ensemble reaches 0.773512 standard argmax accuracy on
5,754,266 primary-period flights versus 0.773308 for the prior meta-stack: +0.000203
absolute. This is far below the predeclared +0.05 and +0.10 gates, both of which fail.

The method slightly improves multiclass Brier but significantly worsens joint log loss
against the prior meta-stack. The counterfactual-residual candidate receives zero
ensemble weight. Severe boundary-induced route-shadow pressure nevertheless identifies
a higher-disruption regime. These mixed and negative results are part of the release,
not filtered out.

## What is new

- A flight-specific scheduled-departure-minus-24-hour information boundary.
- Fixed-vintage Open-Meteo GFS weather selected by issue time, with day-2 fallback and
  revision features.
- FAA runway-heading-envelope, bank-load, endpoint, window, and corridor transforms.
- Capacitated schedule-only latent predecessor probabilities that never read
  target-year tail identity.
- Coherent three-state hurdle probabilities for on-time, delayed, and cancelled.
- Forward-only calibration, simplex ensemble, independent aggregate forecasting, and
  uncertainty-weighted marginal-alignment selection.
- Proper-score inference with 2,000 paired bootstrap resamples of flight dates.
- Independent validators, model/method/confirmation locks, publication figures,
  data/model cards, a technical report, and a complete reproduction guide.
- Curated distribution rules and an installed-wheel, frozen-model inference smoke
  command.
- A boundary-complete schedule graph, context-only predecessor states, paired
  induced/complete views, and an exact same-flight small-airport boundary ablation.
- An audited v6 report recovery that refits no model, regenerates no prediction, and
  reproduces persisted ensemble probabilities below `3.5e-8`.

## What actually survived selection

The winning model is the structural-rotation candidate with identity calibration.
The convex ensemble puts weight 1.0 on that candidate. Aggregate alignment is locked
off. The name FLARE-24 describes the full tested framework; it does not imply that
every inventive component improved prediction.

Fixed-vintage weather supplies most of the gain. Structural rotation improves joint
Brier and conditional-delay log loss beyond weather, but its incremental joint-log-
loss interval crosses zero and its cancellation log loss is worse. Propagated
predecessor risk is worse than structural rotation on both joint proper scores.

## Numerical-method result

Diagonal preconditioning of the reconciliation dual is an exact reparameterization.
On the recorded selection problem it converged in 30 iterations and 0.278 seconds,
where the old unscaled L-BFGS solve failed at 1,000 iterations. This is a substantial
numerical improvement within the codebase. Because all nonzero alignment strengths
lost in 2024, it is not promoted as a predictive breakthrough.

## Failure and recovery disclosure

The original 2025 finalizer completed all 12 monthly inference and aggregate
partitions, then rejected normal float32 simplex round-off under an overly strict
`1e-8` tolerance. Maximum row-sum error was `4.470348e-08`; no row exceeded the
recovery bound of `1e-6`. The recovery verified immutable hashes, normalized each row
in float64, and reran report assembly only. Maximum probability movement was about
`3.33e-08`; no fit, prediction, calibrator, or selected choice changed. Both the
failure and recovery evidence are retained.

## Verification state

- All 11,759,279 weather/aviation and rotation feature rows reopened successfully.
- The 2024 selection, method lock, all eight models/calibrators, 12 audit prediction
  partitions, 12 aggregate partitions, 2025 report, nested ablation, publication
  bundle, and future-confirmation lock passed the combined release validator.
- The source suite, strict typing, lint, lock consistency, curated archives, and
  fresh-wheel inference are recorded in the release-candidate ledger generated after
  packaging.
- The BC-POT-R recovered report, independent validator, publication manifest, and
  publication validator pass; the installed wheel independently reruns both boundary
  validators.
- The two old integration tests invoke legacy scripts that overwrite fixed plot
  outputs. They are not used as FLARE release evidence; manifest-backed validation
  plus the create-only wheel smoke replace that unsafe gate.

## Evidence identities

The frozen scientific evidence hashes are listed in `docs/RESULTS.md`. The combined
release asset validation file has SHA-256
`0035f9dbf814346ad61b649d5df204d0c41995ef98a68ebaf6f23704a11e3517`.
The final release ledger records the exact wheel, source archive, source tree, quality
report, and fresh-smoke hashes.

## Known limits

This release does not establish causal benefit, production feed readiness, active
runway state, target-year aircraft identity, operational safety, demographic fairness,
or universal state-of-the-art performance. It covers a retrospective US BTS
top-100-airport cohort. January-June 2026 outcomes remain unopened behind a frozen
confirmation analysis.

## External publication handoff

The local candidate is ready to be reviewed for a release commit. Publishing still
requires an authorized commit, annotated tag, GitHub release, and/or Zenodo upload.
Only Zenodo can assign the DOI; no placeholder DOI is present in the metadata.
