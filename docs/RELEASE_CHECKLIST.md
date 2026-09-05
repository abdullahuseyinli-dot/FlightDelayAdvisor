# Release checklist

The historical `0.1.0rc1` candidate is withdrawn. This checklist describes the
current gates for a new, accurately scoped source/evidence release; old checklist
passes and package archives remain historical records, not current approval.

## Evidence and documentation

- [x] Preserve historical predictions, scores, source responses, failures and archives.
- [x] Record the v3 withdrawal with a hash-bound reference to the old ledger.
- [x] Separate historical proxy results, software checks and uncompleted experiments.
- [x] Disclose the timing defect, boundary-study confounding and failed +5/+10-point gates.
- [x] Keep 2025 development-informed; distinguish the unopened 2026 confirmation
  dataset from the later recorded aggregate search exposure.
- [x] Include benchmark/data/model cards, limitations, usage, artifact inventory,
  current results, source notices and citation metadata.
- [x] Index the complete experiment history, stopped continuation and private
  transfer; provide architecture, acquisition and conditional improvement guides.
- [ ] Review a new final source/evidence inventory after all changes are complete.
- [ ] Review the exact redistribution rights and contributor attribution for the
  proposed archive payload.

## Scientific gates for corrected forecast claims

- [ ] Establish authentic feature and label availability at the declared horizon.
- [ ] Materialize and independently validate a new corrected dataset.
- [ ] Measure feature-build runtime, memory and partition-boundary equivalence.
- [ ] Run matched baseline/context comparisons with equal model settings and budgets.
- [ ] Complete model-family, sample-size, retuning and finalist-repeat comparisons.
- [ ] Generate honest forward incumbent predictions before stacking.
- [ ] Report all trials, proper scores, class metrics and date/week uncertainty.
- [ ] Freeze corrected inputs, models and analysis before independent confirmation.
- [ ] Complete independent confirmation under the reviewed lock.

A clearly labelled software/negative-results archive can have a narrower scope
than corrected forecast validation. It must not inherit the withdrawn release's
claims. No documentation cleanup alone satisfies the scientific gates above.

## Source and distribution gates

Run these against the exact final checkout and record new outputs. The presentation
review has passing [source/documentation checks](../reports/validation/repository_presentation_source_checks_v1.json)
and [fresh-wheel inference](../reports/validation/repository_presentation_clean_wheel_v1.json);
final release promotion remains a separate decision:

- [x] Compilation, lint, strict typing and source-only tests.
- [x] Dependency-lock consistency.
- [x] Repository evidence, local documentation links, metric tables and metadata.
- [ ] Wheel/source build and byte-level archive inventory.
- [x] Fresh installed-wheel imports and inference semantics where artifacts exist.
- [ ] Review skipped tests, warnings, failures and external artifact prerequisites.
- [ ] Verify that the release manifest binds the current files, not withdrawn hashes.

The [usage guide](USAGE.md) supplies non-destructive commands. Dated correction
checks are in [project status](PROJECT_STATUS.md); new checks must not overwrite
them. Legacy fixed-output evaluation/plot tests are not safe to run over preserved
reports and are excluded from the source-only suite.

## External publication

- [ ] Owner review of the final archive scope and scientific claims.
- [ ] Release commit and tag.
- [ ] GitHub release.
- [ ] Zenodo deposit, if selected, with verified payload and metadata.
- [ ] DOI added only after an actual deposit assigns it.

No commit, tag, external upload or DOI follows automatically from a local check.
