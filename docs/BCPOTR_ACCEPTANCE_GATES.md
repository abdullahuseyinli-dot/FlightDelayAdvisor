# BC-POT-R acceptance gates

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

No gate may be silently weakened after results are observed.

| Gate | Requirement | Evidence | Final state |
|---|---|---|---|
| G0 Evidence preservation | Create-only outputs; raw archives, failures, old graphs, models, and reports remain untouched | Immutable run IDs, manifests, checksums | Enforced |
| G1 Frozen target | Every feature partition has exactly the frozen top-100 target ID set | Boundary feature validator | Passed: 24/24 partitions, 11,759,279 rows, exact target-ID set; validation hash `9a055d9d8d2f4ffeef3e22218b04598c639e9a5cbb9ff49f8f37f175f4e45249` |
| G2 Information boundary | No outcomes, actual operations, target tails, realized weather, or 2026 outcomes enter features | Column projection, manifests, registry, tests | Enforced |
| G3 Temporal frontier | Target year Y uses schedule frontier Y-1; selection uses 2024 only | Frontier records and method lock | Enforced |
| G4 Boundary mechanism | Boundary flights change dynamic state and contribute context-only predecessor messages; static fields do not drift | Diagnostics and paired-view tests | Passed in January smoke |
| G4b Boundary-trained rotation | Y-1 boundary-tail supervision passes validation; January target IDs and non-rotation features remain identical; at least 0.1% of rows have a changed rotation message before a full rerun | Rotation-model validator and outcome-blind smoke comparator | Passed: 2023/2024 fits validated; 458,266/459,526 January rows (99.726%) changed at least one rotation message, with exact IDs and unchanged non-rotation features |
| G5 Pre-outcome lock | Protocol, candidates, model settings, inputs, +0.05/+0.10 accuracy gates, and source hashes are written before the new study loads outcomes | `method_lock.json` | Passed: v6 lock self-hash `a6338186474dcaa70b92797cf165d7ba6e7addbfc3dfa5bb8490d7147489dff7` |
| G6 Scientific comparison | Same rows; proper scores primary; paired whole-date uncertainty; negative results retained | Study report and validator | Passed: 5,754,266 primary rows; 2,000 whole-date resamples; validation self-hash `c8a9f885305018db3767a1c56800e75db62faf1c3bb6335dfd5c962baf551ce8` |
| G7 Requested breakthrough | Standard argmax accuracy improves by at least +0.05 absolute over the earlier meta-stack | Recomputed study-validation gate | **Failed honestly:** 0.773512 versus 0.773308, absolute gain +0.000203; neither +0.05 nor +0.10 passed |
| G8 Confirmation boundary | 2026 outcomes remain unopened | Lock, run report, validators | Passed: report and validation both record 2026 unopened |

The January 2024 smoke result contains 459,526 exactly matched target IDs, 38,781
context-only predecessor states, 310,087 context-only rotation edges, and 99.757% mean
rotation-message coverage. These numbers validate mechanism execution only; they are not
predictive-performance evidence.

The embargo-safe boundary-rotation smoke comparison changed at least one rotation
message for 458,266 of the same 459,526 January rows (99.726%). Target IDs, static
features, and every non-rotation feature were bitwise or tolerance-equivalent. This is
also mechanism evidence only.

The v6 predictive run completed all four models, four calibrators, three raw Q4
selection partitions, the joint Q4 selection artifact, and all twelve 2025 prediction
partitions. Final report assembly then encountered a saturated descriptive contrast:
only four primary-period rows had no nonzero boundary residual. The failure remains at
`manifests/failures/flare24_boundary_pot_study_v6_saturated_contrast.json`. The recovered
report re-derived calibration, selection, scores, and uncertainty from the immutable v6
artifacts without refitting or regenerating predictions. The independent validator
confirmed exact selection IDs and maximum reproduction errors below `3.5e-8`.
