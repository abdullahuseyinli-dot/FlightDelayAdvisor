# Continuation stopped at the source-evidence gate

The owner instructed: "if there is no data, then stop". Raw flight data is present,
but the reviewed historical publication, consumer-receipt, label-availability and
schedule-vintage evidence required for corrected T−24 experiments is not available.
Research stopped. No corrected real-data model was fitted and no improvement was
measured. An assumption study was not authorized or started.

The checkout was clean at the start, on `research/point-in-time-flightdelaybench`,
with both HEAD and fetched branch at handoff commit
`972d0554d899317176c53f29fdf2b7db25a1e281`. No commit, tag or publication was made.

## Preserved evidence

- [Experiment register](../reports/experiments/handoff_continuation_20260905_register_v1.json):
  all 42 initial trial slots are explicitly unrun; corrected metrics remain null.
- [Initial source checks](../reports/validation/handoff_continuation_20260905_source_checks_v1.json):
  280 tests passed, one legacy app integration module skipped, two fixed-output
  tests deselected; compilation, Ruff, mypy, lock, repository and documentation
  checks passed before edits.
- [Fresh 2024 schema audit](../reports/validation/handoff_continuation_20260905_schema_v1.json):
  twelve raw/normalized partitions inspected; 5,929,613 normalized rows by metadata.
  Required availability fields are absent. No outcome values were read by this audit.
- [Resource inventory](../reports/validation/handoff_continuation_20260905_resources_v1.json):
  Windows 11, i9-13900HX, roughly 32 GiB RAM, RTX 4060 with 8 GiB VRAM.
  Attached volumes showed roughly 38 GiB free on C: and 214 MiB on D:;
  the stated 1.5 TB was not visible. Installed Torch is CPU-only.
- [Preflight](../reports/validation/handoff_continuation_20260905_preflight_v1.json):
  optional model packages are present; the availability dataset gate remains blocked.
  `.local/research_data` was created for possible future outputs. Existing external
  data remains at its original root; no relocation or manifest rewriting occurred.
- [Small source-access probe](../reports/validation/handoff_continuation_20260905_source_probe_v1/report.json):
  four fixed 2024 FAA/NWS requests preserved 48,263 response bytes and hashes.
  FAA advisory send time and NWS producer issue time are available in these cases;
  neither establishes historical consumer receipt or flight-weighted coverage.
- [Search exposure incident](../manifests/failures/handoff_continuation_20260905_incidental_search_exposure_v1.json):
  unsolicited search snippets exposed 2026 aggregate statistics. No confirmation
  dataset or outcome page was opened, and the values were not used. Zero exposure
  must not be claimed. Broad outcome-adjacent searches were stopped.

## Code state at stop

Endpoint validation now rejects masks that hide observed arrival labels and
contradictory cancellation/diversion labels while preserving genuinely missing
outcomes. History validation rejects incompatible cancellation/delay observations
and conflicting per-sample identity/group metadata. Focused validation passed
66 fixture tests, Ruff and mypy, with three LightGBM deprecation warnings.
These checks do not establish source authenticity or predictive performance.

The create-only [source probe](../tools/probe_cutoff_2024_sources.py) has three passing
fixture tests and passing Ruff after two initial lint findings were fixed.
The [artifact inventory tool](../tools/inventory_research_handoff.py) is an incomplete,
untested draft stopped before execution. Git LFS marks 15 files as materialized,
and eight raw manifests list 96 archives through 2025, but this continuation did
not verify their payload hashes. Do not treat those observations as an integrity pass.

No full post-edit suite, new package build or scientific acceptance was completed
after the stop instruction. Retuning, repeats, forward stacking, resumable grid
execution, complete diagnostics and later research gates remain unfinished.
Historical results and failures are preserved unchanged.
