# Artifacts and result lineage

[Documentation index](README.md) · [Current status](PROJECT_STATUS.md)

## Curated evidence

| Evidence | Location | Interpretation |
|---|---|---|
| Current withdrawal | [v3 cutoff withdrawal](../manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json) | Governs prior release claims |
| Repository alignment | [Source checks](../reports/validation/repository_alignment_source_checks_v1.json), [clean wheel](../reports/validation/repository_alignment_clean_wheel_v1.json), [archive audit](../reports/validation/repository_alignment_bundle_v1.json) | 320 passed tests; complete experiment index and current public structure; inference remains historical |
| Earlier presentation review | [Source and documentation checks](../reports/validation/repository_presentation_source_checks_v1.json), [fresh-wheel validation](../reports/validation/repository_presentation_clean_wheel_v1.json) | 269 passed tests at that revision; checked links/metrics and historical inference |
| Parent selection | [2024 selection](../reports/experiments/flare24_2024_selection_v2.json), [method lock](../manifests/flare24_method_lock_v1.json) | Historical choices |
| Parent audit | [2025 report](../reports/experiments/flare24_2025_audit_v1_recovered.json), [nested ablations](../reports/experiments/flare24_2025_nested_ablation_v1.json) | Full-year proxy results |
| Airport resources | [CC-RTH](../reports/experiments/flare24_ccrth_2025_retrospective_v5.json), [factorized model](../reports/experiments/flare24_ccrth_task_factorized_v1.json), [meta-stack](../reports/experiments/flare24_ccrth_metastack_v1.json) | Historical model evolution |
| Boundary experiment | [Recovered study](../reports/experiments/flare24_boundary_pot_v6_recovered.json), [independent validation](../reports/validation/flare24_boundary_pot_study_v6_recovered_v2.json) | Source of latest same-cohort table |
| Census coverage | [Context audit](../reports/validation/census_top100_context_completeness_v1.json) | Continuity and network-boundary counts |
| Source schema | [2024 audit](../reports/validation/cutoff_source_schema_2024_v1.json) | Headers/metadata, not outcome-value completeness |
| Issued-weather pilot | [Report](../data/external/cutoff_taf_pilot_v1/report.json) | 16 cases, raw source text and retrieval records alongside |
| Corrected source checks | [v14 source suite](../reports/validation/flare24_source_checks_v14_cutoff_correction.json) | Dated software checks |
| Corrected package checks | [Fresh-wheel audit](../reports/validation/cutoff_clean_wheel_v1.json), [archive audit](../reports/validation/cutoff_evidence_bundle_v5.json) | Inference semantics and distribution contents |
| Research preflight | [Prerequisite report](../reports/validation/cutoff_research_preflight_v1.json) | Blocked, zero real-data trials |
| Failures and supersessions | [Failure directory](../manifests/failures) | Preserved, not cleanup targets |
| Complete experiment register | [Human-readable index](EXPERIMENT_INDEX.md), [byte inventory](../manifests/research_experiment_index_v1.json) | All retained experiment JSON/Markdown; blocked registers are not completed fits |
| Later continuation | [Stop/source investigation record](RESEARCH_CONTINUATION_20260905.md) | Source-access findings, validation amendments and exposure disclosure |
| Private evidence transfer | [Portable summary](../manifests/transfer/evidence_transfer_20260905_v1.json), [restoration guide](DATA_ACQUISITION_RUNBOOK.md) | 2,471 payload files verified; historical timing evidence still absent |

Historical publication-asset manifests describe validated tables/figures, not a
completed external publication. Historical `PASS` statuses establish only their
stated contracts; the later cutoff withdrawal is not erased by them.

## Storage and portability

Small protocols, reports, source metadata, figures and manifests belong in the
repository. Large research data, predictions, models and forecast archives remain
in an external workspace with a logical layout such as:

```text
research-data/
  raw_bts/<year>/
  normalized_top100_census_v2/year=<year>/month=<month>.parquet
  derived/<feature-generation>/
  runs/<unique-run-id>/
```

Use the exact manifest record, not this illustrative tree, to identify an artifact.
Verify SHA-256, byte size, schema, row count, target IDs and time coverage after
relocating it. Some historical commands contain absolute machine paths and require
explicit remapping; automatic cross-machine reconstruction is not claimed.

The legacy demo's curated Parquet file and two deployed model files use Git LFS.
Their presence as pointer files does not mean the model/data bytes are materialized.
The twelve raw 2025 BTS ZIPs also use LFS. The verified private transfer contains
all 15 current LFS payloads plus external research inputs and outputs. Its source
snapshot predates the stopped continuation; preserve it and use a newer checkout
for subsequent code and documentation rather than overwriting either generation.

The public repository retains compact evidence and navigable provenance; it does
not upload the 24.4-GiB private transfer as ordinary Git files. Environment caches,
installed dependencies and duplicate checkout exports were excluded from transfer,
not deleted from the source computer. Original machine paths inside immutable
evidence are provenance, not instructions to reuse those paths on a new machine.

## Integrity and recovery

Reports/manifests may have both a file SHA-256 and a canonical-JSON self-hash;
these are different quantities. Do not reformat an immutable JSON record or replace
its recorded paths in place. Portable annotations should be separate, with a link
back to the original bytes.

The parent float32 probability recovery and the boundary saturated-contrast
recovery reassembled reports from preserved predictions. Failed finalizers,
recovery rules and recovered reports remain separate. Their detailed records are
linked from the historical technical guides.

The source archive excludes large raw/model payloads. It is a code-and-small-evidence
bundle, not a full-data reproduction deposit. Future archives need an explicit
redistribution review and a fresh archive inventory.
