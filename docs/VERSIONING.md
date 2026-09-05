# Versioning and branches

[Project status](PROJECT_STATUS.md) · [Legacy application](LEGACY_APPLICATION.md)

## Repository branches

| Branch | Role |
|---|---|
| `main` | Current maintained research code, evidence, validation and documentation |
| `legacy/streamlit-baseline-20260905` | Preserved previous main at `34ac793a6e681a95fdd0ec2044918cb615d146aa`; original application-focused snapshot |
| `research/point-in-time-flightdelaybench` | Retained research-development reference, synchronized with main at the 2026-09-05 promotion |

The promotion is a fast-forward from the old main's history. No old commits,
models, raw evidence, reports, failed runs or tags are removed. The legacy branch
is an archival reference, not the starting point for new research. Existing legacy
application paths remain available in main to avoid breaking reproduction records.

Use main for a new checkout. A previously cloned research branch remains available;
fetch and review local changes before switching. Do not reset or overwrite an
existing checkout to follow the new branch convention.

## Software, evidence and release status

The package version remains `0.1.0rc1`, and citation metadata retain the corresponding
`0.1.0-rc.1` identifier. These identify the current development line and the withdrawn
candidate's history; they do not constitute a new scientific release.

The previous release candidate remains withdrawn pending cutoff correction and
new release validation. No corrected real-data forecast result, release tag,
GitHub release, Zenodo deposit or DOI is created merely by updating main. A
separately scoped software/audit archive still requires the [release checklist](RELEASE_CHECKLIST.md).

Immutable evidence keeps its original version, source commit, path and checksum.
The private transfer refers to commit `972d0554d899317176c53f29fdf2b7db25a1e281`,
before the later source investigation and documentation changes. Restore its data
alongside current source rather than treating that snapshot as the newest checkout.

New experiments, inventories and validation outputs use new identifiers and output
paths. Preserve the old record when an interpretation or validation changes, and
document the reason in the current research account.
