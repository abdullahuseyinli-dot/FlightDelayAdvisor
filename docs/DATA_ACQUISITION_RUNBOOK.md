# Data acquisition and restoration

[Artifact guide](ARTIFACTS.md) · [Usage](USAGE.md) · [Data contract](CUTOFF_DATA_CONTRACT.md)

Acquisition, restoration and scientific eligibility are separate operations.
Do not download data or start training merely to read the repository or run the
source-only checks. No command in this guide authorizes 2026+ outcome acquisition.

## Choose the artifact tier

| Tier | Contents | Retrieval or restoration |
|---|---|---|
| Git source and small evidence | Code, configs, tests, reports, source metadata and pilots | Check out `main`; the earlier application baseline is preserved on the documented legacy branch |
| Current Git LFS payloads | Twelve 2025 BTS ZIPs, legacy sampled Parquet and two application models | Retrieve only the intended paths or restore their exact bytes from the private transfer |
| External historical workspace | Raw 2018–2024 BTS, normalized census, feature generations, model/prediction artifacts and FAA archive | Restore the checksummed transfer or follow a named historical generation's reconstruction guide |
| Corrected dataset | Reviewed availability and label timestamps plus new partitions | Not supplied; requires authentic evidence and independent validation |

For a source-first clone, set `GIT_LFS_SKIP_SMUDGE=1` only for the clone process,
then remove the temporary setting. Pointer files are not data/model payloads.
See [usage](USAGE.md) for the exact branch and installation commands.

## Restore the private transfer

The verified 2026-09-05 transfer contains 2,471 payload files, 26,223,570,444 payload
bytes, and 13 independent ZIP volumes. It includes ignored weather responses,
older derived generations, research models, predictions, failure logs and source
archives, as well as the Git-visible files. All 15 LFS payloads and the 84 external
BTS ZIPs were checked against their recorded hashes.

Copy the entire transfer folder and follow its `START_HERE.md` and `transfer.py`
instructions. The restore target must not already exist. Verify every volume and
member, preserve `RESTORE_REPORT.json`, and use `RELOCATION_MAP.json` instead of
editing immutable manifest paths. Model files are trusted inputs only when their
origin and exact hashes have been verified.

That transfer snapshots commit `972d0554d899317176c53f29fdf2b7db25a1e281`; it predates
the later stopped source investigation and this documentation revision. Keep a
current source checkout alongside the restored data. Do not overwrite newer source
with the archived snapshot. The public [transfer record](../manifests/transfer/evidence_transfer_20260905_v1.json)
records the scope and checksum bindings without bundling the large payloads.

## Reacquire or rebuild a missing historical file

1. Identify its exact original record in `manifests/` and the corresponding method
   guide. Preserve URLs, source versions, byte sizes and hashes.
2. Check the source terms and access requirements in [third-party notices](../THIRD_PARTY_NOTICES.md).
   Do not bypass access controls or assume a dated resource snapshot is contemporaneous
   operating configuration.
3. Download to a new path; verify archive integrity and its recorded SHA-256.
   Changed provider bytes form a new source version, not a silently repaired old file.
4. Rebuild only the named generation, with new output and failure records. Inspect
   row counts, sample IDs, endpoints, date coverage and joins in its validator.
5. Keep old histories/graph/rotation features labelled historical proxies. Rebuilding
   them does not repair the later-discovered availability limitation.

The acquisition entry points are in [pyproject.toml](../pyproject.toml); inspect
their `--help` and the generation-specific guides before execution. No universal
one-command reconstruction of every historical run is claimed.

## Source tools and the stop boundary

`tools/probe_cutoff_2024_sources.py` preserves a bounded set of 2024 source responses;
its existing four-request record establishes archive access, not broad receipt or
coverage validity. `tools/inventory_research_handoff.py` is a preserved draft,
not a validated replacement for the completed transfer verifier. Do not assume its
default hash budget verifies every large artifact.

The raw monthly BTS files retain actual departure/arrival fields and source HTTP
metadata, but no reviewed historical per-flight receipt/label-availability ledger
has been found. The 16-case TAF pilot contains producer issue times and assumed
latencies. These distinctions remain hard limits on corrected training.
Read the [stopped continuation](RESEARCH_CONTINUATION_20260905.md) before resuming.
