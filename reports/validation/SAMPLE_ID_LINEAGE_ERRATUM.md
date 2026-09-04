# Sample-ID lineage erratum

The inherited chunked BTS normalizer added `source_offset` to a pandas index that was
already global across CSV chunks. Consequently, normalized files created before the
v2 repair have unique and stable IDs, but IDs after the first 200,000 raw rows do not
encode the exact original CSV row number as claimed.

This defect does not change rows, labels, features, order, model fitting, or recorded
metrics. It does weaken raw-row traceability. The new official-census pilot
(`census_top100_2018_v1`) is therefore quarantined in full and will not be used for
modeling. Its raw files, normalized outputs, manifest, logs, and original PASS report
are preserved as failure evidence. The repaired census is written to a new v2 path and
must pass an explicit cross-chunk ID test plus raw-to-normalized reconstruction.

The earlier 2025 normalized cohort and point-in-time derivatives are retained and may
still support the legacy benchmark because their IDs remain unique; they must not be
described as exact raw-row locators. Future release documentation binds this erratum.
