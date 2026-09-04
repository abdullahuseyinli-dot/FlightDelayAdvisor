from __future__ import annotations

import pytest

from flightdelaybench.census_validation import _verify_lineage_addendum
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json


def _self_hashed(payload: dict) -> dict:
    result = dict(payload)
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def test_lineage_addendum_requires_exact_partition_coverage(tmp_path) -> None:
    census_path = tmp_path / "census.json"
    census = _self_hashed(
        {
            "outputs": [
                {"year": 2024, "month": 1},
                {"year": 2024, "month": 2},
                {"year": 2024, "month": 3},
            ]
        }
    )
    write_canonical_json(census_path, census)
    addendum_path = tmp_path / "lineage.json"
    addendum = _self_hashed(
        {
            "status": "TEST",
            "census_manifest_sha256": sha256_file(census_path),
            "census_manifest_self_hash": census["manifest_sha256"],
            "partition_segments": [
                {
                    "start": "2024-01",
                    "end": "2024-03",
                    "partitions": 3,
                    "bts_source_sha256": "abc",
                }
            ],
        }
    )
    write_canonical_json(addendum_path, addendum)

    result = _verify_lineage_addendum(
        addendum_path,
        census_manifest=census_path,
        census=census,
    )
    assert result["partitions_covered"] == 3

    broken = dict(addendum)
    broken.pop("manifest_sha256")
    broken["partition_segments"] = [
        {
            "start": "2024-01",
            "end": "2024-02",
            "partitions": 2,
            "bts_source_sha256": "abc",
        }
    ]
    write_canonical_json(addendum_path, _self_hashed(broken))
    with pytest.raises(ValueError, match="cover each census partition"):
        _verify_lineage_addendum(
            addendum_path,
            census_manifest=census_path,
            census=census,
        )
