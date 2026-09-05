"""Create or check the compact experiment navigation index without loading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "manifests/research_experiment_index_v1.json"
MARKDOWN = "docs/EXPERIMENT_INDEX.md"


def evidence_class(name: str) -> str:
    if name.startswith("handoff_continuation_"):
        return "Stopped corrected-campaign register; no real-data trials"
    if name.startswith("flare24_boundary_"):
        return "Boundary retrospective proxy"
    if name.startswith("flare24_ccrth_"):
        return "Airport-resource retrospective proxy"
    if name.startswith("flare24_"):
        return "FLARE selection / retrospective proxy"
    if name.startswith("forecast24_"):
        return "Forecast adaptation / retrospective proxy"
    if name.endswith(".md"):
        return "Historical limitation record"
    return "Historical development / screening"


def experiment_sort_key(path: PurePath) -> tuple[str, str]:
    """Keep manifest ordering identical on case-sensitive and insensitive platforms."""
    return path.name.casefold(), path.name


def build_index(root: Path) -> tuple[dict[str, Any], str]:
    directory = root / "reports/experiments"
    records = []
    for path in sorted(directory.iterdir(), key=experiment_sort_key):
        if path.suffix not in {".json", ".md"}:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unsafe experiment path: {path.name}")
        payload = path.read_bytes()
        status = "NARRATIVE_RECORD" if path.suffix == ".md" else "NOT_RECORDED_AT_TOP_LEVEL"
        if path.suffix == ".json":
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"experiment must be a JSON object: {path.name}")
            status = str(data.get("status", status))
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "evidence_class": evidence_class(path.name), "recorded_status": status,
        })
    if not records:
        raise ValueError("no experiment records found")
    result = {
        "schema_version": 1,
        "scope": "Every direct JSON/Markdown file in reports/experiments; console logs and external run payloads are retained separately.",
        "interpretation": "Indexing does not validate a model, certify availability, or promote a recorded historical PASS to current scientific acceptance.",
        "files": records,
    }
    lines = [
        "# Complete experiment index", "",
        "[Result lineage](RESULT_LINEAGE.md) · [Current results](CURRENT_RESULTS.md)", "",
        f"This index contains {len(records)} JSON/Markdown experiment records. It includes",
        "development screens, negative results, historical proxy studies and the stopped",
        "corrected-campaign register. Presence here does not imply a completed fit or",
        "corrected T\N{MINUS SIGN}24 validity. The index does not read prediction rows or model files.", "",
        "Exact bytes, SHA-256 and original top-level statuses are recorded in the",
        f"[machine-readable inventory](../{MANIFEST}). Original report names are stable",
        "provenance identifiers. Use result lineage for readable study descriptions.", "",
        "| Record | Evidence class |", "|---|---|",
    ]
    for record in records:
        name = Path(record["path"]).name
        lines.append(f"| [{name}](../{record['path']}) | {record['evidence_class']} |")
    lines += [
        "", "## Verification and later additions", "",
        "Run `python tools/build_experiment_index.py --check` to verify completeness,",
        "bytes, hashes and the rendered index. No source report is modified.", "",
        "Create a new inventory/index revision when adding experiment records; preserve",
        "the previous inventory as evidence. Update the current references and checks",
        "together. The generator refuses to overwrite existing outputs.", "",
        "Source-access pilots, schema/coverage checks, package validation and failures",
        "are separately indexed in [artifacts](ARTIFACTS.md) and",
        "[result lineage](RESULT_LINEAGE.md). Private console logs, model/prediction",
        "partitions and older source archives remain in their original run/transfer tiers.", "",
    ]
    return result, "\n".join(lines)


def encoded_manifest(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def check_index(root: Path) -> int:
    manifest, markdown = build_index(root)
    if (root / MANIFEST).read_text(encoding="utf-8") != encoded_manifest(manifest):
        raise ValueError("experiment inventory differs: missing/new record or changed bytes/status")
    if (root / MARKDOWN).read_text(encoding="utf-8") != markdown:
        raise ValueError("experiment Markdown index differs")
    return len(manifest["files"])


def write_index(root: Path) -> int:
    manifest_path, markdown_path = root / MANIFEST, root / MARKDOWN
    if manifest_path.exists() or markdown_path.exists():
        raise FileExistsError("index outputs are create-only; preserve existing revisions")
    manifest, markdown = build_index(root)
    for path, content in ((manifest_path, encoded_manifest(manifest)), (markdown_path, markdown)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return len(manifest["files"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count = check_index(ROOT) if args.check else write_index(ROOT)
    print(f"Experiment index {'verified' if args.check else 'created'}: {count} records.")


if __name__ == "__main__":
    main()
