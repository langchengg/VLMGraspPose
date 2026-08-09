from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import DEFAULT_SEED, SCHEMA_VERSION
from .schema import artifact_identity, atomic_write_json, sha256_file


def _group_digest(group: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{group}".encode()).hexdigest()


def build_v3_split_manifest(
    v2_manifest_path: str | Path,
    output_path: str | Path,
    *,
    select_fraction: float = 0.70,
    seed: int = DEFAULT_SEED,
    group_key: str = "sequence_id",
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("V3 select/lockcheck split is immutable")
    source = Path(v2_manifest_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    validation = [row for row in payload["rows"] if row["development_partition"] == "validation"]
    groups = sorted({str(row[group_key]) for row in validation}, key=lambda group: (_group_digest(group, seed), group))
    select_count = max(1, min(len(groups) - 1, round(len(groups) * float(select_fraction))))
    select_groups = set(groups[:select_count])
    rows = []
    for row in validation:
        partition = "v3_select" if str(row[group_key]) in select_groups else "v3_lockcheck"
        rows.append({**row, "v3_partition": partition})
    partitions = {name: [row for row in rows if row["v3_partition"] == name] for name in ("v3_select", "v3_lockcheck")}
    group_sets = {name: {str(row[group_key]) for row in part} for name, part in partitions.items()}
    frame_sets = {name: {str(row["frame_id"]) for row in part} for name, part in partitions.items()}
    sample_sets = {name: {str(row["sample_id"]) for row in part} for name, part in partitions.items()}
    audit = {
        "partition_counts": {
            name: {
                "expressions": len(part),
                "groups": len(group_sets[name]),
                "frames": len(frame_sets[name]),
                "sequences": len({str(row["sequence_id"]) for row in part}),
            }
            for name, part in partitions.items()
        },
        "group_overlap": len(group_sets["v3_select"] & group_sets["v3_lockcheck"]),
        "frame_overlap": len(frame_sets["v3_select"] & frame_sets["v3_lockcheck"]),
        "sample_overlap": len(sample_sets["v3_select"] & sample_sets["v3_lockcheck"]),
        "all_validation_rows_assigned_once": len(rows) == len(validation) == len(set().union(*sample_sets.values())),
    }
    if any(audit[key] for key in ("group_overlap", "frame_overlap", "sample_overlap")):
        raise AssertionError(f"V3 partition overlap: {audit}")
    if not audit["all_validation_rows_assigned_once"]:
        raise AssertionError("V3 validation assignment is incomplete or duplicated")
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "immutable_v3_select_lockcheck_split",
        "source_v2_split_manifest": artifact_identity(source),
        "source_partition": "validation",
        "group_key": group_key,
        "assignment": "sort groups by sha256(seed:group), first round(70%) to select",
        "seed": int(seed),
        "select_fraction": float(select_fraction),
        "audit": audit,
        "rows": rows,
    }
    atomic_write_json(output, result)
    from .artifacts import write_sha_sidecar
    write_sha_sidecar(output)
    return result


def verify_v3_split_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(source) != expected:
        raise ValueError("V3 split SHA mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if any(payload["audit"][key] for key in ("group_overlap", "frame_overlap", "sample_overlap")):
        raise ValueError("V3 split no longer has zero overlap")
    return payload

