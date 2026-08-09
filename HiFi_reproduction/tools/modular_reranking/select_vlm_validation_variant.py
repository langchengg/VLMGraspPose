#!/usr/bin/env python3
"""Select the local VLM model/variant from full validation evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    local_vlm_preregistration_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_vlm_summary_runtime_binding,
)

VLM_VISUAL_METHOD = "repeatedfilm_local_vlm_visual"
VLM_METADATA_METHOD = "repeatedfilm_local_vlm_visual_metadata"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def semantic_input_rows(rows: list[dict]) -> list[dict]:
    """Remove only the fields intentionally changed by the metadata ablation."""

    ignored = {
        "aggregate_visual_manifest_path",
        "candidate_metadata",
        "include_metadata",
    }
    return [
        {key: value for key, value in row.items() if key not in ignored}
        for row in rows
    ]


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def parse_run(value: str) -> tuple[str, tuple[Path, Path, Path]]:
    if "=" not in value:
        raise ValueError("--variant-run requires METHOD=RESULTS,SUMMARY,RUNTIME")
    method, raw = value.split("=", 1)
    paths = tuple(Path(item).expanduser().resolve() for item in raw.split(","))
    if not method or len(paths) != 3:
        raise ValueError("--variant-run requires exactly three paths")
    return method, paths  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-bundle", type=Path, required=True)
    parser.add_argument("--variant-run", action="append", required=True)
    parser.add_argument("--model-candidate-audit", type=Path, required=True)
    parser.add_argument("--maximum-memory-mib", type=float, required=True)
    parser.add_argument("--maximum-p95-seconds", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_memory_mib <= 0 or args.maximum_p95_seconds <= 0:
        parser.error("resource limits must be positive")
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("VLM selection config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    if (
        args.maximum_memory_mib
        != preregistered["maximum_memory_mib"]
        or args.maximum_p95_seconds
        != preregistered["maximum_p95_seconds"]
    ):
        raise ValueError(
            "VLM selection resource limits disagree with preregistered config"
        )

    bundle_path = args.evaluation_bundle.expanduser().resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        bundle, context="VLM validation evaluation bundle"
    )
    if (
        bundle.get("report_recomputation_verified") is not True
        or bundle.get("candidate_pool_modified") is not False
    ):
        raise ValueError("VLM selection requires a recomputation-verified bundle")
    metrics = {
        (str(row["protocol"]), str(row["method"])): row
        for row in bundle.get("per_method_metrics", [])
    }
    provenance_sources = bundle.get("provenance", {}).get("sources", [])
    if not isinstance(provenance_sources, list):
        raise ValueError("evaluation bundle provenance sources are malformed")
    source_index: dict[tuple[str, str, str], dict] = {}
    for source in provenance_sources:
        role = str(source.get("role", ""))
        method = str(source.get("method", ""))
        protocol = str(source.get("protocol", ""))
        if role in {"vlm_prediction_and_runtime", "vlm_runtime_summary"}:
            key = (role, protocol, method)
            if key in source_index:
                raise ValueError(f"duplicate evaluation provenance source: {key}")
            source_index[key] = source
    runs = dict(parse_run(value) for value in args.variant_run)
    expected_methods = {VLM_VISUAL_METHOD, VLM_METADATA_METHOD}
    if set(runs) != expected_methods:
        raise ValueError(f"variant runs must be exactly {sorted(expected_methods)}")

    audit_path = args.model_candidate_audit.expanduser().resolve()
    model_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        model_audit, context="VLM model-candidate audit"
    )
    if model_audit.get("selection_split") not in {"val", "validation"}:
        raise ValueError("model candidate audit must be validation-only")
    if (
        Path(str(model_audit.get("config", ""))).resolve()
        != config_path
        or model_audit.get("config_sha256") != sha256_file(config_path)
        or model_audit.get("maximum_memory_mib")
        != preregistered["maximum_memory_mib"]
        or model_audit.get("maximum_p95_seconds")
        != preregistered["maximum_p95_seconds"]
    ):
        raise ValueError(
            "model candidate audit resource/config binding is invalid"
        )
    model_candidates = model_audit.get("model_candidates")
    if not isinstance(model_candidates, list) or not {
        str(item.get("size_class")) for item in model_candidates
    } >= {"4B", "8B"}:
        raise ValueError("model candidate audit must explicitly cover 4B and 8B")
    for artifact in model_audit.get("source_artifacts", []):
        path = Path(str(artifact["path"])).expanduser().resolve()
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"model candidate audit source changed: {path}")
    eligible_models = [
        item for item in model_candidates if item.get("eligible") is True
    ]
    if not eligible_models:
        raise RuntimeError("no local VLM model is validation/resource eligible")

    candidates = []
    input_rows_by_method: dict[str, list[dict]] = {}
    for method, (results_path, summary_path, runtime_path) in sorted(runs.items()):
        result_source = source_index.get(
            ("vlm_prediction_and_runtime", "gqcnn_top5", method)
        )
        summary_source = source_index.get(
            ("vlm_runtime_summary", "gqcnn_top5", method)
        )
        if result_source is None or summary_source is None:
            raise ValueError(f"evaluation bundle omits VLM provenance: {method}")
        if (
            Path(str(result_source.get("path", ""))).resolve() != results_path
            or result_source.get("sha256") != sha256_file(results_path)
            or Path(str(summary_source.get("path", ""))).resolve() != summary_path
            or summary_source.get("sha256") != sha256_file(summary_path)
            or results_path.parent != summary_path.parent
            or runtime_path.parent != summary_path.parent
        ):
            raise ValueError(
                f"VLM run artifacts disagree with evaluation provenance: {method}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        validate_artifact_identity(
            summary, context=f"VLM validation summary ({method})"
        )
        validate_artifact_identity(
            runtime, context=f"VLM validation runtime ({method})"
        )
        input_path = Path(str(summary.get("input_jsonl", ""))).resolve()
        if not input_path.is_file():
            raise ValueError(f"VLM summary omits a readable input JSONL: {method}")
        input_rows = read_jsonl(input_path)
        result_rows = read_jsonl(results_path)
        expected_metadata = method == VLM_METADATA_METHOD
        eligible_count = sum(
            bool(row.get("candidate_ids")) for row in input_rows
        )
        validate_vlm_summary_runtime_binding(
            summary,
            runtime,
            results_path=results_path,
            expected_split="validation",
            expected_formal_mode=False,
            expected_sample_count=int(bundle["sample_count_all"]),
            expected_eligible_count=eligible_count,
            expected_max_output_tokens=int(
                preregistered["max_output_tokens"]
            ),
            context=f"VLM validation run ({method})",
        )
        if (
            len(input_rows) != int(bundle["sample_count_all"])
            or any(
                bool(row.get("include_metadata")) != expected_metadata
                or row.get("gt_fields_included") is not False
                for row in input_rows
            )
        ):
            raise ValueError(f"VLM input variant semantics disagree with method: {method}")
        if len(result_rows) != len(input_rows) or any(
            str(result.get("sample_id")) != str(input_row.get("sample_id"))
            or result.get("input_record_sha256")
            != canonical_sha256(input_row)
            for result, input_row in zip(
                result_rows, input_rows, strict=True
            )
        ):
            raise ValueError(
                f"VLM results are not bound to their exact input rows: {method}"
            )
        for row in input_rows:
            candidate_ids = list(map(str, row.get("candidate_ids", [])))
            metadata = row.get("candidate_metadata")
            if not isinstance(metadata, dict) or (
                expected_metadata
                and candidate_ids
                and set(map(str, metadata)) != set(candidate_ids)
            ) or (not expected_metadata and metadata):
                raise ValueError(f"VLM input metadata contract is invalid: {method}")
        input_rows_by_method[method] = input_rows
        metric = metrics.get(("gqcnn_top5", method))
        if metric is None:
            raise ValueError(f"evaluation bundle omits VLM method: {method}")
        memory_mib = float(summary.get("memory_peak_mib", float("inf")))
        p95 = float(runtime.get("fresh_latency_p95_seconds", float("inf")))
        model_matches = [
            item
            for item in eligible_models
            if (
                str(item.get("model_digest"))
                == str(summary.get("model_digest"))
                and str(item.get("stable_session_contract_sha256"))
                == str(
                    summary.get("stable_session_contract_sha256")
                )
                and str(item.get("stable_session_contract_sha256"))
                == str(
                    runtime.get("stable_session_contract_sha256")
                )
            )
        ]
        if len(model_matches) != 1:
            raise ValueError(f"run model is not eligible in candidate audit: {method}")
        resource_pass = (
            memory_mib <= args.maximum_memory_mib
            and p95 <= args.maximum_p95_seconds
            and int(runtime.get("remote_established_connection_event_count", -1)) == 0
        )
        candidates.append(
            {
                "method": method,
                "variant": (
                    "visual_metadata" if method == VLM_METADATA_METHOD else "visual"
                ),
                "model_name": summary["model_name"],
                "model_digest": summary["model_digest"],
                "stable_session_contract_sha256": summary[
                    "stable_session_contract_sha256"
                ],
                "net_gain": int(metric["net_count"]),
                "j_at_1_all": float(metric["j_at_1_all"]),
                "memory_peak_mib": memory_mib,
                "fresh_latency_p95_seconds": p95,
                "resource_pass": resource_pass,
                "eligible": resource_pass,
                "results": str(results_path),
                "results_sha256": sha256_file(results_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "runtime_metrics": str(runtime_path),
                "runtime_metrics_sha256": sha256_file(runtime_path),
                "input_jsonl": str(input_path),
                "input_jsonl_sha256": sha256_file(input_path),
            }
        )
    if semantic_input_rows(
        input_rows_by_method[VLM_VISUAL_METHOD]
    ) != semantic_input_rows(
        input_rows_by_method[VLM_METADATA_METHOD]
    ):
        raise ValueError(
            "visual and visual-metadata validation inputs differ beyond metadata"
        )
    model_digests = {str(item["model_digest"]) for item in candidates}
    if len(model_digests) != 1:
        raise ValueError(
            "visual and visual-metadata validation runs must use one locked model"
        )
    stable_contracts = {
        str(item["stable_session_contract_sha256"])
        for item in candidates
    }
    if len(stable_contracts) != 1:
        raise ValueError(
            "visual and visual-metadata validation runs must use one "
            "stable local runtime contract"
        )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise RuntimeError("neither full validation VLM variant passes resource limits")
    selected = sorted(
        eligible,
        key=lambda item: (
            -item["net_gain"],
            -item["j_at_1_all"],
            item["fresh_latency_p95_seconds"],
            item["memory_peak_mib"],
            item["method"],
        ),
    )[0]
    selection = {
        "schema_version": 1,
        **identity_payload(),
        "selection_split": "validation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_method": selected["method"],
        "selected_variant": selected["variant"],
        "selected_model_name": selected["model_name"],
        "selected_model_digest": selected["model_digest"],
        "selected_stable_session_contract_sha256": selected[
            "stable_session_contract_sha256"
        ],
        "selection_rule": (
            "resource-eligible full-validation maximum net gain, then J@1, "
            "latency, memory, method name"
        ),
        "maximum_memory_mib": args.maximum_memory_mib,
        "maximum_p95_seconds": args.maximum_p95_seconds,
        "candidates": candidates,
        "model_candidates": model_candidates,
        "inputs": {
            "evaluation_bundle": str(bundle_path),
            "evaluation_bundle_sha256": sha256_file(bundle_path),
            "model_candidate_audit": str(audit_path),
            "model_candidate_audit_sha256": sha256_file(audit_path),
        },
    }
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    temporary = root / f".selection.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "selection.json")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
