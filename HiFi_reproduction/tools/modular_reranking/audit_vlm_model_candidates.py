#!/usr/bin/env python3
"""Create a validation-only eligibility audit for the 4B/8B VLM candidates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    SESSION_AUDIT_POLICY_VERSION,
    stable_session_contract,
    stable_session_contract_sha256,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    validate_artifact_identity,
    validate_config_identity,
    local_vlm_preregistration_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--four-b-local-audit", type=Path, required=True)
    parser.add_argument("--four-b-pilot-summary", type=Path, required=True)
    parser.add_argument("--four-b-pilot-runtime", type=Path, required=True)
    parser.add_argument("--four-b-repeat-a", type=Path, required=True)
    parser.add_argument("--four-b-repeat-b", type=Path, required=True)
    parser.add_argument("--four-b-repeat-a-runtime", type=Path, required=True)
    parser.add_argument("--four-b-repeat-b-runtime", type=Path, required=True)
    parser.add_argument("--four-b-repeat-comparison", type=Path, required=True)
    parser.add_argument("--eight-b-local-audit", type=Path)
    parser.add_argument("--eight-b-pilot-summary", type=Path)
    parser.add_argument("--eight-b-pilot-runtime", type=Path)
    parser.add_argument("--eight-b-repeat-a", type=Path)
    parser.add_argument("--eight-b-repeat-b", type=Path)
    parser.add_argument("--eight-b-repeat-a-runtime", type=Path)
    parser.add_argument("--eight-b-repeat-b-runtime", type=Path)
    parser.add_argument("--eight-b-repeat-comparison", type=Path)
    parser.add_argument("--maximum-memory-mib", type=float, default=20_000.0)
    parser.add_argument("--maximum-p95-seconds", type=float, default=180.0)
    parser.add_argument(
        "--minimum-repeat-agreement", type=float, default=0.90
    )
    parser.add_argument(
        "--minimum-valid-structured-rate", type=float, default=0.90
    )
    parser.add_argument("--maximum-fallback-rate", type=float, default=0.10)
    parser.add_argument(
        "--ollama", type=Path, default=Path("/opt/homebrew/bin/ollama")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_count(path: Path) -> int:
    return sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _remote_connection_evidence_is_clean(evidence: dict[str, Any]) -> bool:
    events = evidence.get("remote_established_connection_events")
    return (
        isinstance(events, list)
        and int(
            evidence.get(
                "remote_established_connection_event_count", -1
            )
        )
        == len(events)
        and not events
    )


def _source(path: Path, role: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"role": role, "path": str(resolved), "sha256": _sha256(resolved)}


def _candidate(
    *,
    size_class: str,
    local_audit_path: Path | None,
    pilot_summary_path: Path | None,
    pilot_runtime_path: Path | None,
    repeat_a_path: Path | None,
    repeat_b_path: Path | None,
    repeat_a_runtime_path: Path | None,
    repeat_b_runtime_path: Path | None,
    repeat_comparison_path: Path | None,
    installed_names: set[str],
    maximum_memory_mib: float,
    maximum_p95_seconds: float,
    max_output_tokens: int,
    minimum_repeat_agreement: float,
    minimum_valid_structured_rate: float,
    maximum_fallback_rate: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    expected_prefix = f"qwen3-vl:{size_class.lower()}"
    installed = sorted(
        name for name in installed_names if name.startswith(expected_prefix)
    )
    sources: list[dict[str, str]] = []
    missing_evidence: list[str] = []
    for role, path in (
        ("local_audit", local_audit_path),
        ("pilot_summary", pilot_summary_path),
        ("pilot_runtime", pilot_runtime_path),
        ("repeat_a", repeat_a_path),
        ("repeat_b", repeat_b_path),
        ("repeat_a_runtime", repeat_a_runtime_path),
        ("repeat_b_runtime", repeat_b_runtime_path),
        ("repeat_comparison", repeat_comparison_path),
    ):
        if path is None:
            missing_evidence.append(role)
        else:
            sources.append(_source(path, f"{size_class}_{role}"))

    model_name = installed[0] if len(installed) == 1 else None
    model_digest = None
    pilot_samples = 0
    eligible_pilot_samples = 0
    memory_peak_mib = None
    p95_seconds = None
    local_only = False
    repeat_samples = 0
    repeat_selected_agreement = None
    repeat_ranking_agreement = None
    repeat_json_agreement = None
    repeat_fallback_agreement = None
    valid_structured_rate = None
    fallback_rate = None
    quality_pass = False
    evidence_consistent = False
    stable_contract_valid = False
    stable_contract_hash = None
    installed_identity_pass = False
    if not missing_evidence:
        assert local_audit_path is not None
        assert pilot_summary_path is not None
        assert pilot_runtime_path is not None
        assert repeat_a_path is not None
        assert repeat_b_path is not None
        assert repeat_a_runtime_path is not None
        assert repeat_b_runtime_path is not None
        assert repeat_comparison_path is not None
        local_audit = _read_json(local_audit_path.resolve())
        summary = _read_json(pilot_summary_path.resolve())
        runtime = _read_json(pilot_runtime_path.resolve())
        repeat = _read_json(repeat_comparison_path.resolve())
        repeat_a_runtime = _read_json(repeat_a_runtime_path.resolve())
        repeat_b_runtime = _read_json(repeat_b_runtime_path.resolve())
        for context, artifact in (
            ("VLM pilot summary", summary),
            ("VLM pilot runtime", runtime),
            ("VLM repeat comparison", repeat),
            ("VLM repeat A runtime", repeat_a_runtime),
            ("VLM repeat B runtime", repeat_b_runtime),
        ):
            validate_artifact_identity(artifact, context=context)
        repeat_inputs = repeat.get("inputs", {})
        audited_name = str(local_audit.get("model", {}).get("exact_name", ""))
        model_name = audited_name or model_name
        manifest_digest = str(
            local_audit.get("model", {}).get("manifest_sha256", "")
        )
        model_digest = f"sha256:{manifest_digest}" if manifest_digest else None
        try:
            stable_contract_hash = stable_session_contract_sha256(
                local_audit
            )
            stable_contract_valid = (
                int(local_audit.get("session_audit_policy_version", -1))
                == SESSION_AUDIT_POLICY_VERSION
                and local_audit.get("stable_session_contract")
                == stable_session_contract(local_audit)
                and local_audit.get("stable_session_contract_sha256")
                == stable_contract_hash
            )
        except (KeyError, TypeError, ValueError):
            stable_contract_valid = False
        installed_identity_pass = (
            len(installed) == 1
            and audited_name == installed[0]
            and audited_name.startswith(expected_prefix)
        )
        pilot_samples = int(summary.get("sample_count", -1))
        eligible_pilot_samples = int(
            summary.get("eligible_sample_count", -1)
        )
        memory_peak_mib = float(summary.get("memory_peak_mib", float("inf")))
        p95_seconds = float(
            runtime.get("fresh_latency_p95_seconds", float("inf"))
        )
        valid_structured_rate = float(
            runtime.get("valid_structured_response_rate", -1.0)
        )
        fallback_rate = float(runtime.get("fallback_rate", float("inf")))
        local_only = (
            local_audit.get("local_only") is True
            and local_audit.get("remote_api_used") is False
            and runtime.get("local_only_runtime_passed") is True
            and _remote_connection_evidence_is_clean(runtime)
        )
        repeat_samples = int(repeat.get("samples", -1))
        repeat_selected_agreement = float(
            repeat.get("selected_candidate_agreement_rate", -1.0)
        )
        repeat_ranking_agreement = float(
            repeat.get("ranking_id_agreement_rate", -1.0)
        )
        repeat_json_agreement = float(
            repeat.get("exact_parsed_json_agreement_rate", -1.0)
        )
        repeat_fallback_agreement = float(
            repeat.get("fallback_agreement_rate", -1.0)
        )

        def result_binding(
            evidence: dict[str, Any], result_path: Path
        ) -> bool:
            return (
                Path(str(evidence.get("results_jsonl", ""))).resolve()
                == result_path.resolve()
                and evidence.get("results_jsonl_sha256")
                == _sha256(result_path)
                and evidence.get("model_name") == audited_name
                and evidence.get("model_digest") == model_digest
                and evidence.get("stable_session_contract_sha256")
                == stable_contract_hash
                and evidence.get("input_split") == "validation"
            )

        pilot_result_path = Path(
            str(summary.get("results_jsonl", ""))
        ).resolve()
        pilot_result_bound = (
            pilot_result_path.is_file()
            and result_binding(summary, pilot_result_path)
            and result_binding(runtime, pilot_result_path)
            and Path(str(runtime.get("input_jsonl", ""))).resolve()
            == Path(str(summary.get("input_jsonl", ""))).resolve()
            and runtime.get("input_jsonl_sha256")
            == summary.get("input_jsonl_sha256")
        )
        repeat_runtime_pass = all(
            item.get("local_only_runtime_passed") is True
            and _remote_connection_evidence_is_clean(item)
            and int(item.get("sample_count", -1)) == 20
            and int(item.get("eligible_sample_count", -1)) == 20
            and int(item.get("fresh_http_call_count", -1)) == 20
            and int(item.get("cache_hit_count", -1)) == 0
            and int(item.get("max_output_tokens", -1))
            == int(max_output_tokens)
            and result_binding(item, result_path)
            for item, result_path in (
                (repeat_a_runtime, repeat_a_path),
                (repeat_b_runtime, repeat_b_path),
            )
        )
        quality_pass = (
            valid_structured_rate >= minimum_valid_structured_rate
            and fallback_rate <= maximum_fallback_rate
            and repeat_selected_agreement >= minimum_repeat_agreement
            and repeat_ranking_agreement >= minimum_repeat_agreement
            and repeat_json_agreement >= minimum_repeat_agreement
            and repeat_fallback_agreement >= minimum_repeat_agreement
            and repeat_runtime_pass
        )
        evidence_consistent = (
            installed_identity_pass
            and stable_contract_valid
            and summary.get("model_name") == audited_name
            and summary.get("model_digest") == model_digest
            and summary.get("formal_mode") is False
            and runtime.get("formal_mode") is False
            and int(summary.get("max_output_tokens", -1))
            == int(max_output_tokens)
            and int(runtime.get("max_output_tokens", -1))
            == int(max_output_tokens)
            and pilot_result_bound
            and int(runtime.get("sample_count", -1)) == pilot_samples
            and int(runtime.get("eligible_sample_count", -1))
            == eligible_pilot_samples
            and pilot_samples == 100
            and _jsonl_count(pilot_result_path) == pilot_samples
            and eligible_pilot_samples >= 0
            and int(summary.get("newly_processed", -1))
            == eligible_pilot_samples
            and int(summary.get("cache_hits", -1)) == 0
            and repeat_samples == 20
            and _jsonl_count(repeat_a_path) == repeat_samples
            and _jsonl_count(repeat_b_path) == repeat_samples
            and repeat.get("both_runs_cache_cold") is True
            and repeat.get("request_hash_agreement_rate") == 1.0
            and quality_pass
            and Path(str(repeat_inputs.get("run_a", ""))).resolve()
            == repeat_a_path.resolve()
            and repeat_inputs.get("run_a_sha256") == _sha256(repeat_a_path)
            and Path(str(repeat_inputs.get("run_b", ""))).resolve()
            == repeat_b_path.resolve()
            and repeat_inputs.get("run_b_sha256") == _sha256(repeat_b_path)
        )

    resource_pass = (
        memory_peak_mib is not None
        and p95_seconds is not None
        and memory_peak_mib <= maximum_memory_mib
        and p95_seconds <= maximum_p95_seconds
    )
    eligible = (
        installed_identity_pass
        and not missing_evidence
        and local_only
        and evidence_consistent
        and resource_pass
    )
    reasons = []
    if len(installed) != 1:
        reasons.append(
            "not installed exactly once"
            if not installed
            else "ambiguous installed tags"
        )
    elif not installed_identity_pass and not missing_evidence:
        reasons.append("installed model and audited model identity differ")
    if missing_evidence:
        reasons.append("missing " + ", ".join(missing_evidence))
    if not local_only and not missing_evidence:
        reasons.append("local-only audit failed")
    if not evidence_consistent and not missing_evidence:
        reasons.append("pilot/repeat evidence inconsistent or incomplete")
    if not quality_pass and not missing_evidence:
        reasons.append(
            "parser/fallback/stability or repeat local-only gate failed"
        )
    if not resource_pass and not missing_evidence:
        reasons.append("pilot resource limit failed")
    return (
        {
            "size_class": size_class,
            "expected_tag_prefix": expected_prefix,
            "installed_matches": installed,
            "installed": len(installed) == 1,
            "model_name": model_name,
            "model_digest": model_digest,
            "validation_pilot_sample_count": pilot_samples,
            "validation_pilot_eligible_sample_count": (
                eligible_pilot_samples
            ),
            "deterministic_repeat_sample_count": repeat_samples,
            "selected_candidate_agreement_rate": repeat_selected_agreement,
            "ranking_id_agreement_rate": repeat_ranking_agreement,
            "exact_parsed_json_agreement_rate": repeat_json_agreement,
            "fallback_agreement_rate": repeat_fallback_agreement,
            "minimum_repeat_agreement": minimum_repeat_agreement,
            "valid_structured_response_rate": valid_structured_rate,
            "minimum_valid_structured_response_rate": (
                minimum_valid_structured_rate
            ),
            "fallback_rate": fallback_rate,
            "maximum_fallback_rate": maximum_fallback_rate,
            "quality_and_repeat_runtime_pass": quality_pass,
            "memory_peak_mib": memory_peak_mib,
            "fresh_latency_p95_seconds": p95_seconds,
            "max_output_tokens": int(max_output_tokens),
            "local_only": local_only,
            "stable_contract_valid": stable_contract_valid,
            "stable_session_contract_sha256": stable_contract_hash,
            "installed_identity_pass": installed_identity_pass,
            "resource_pass": resource_pass,
            "eligible": eligible,
            "exclusion_reasons": reasons,
        },
        sources,
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("VLM model audit config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    if args.maximum_memory_mib <= 0 or args.maximum_p95_seconds <= 0:
        raise ValueError("resource limits must be positive")
    if (
        args.maximum_memory_mib
        != preregistered["maximum_memory_mib"]
        or args.maximum_p95_seconds
        != preregistered["maximum_p95_seconds"]
    ):
        raise ValueError(
            "VLM model audit resource limits disagree with preregistered config"
        )
    for name, value in (
        ("minimum-repeat-agreement", args.minimum_repeat_agreement),
        ("minimum-valid-structured-rate", args.minimum_valid_structured_rate),
        ("maximum-fallback-rate", args.maximum_fallback_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    ollama_list = subprocess.run(
        [str(args.ollama.expanduser().resolve()), "list"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    installed_names = {
        line.split()[0]
        for line in ollama_list.splitlines()[1:]
        if line.split()
    }
    four_b, four_sources = _candidate(
        size_class="4B",
        local_audit_path=args.four_b_local_audit,
        pilot_summary_path=args.four_b_pilot_summary,
        pilot_runtime_path=args.four_b_pilot_runtime,
        repeat_a_path=args.four_b_repeat_a,
        repeat_b_path=args.four_b_repeat_b,
        repeat_a_runtime_path=args.four_b_repeat_a_runtime,
        repeat_b_runtime_path=args.four_b_repeat_b_runtime,
        repeat_comparison_path=args.four_b_repeat_comparison,
        installed_names=installed_names,
        maximum_memory_mib=args.maximum_memory_mib,
        maximum_p95_seconds=args.maximum_p95_seconds,
        max_output_tokens=int(preregistered["max_output_tokens"]),
        minimum_repeat_agreement=args.minimum_repeat_agreement,
        minimum_valid_structured_rate=args.minimum_valid_structured_rate,
        maximum_fallback_rate=args.maximum_fallback_rate,
    )
    eight_b, eight_sources = _candidate(
        size_class="8B",
        local_audit_path=args.eight_b_local_audit,
        pilot_summary_path=args.eight_b_pilot_summary,
        pilot_runtime_path=args.eight_b_pilot_runtime,
        repeat_a_path=args.eight_b_repeat_a,
        repeat_b_path=args.eight_b_repeat_b,
        repeat_a_runtime_path=args.eight_b_repeat_a_runtime,
        repeat_b_runtime_path=args.eight_b_repeat_b_runtime,
        repeat_comparison_path=args.eight_b_repeat_comparison,
        installed_names=installed_names,
        maximum_memory_mib=args.maximum_memory_mib,
        maximum_p95_seconds=args.maximum_p95_seconds,
        max_output_tokens=int(preregistered["max_output_tokens"]),
        minimum_repeat_agreement=args.minimum_repeat_agreement,
        minimum_valid_structured_rate=args.minimum_valid_structured_rate,
        maximum_fallback_rate=args.maximum_fallback_rate,
    )
    value = {
        "schema_version": 1,
        **identity_payload(),
        "selection_split": "validation",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "selection_rule": (
            "model must be installed, local-only, complete the 100-sample "
            "validation pilot and two independent cache-cold 20-sample repeats, "
            "pass parser/fallback/repeat-locality/stability gates, and pass "
            "predeclared memory/latency limits"
        ),
        "maximum_memory_mib": args.maximum_memory_mib,
        "maximum_p95_seconds": args.maximum_p95_seconds,
        "max_output_tokens": int(preregistered["max_output_tokens"]),
        "minimum_repeat_agreement": args.minimum_repeat_agreement,
        "minimum_valid_structured_rate": (
            args.minimum_valid_structured_rate
        ),
        "maximum_fallback_rate": args.maximum_fallback_rate,
        "model_candidates": [four_b, eight_b],
        "source_artifacts": [*four_sources, *eight_sources],
        "ollama_list": ollama_list,
        "test_evidence_used": False,
    }
    if not any(item["eligible"] for item in value["model_candidates"]):
        raise RuntimeError("no validation/resource-eligible local VLM candidate")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
