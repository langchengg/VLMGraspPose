from __future__ import annotations

import json
import fcntl
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from failure_analysis.gemini_crog_evidence_v1.environment import (
    load_private_env,
    validate_gemini_environment,
)

from .api import MODEL_IDS
from .dataset import load_feature_index, load_label_index
from .full_list_api import (
    FULL_LIST_PROTOCOL,
    FullListInteractionsRunner,
    load_full_list_contract,
)
from .full_list_renderer import render_full_list_board
from .ledger import PairwiseLedger
from .policy import full_denominator_metrics
from .runner import (
    TRAIN_CORRECTED,
    _atomic_json,
    _exclusive_json,
    _parquet,
    _generic_runner_forbids_phase,
    acquire_global_api_lock,
    release_provider_locks,
    assert_audited_feature_source,
    assert_phase_inference_manifest_frozen,
)


@release_provider_locks
def run_full_list_phase(
    *,
    run_dir: str | Path,
    source_phase: str = "smoke",
    output_phase: str = "p2_full_list_smoke",
    env_file: str | Path,
    models: Sequence[str] = MODEL_IDS,
    max_samples: int | None = None,
    max_transport_retries: int = 1,
) -> dict[str, Any]:
    """Run label-free P2 once per sample/model over a frozen source manifest."""

    root = Path(run_dir)
    source = root / source_phase
    output = root / output_phase
    if _generic_runner_forbids_phase(source_phase) or _generic_runner_forbids_phase(output_phase):
        from .locking import FormalRunDenied

        raise FormalRunDenied("generic full-list runner cannot execute formal data")
    output.mkdir(parents=True, exist_ok=True)
    global_lock_handle = acquire_global_api_lock(root)
    lock_handle = (output / "RUNNER.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another runner already owns phase {output_phase}") from exc
    from .runner import _PROVIDER_LOCK_STATE

    _PROVIDER_LOCK_STATE.handles.append(lock_handle)
    load_private_env(env_file)
    environment = validate_gemini_environment()
    assert_phase_inference_manifest_frozen(source)
    manifest = json.loads((source / "inference_manifest.json").read_text(encoding="utf-8"))
    source_partition = assert_audited_feature_source(root, manifest["feature_file"])
    if _generic_runner_forbids_phase(source_partition):
        from .locking import FormalRunDenied

        raise FormalRunDenied("generic full-list runner cannot consume formal features")
    by_sample: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        by_sample.setdefault(str(row["sample_id"]), dict(row))
    sample_rows = [by_sample[key] for key in sorted(by_sample)]
    if max_samples is not None:
        sample_rows = sample_rows[: int(max_samples)]
    features = load_feature_index(
        manifest["feature_file"], {str(row["sample_id"]) for row in sample_rows}
    )
    contract = load_full_list_contract()
    records: list[dict[str, Any]] = []
    with PairwiseLedger(root / "pairwise_cache.sqlite") as ledger:
        runner = FullListInteractionsRunner(
            ledger=ledger,
            max_spend_usd=float(environment["gemini_max_spend_usd"]),
            er2_request_reserve_usd=float(
                environment["gemini_er2_cost_cap_per_request_usd"]
            ),
            max_transport_retries=max_transport_retries,
        )
        for model_id in models:
            observed_models: set[str] = set()
            for row in sample_rows:
                sample_id = str(row["sample_id"])
                board_png, evidence, board_metadata = render_full_list_board(
                    features[sample_id]
                )
                result = runner.run(
                    sample_id=sample_id,
                    model_id=model_id,
                    protocol=FULL_LIST_PROTOCOL,
                    evidence=evidence,
                    board_png=board_png,
                    board_metadata=board_metadata,
                    system_prompt=contract.system_prompt,
                    response_schema=contract.response_schema,
                    prompt_hash=contract.prompt_hash,
                    schema_hash=contract.schema_hash,
                    renderer_hash=str(board_metadata["renderer_contract_hash"]),
                )
                if result.response_model:
                    observed_models.add(str(result.response_model))
                    if len(observed_models) > 1 or str(result.response_model) != model_id:
                        _atomic_json(
                            output / "MODEL_DRIFT.json",
                            {
                                "phase": output_phase,
                                "requested_model": model_id,
                                "observed_models": sorted(observed_models),
                                "sample_id": sample_id,
                                "request_hash": result.request_hash,
                            },
                        )
                        raise RuntimeError("P2 provider model/version metadata drift detected")
                parsed = result.parsed.model_dump(mode="json") if result.parsed else None
                record = {
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "protocol": FULL_LIST_PROTOCOL,
                    "request_hash": result.request_hash,
                    "status": result.status,
                    "selected_candidate_id": result.selected_candidate_id,
                    "cache_hit": result.cache_hit,
                    "api_attempts": result.api_attempts,
                    "fallback_reason": result.fallback_reason,
                    "latency_seconds": result.latency_seconds,
                    "estimated_cost_usd": result.estimated_cost_usd,
                    "response_model": result.response_model,
                    "api_request_id": result.api_request_id,
                    "board_sha256": board_metadata["image_sha256"],
                    "parsed": parsed,
                }
                records.append(record)
                raw_path = root / "raw_api" / (
                    "p2_er2" if "robotics" in model_id else "p2_flash"
                ) / f"{result.request_hash}.json"
                if not raw_path.exists() and result.api_attempts > 0:
                    _exclusive_json(
                        raw_path,
                        {**record, "raw_response": result.raw_response, "usage": result.usage},
                    )
                _atomic_json(
                    output / "API_PROGRESS.json",
                    {
                        "phase": output_phase,
                        "completed": len(records),
                        "planned": len(sample_rows) * len(models),
                        "ledger": ledger.summary(),
                    },
                )
        ledger_summary = ledger.summary()
    _parquet(output / "full_list_responses.parquet", records)
    summary = {
        "phase": output_phase,
        "samples": len(sample_rows),
        "model_requests": len(records),
        "successful": sum(row["status"] == "SUCCEEDED" for row in records),
        "fallback": sum(row["status"] != "SUCCEEDED" for row in records),
        "cache_hits": sum(bool(row["cache_hit"]) for row in records),
        "ledger": ledger_summary,
    }
    _atomic_json(output / "API_SUMMARY.json", summary)
    lock_handle.close()
    global_lock_handle.close()
    return summary


def evaluate_full_list_phase(
    run_dir: str | Path,
    *,
    output_phase: str = "p2_full_list_smoke",
) -> dict[str, Any]:
    """Evaluator-only P2 diagnostic with one full-denominator outcome per sample."""

    import pyarrow.parquet as pq

    root = Path(run_dir)
    output = root / output_phase
    records = pq.read_table(output / "full_list_responses.parquet").to_pylist()
    sample_ids = {str(row["sample_id"]) for row in records}
    labels = load_label_index(TRAIN_CORRECTED, sample_ids)
    outcomes: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        selected_id = str(row.get("selected_candidate_id") or "candidate_0")
        if selected_id not in labels[sample_id]:
            selected_id = "candidate_0"
        baseline_correct = bool(labels[sample_id]["candidate_0"])
        selected_correct = bool(labels[sample_id][selected_id])
        outcomes.append(
            {
                "sample_id": sample_id,
                "model_id": row["model_id"],
                "status": row["status"],
                "baseline_candidate_id": "candidate_0",
                "selected_candidate_id": selected_id,
                "baseline_correct": baseline_correct,
                "selected_correct": selected_correct,
                "switched": selected_id != "candidate_0",
                "fallback": row["status"] != "SUCCEEDED",
            }
        )
    result: dict[str, Any] = {"phase": output_phase, "models": {}}
    for model_id in sorted({str(row["model_id"]) for row in outcomes}):
        rows = [row for row in outcomes if row["model_id"] == model_id]
        result["models"][model_id] = {
            "corrected": full_denominator_metrics(
                rows, baseline_key="baseline_correct", selected_key="selected_correct"
            ),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        }
    _parquet(output / "full_list_outcomes.parquet", outcomes)
    _atomic_json(output / "DIAGNOSTIC_RESULTS.json", result)
    return result


__all__ = ["evaluate_full_list_phase", "run_full_list_phase"]
