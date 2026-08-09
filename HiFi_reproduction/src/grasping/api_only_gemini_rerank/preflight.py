"""Request/cost planning and fail-closed paid-run authorization."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import atomic_json, utc_now


def planned_request_counts(run_dir: str | Path) -> dict[str, Any]:
    cohorts = pd.read_parquet(Path(run_dir) / "STAGE_COHORTS.parquet")
    sizes = cohorts.groupby(["stage", "backend"]).size().to_dict()
    smoke = sum(sizes.get(("smoke", backend), 0) for backend in ("G1", "C1")) * 2
    diagnostic_base = sum(sizes.get(("diagnostic", backend), 0) for backend in ("G1", "C1")) * 4 * 2
    diagnostic_perturb = sum(min(30, sizes.get(("diagnostic", backend), 0)) for backend in ("G1", "C1")) * 2 * 2
    policy_base = sum(sizes.get(("policy_selection", backend), 0) for backend in ("G1", "C1")) * 2 * 2
    policy_confirmation_upper = sum(sizes.get(("policy_selection", backend), 0) for backend in ("G1", "C1")) * 2
    validation_base = sum(sizes.get(("untouched_validation", backend), 0) for backend in ("G1", "C1")) * 2
    validation_confirmation_upper = validation_base
    development_upper = smoke + diagnostic_base + diagnostic_perturb + policy_base + policy_confirmation_upper + validation_base + validation_confirmation_upper
    # Every development protocol is planned symmetrically for the two exact
    # model IDs, so half of the conservative upper bound belongs to ER2.
    development_er2_upper = development_upper // 2
    return {
        "smoke": int(smoke), "diagnostic_original": int(diagnostic_base),
        "diagnostic_additional_perturbations": int(diagnostic_perturb),
        "policy_p1_p2": int(policy_base), "policy_conditional_confirmation_upper": int(policy_confirmation_upper),
        "untouched_validation_primary": int(validation_base),
        "untouched_validation_conditional_confirmation_upper": int(validation_confirmation_upper),
        "development_unique_request_upper_bound": int(development_upper),
        "development_er2_request_upper_bound": int(development_er2_upper),
        "formal_excluded_until_go_and_dual_authorization": True,
    }


def create_preflight(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    counts = planned_request_counts(run)
    max_cost = os.environ.get("MAX_API_COST_USD")
    max_requests = os.environ.get("MAX_PROVIDER_REQUESTS")
    er2_reserve = os.environ.get("GEMINI_ER2_COST_CAP_PER_REQUEST_USD")
    key_present = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    paid_flag = os.environ.get("ALLOW_PAID_API_RUN") == "1"
    blockers = []

    def positive_finite(raw: str | None, *, integer: bool = False) -> float | int | None:
        if raw is None:
            return None
        try:
            value = int(raw) if integer else float(raw)
        except (TypeError, ValueError):
            return None
        if value <= 0 or (not integer and not math.isfinite(value)):
            return None
        return value

    max_cost_value = positive_finite(max_cost)
    max_requests_value = positive_finite(max_requests, integer=True)
    er2_reserve_value = positive_finite(er2_reserve)
    if not key_present:
        blockers.append("API credential environment variable is absent")
    if max_cost is None:
        blockers.append("MAX_API_COST_USD is absent")
    elif max_cost_value is None:
        blockers.append("MAX_API_COST_USD must be positive and finite")
    if max_requests is None:
        blockers.append("MAX_PROVIDER_REQUESTS is absent")
    elif max_requests_value is None:
        blockers.append("MAX_PROVIDER_REQUESTS must be a positive integer")
    if er2_reserve is None:
        blockers.append("ER2 price is unverified and GEMINI_ER2_COST_CAP_PER_REQUEST_USD is absent")
    elif er2_reserve_value is None:
        blockers.append("GEMINI_ER2_COST_CAP_PER_REQUEST_USD must be positive and finite")
    if not paid_flag:
        blockers.append("ALLOW_PAID_API_RUN is not 1")
    if max_requests_value is not None and max_requests_value < counts["development_unique_request_upper_bound"]:
        blockers.append("MAX_PROVIDER_REQUESTS is below the conservative development request upper bound")
    er2_development_reserve = None
    if er2_reserve_value is not None:
        er2_development_reserve = float(er2_reserve_value) * counts["development_er2_request_upper_bound"]
    if max_cost_value is not None and er2_development_reserve is not None and er2_development_reserve > float(max_cost_value):
        blockers.append("ER2 conservative development reserve alone exceeds MAX_API_COST_USD")
    pricing = {
        "as_of_utc": utc_now(),
        "gemini-3.6-flash": {
            "standard_input_usd_per_million_tokens": 1.50,
            "standard_output_including_thinking_usd_per_million_tokens": 7.50,
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "provider_invoice_verified": False,
        },
        "gemini-robotics-er-2-preview": {
            "monetary_cost": None, "public_price_independently_verified": False,
            "budget_rule": "logical attempts * GEMINI_ER2_COST_CAP_PER_REQUEST_USD",
            "provider_invoice_verified": False,
        },
    }
    atomic_json(run / "pricing_manifest.json", pricing)
    report = {
        "created_at_utc": utc_now(), "planned_requests": counts,
        "credential_present": key_present, "paid_authorized": paid_flag,
        "max_api_cost_usd_configured": max_cost is not None,
        "max_provider_requests_configured": max_requests is not None,
        "er2_per_request_reserve_configured": er2_reserve is not None,
        "max_api_cost_usd_valid": max_cost_value is not None,
        "max_provider_requests_valid": max_requests_value is not None,
        "er2_per_request_reserve_valid": er2_reserve_value is not None,
        "er2_development_reserve_fits_total_cap": bool(
            max_cost_value is not None
            and er2_development_reserve is not None
            and er2_development_reserve <= float(max_cost_value)
        ),
        "ready_for_paid_api": not blockers, "blockers": blockers,
        "secrets_recorded": False,
    }
    atomic_json(run / "audit/api_preflight.json", report)
    lines = [
        "# Preflight cost estimate", "", "No API credential value or identifying characteristic is recorded.", "",
        f"- Paid API ready: **{str(not blockers).lower()}**",
        f"- Development unique-request upper bound: **{counts['development_unique_request_upper_bound']:,}**",
        f"- Development ER2 request upper bound: **{counts['development_er2_request_upper_bound']:,}**",
        f"- ER2 conservative development reserve fits total cap: **{str(bool(max_cost_value is not None and er2_development_reserve is not None and er2_development_reserve <= float(max_cost_value))).lower()}**",
        f"- Flash pricing basis: $1.50/M input and $7.50/M output including thinking (standard tier, checked 2026-08-05).",
        "- ER2 monetary price: not independently verified; budget uses a configurable conservative per-request reserve.",
        "- Batch decision: not selected because the documented Batch path is generateContent and exact Interactions-contract parity for both requested model IDs was not established.",
        "", "## Hard blockers", "",
    ]
    lines.extend([f"- {item}" for item in blockers] or ["- None"])
    (run / "PREFLIGHT_COST_ESTIMATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def require_paid_preflight(report: Mapping[str, Any]) -> None:
    if not report.get("ready_for_paid_api"):
        raise RuntimeError("paid API preflight failed: " + "; ".join(map(str, report.get("blockers", []))))
