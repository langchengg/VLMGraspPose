"""Deterministically materialise frozen final outcomes from locked formal runs.

This producer has no caller-selectable source paths or outcome bits.  It
resolves the unique formal decision artifacts from the protocol's immutable
source inventories, copies only the already-formal ``selected_correct`` bits,
and delegates the independent row-for-row verification to the postprocess
authority writer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import artifact_record, atomic_parquet
from .postprocess import (
    EXPECTED_SAMPLE_COUNT,
    FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH,
    PostprocessContractError,
    ROUTES,
    _locked_source_inventory,
    write_final_outcomes_authority,
)
from .contracts import RunState
from .protocol import LOCK_RELATIVE_PATH, verify_protocol_lock


FINAL_OUTCOMES_RELATIVE_PATH = Path(
    "04_predicted_replay/frozen_final_outcomes.parquet"
)
UNIFIED_DECISIONS_SUFFIX = "09_formal_test/per_sample_decisions.parquet"
D1_DECISIONS_SUFFIX = (
    "09_formal_test/formal_candidate_score_decision_bundle.parquet"
)
SYSTEMS = {
    "G1": "g1_gated_primary",
    "C1": "c1_gated_primary",
    "D1": "d1_top5_r7_gated",
}


def _record_for_suffix(
    inventory: set[tuple[str, str, int]], *, suffix: str
) -> dict[str, Any]:
    matches = [identity for identity in inventory if Path(identity[0]).as_posix().endswith(suffix)]
    if len(matches) != 1:
        raise PostprocessContractError(
            f"locked source inventory must contain one {suffix}; found {len(matches)}"
        )
    path_text, digest, byte_count = matches[0]
    path = Path(path_text).expanduser().resolve()
    observed = artifact_record(path)
    expected = {"path": str(path), "sha256": digest, "bytes": byte_count}
    if observed != expected:
        raise PostprocessContractError(
            f"locked formal source bytes differ: {suffix}"
        )
    return expected


def _selector_rows(
    source: Path, *, route: str, system: str, denominator: set[str]
) -> pd.DataFrame:
    names = set(pd.read_parquet(source, engine="pyarrow", columns=[]).columns)
    # pandas returns no schema columns for columns=[] on some versions; use
    # pyarrow metadata without opening row groups in that case.
    if not names:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(source).names)
    required = {"sample_id", "system_name", "selected_correct"}
    if not required.issubset(names):
        raise PostprocessContractError(
            f"{route} frozen selector source misses {sorted(required.difference(names))}"
        )
    columns = sorted(required | ({"is_selected", "no_output"} & names))
    frame = pd.read_parquet(source, columns=columns)
    frame = frame.loc[frame["system_name"].astype(str).eq(system)].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any() and "is_selected" in frame:
        keep = frame["is_selected"].astype(bool)
        if "no_output" in frame:
            keep |= frame["no_output"].astype(bool)
        frame = frame.loc[keep].copy()
    if frame["sample_id"].duplicated().any() or set(frame["sample_id"]) != denominator:
        raise PostprocessContractError(
            f"{route} frozen selector does not exactly cover the denominator"
        )
    values = pd.to_numeric(frame["selected_correct"], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise PostprocessContractError(
            f"{route} frozen selected_correct is not binary"
        )
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "route": route,
            "final_correct": values.astype(bool),
        }
    ).sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _publish_exact(path: Path, frame: pd.DataFrame, *, resume: bool) -> Path:
    if path.exists():
        if not resume:
            raise FileExistsError(f"frozen final outcomes exist; pass --resume: {path}")
        observed = pd.read_parquet(path).sort_values(
            ["route", "sample_id"], kind="mergesort"
        ).reset_index(drop=True)
        expected = frame.sort_values(
            ["route", "sample_id"], kind="mergesort"
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                observed, expected, check_dtype=False, check_exact=True
            )
        except AssertionError as error:
            raise PostprocessContractError(
                "existing frozen final outcomes differ from locked formal sources"
            ) from error
        return path
    return atomic_parquet(frame, path)


def materialize_frozen_final_outcomes(
    run_dir: str | Path,
    *,
    resume: bool = False,
    routes: Sequence[str] | None = None,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> tuple[Path, Path]:
    """Create the unique canonical outcome table and its locked authority."""

    root = Path(run_dir).expanduser().resolve()
    lock_path = root / LOCK_RELATIVE_PATH
    lock = verify_protocol_lock(root)
    if routes is None:
        pipeline_path = root / "pipeline_status.json"
        pipeline = (
            json.loads(pipeline_path.read_text(encoding="utf-8"))
            if pipeline_path.is_file()
            else {}
        )
        routes = (
            ("G1", "C1")
            if pipeline.get("status") == RunState.P5B_G1_FULL_COMPLETE.value
            else ROUTES
        )
    route_names = tuple(str(route).upper() for route in routes)
    if route_names not in {ROUTES, ("G1", "C1")}:
        raise ValueError("final outcome routes must be G1/C1 or G1/C1/D1")
    sample_record = lock.get("sample_manifest")
    if not isinstance(sample_record, Mapping):
        raise PostprocessContractError("protocol lock lacks canonical sample manifest")
    sample_path = Path(str(sample_record.get("path", ""))).expanduser().resolve()
    if artifact_record(sample_path) != dict(sample_record):
        raise PostprocessContractError("protocol sample manifest bytes differ")
    samples = pd.read_parquet(sample_path, columns=["sample_id"])
    identities = samples["sample_id"].astype(str)
    denominator = set(identities)
    if (
        len(samples) != int(expected_count)
        or identities.str.strip().eq("").any()
        or identities.duplicated().any()
    ):
        raise PostprocessContractError("protocol sample denominator differs")

    inventory = _locked_source_inventory(lock)
    unified = _record_for_suffix(inventory, suffix=UNIFIED_DECISIONS_SUFFIX)
    sources: dict[str, dict[str, Any]] = {
        "G1": unified,
        "C1": unified,
    }
    if "D1" in route_names:
        sources["D1"] = _record_for_suffix(inventory, suffix=D1_DECISIONS_SUFFIX)
    rows = [
        _selector_rows(
            Path(str(sources[route]["path"])),
            route=route,
            system=SYSTEMS[route],
            denominator=denominator,
        )
        for route in route_names
    ]
    final = pd.concat(rows, ignore_index=True).sort_values(
        ["route", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    final_path = _publish_exact(
        root / FINAL_OUTCOMES_RELATIVE_PATH, final, resume=resume
    )
    authority_path = root / FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH
    if authority_path.exists() and not resume:
        raise FileExistsError(
            f"final outcomes authority exists; pass --resume: {authority_path}"
        )
    authority = write_final_outcomes_authority(
        root,
        protocol_lock=lock_path,
        final_outcomes=artifact_record(final_path),
        selector_sources=sources,
        routes=route_names,
    )
    return final_path, authority


__all__ = [
    "FINAL_OUTCOMES_RELATIVE_PATH",
    "materialize_frozen_final_outcomes",
]
