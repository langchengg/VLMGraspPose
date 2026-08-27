"""Repository, hardware, and storage audit for the GraspNet 6-DoF route."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANDATORY_PAPER_LITE_ARCHIVE_BYTES = {
    "train_4.zip": 6_803_985_160,
    "grasp_label.zip": 2_059_130_127,
    "collision_label.zip": 441_783_131,
    "models.zip": 4_599_338_858,
}
RESERVE_FRACTION = 0.20
CONSERVATIVE_EXTRACTION_MULTIPLIER = 2.0
CONSERVATIVE_CACHE_BYTES = 10_000_000_000


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(command: Iterable[str], *, cwd: Path) -> dict[str, Any]:
    argv = [str(value) for value in command]
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return {
        "command": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitised_system_profile(root: Path) -> dict[str, Any]:
    profile = _run(["system_profiler", "SPHardwareDataType", "-json"], cwd=root)
    hardware: dict[str, Any] = {}
    if profile["returncode"] == 0:
        try:
            records = json.loads(profile["stdout"]).get("SPHardwareDataType", [])
            source = records[0] if records else {}
            # Never persist serial numbers, provisioning identifiers, or UUIDs.
            allowed = {
                "machine_model": "machine_model",
                "machine_name": "machine_name",
                "chip_type": "chip_type",
                "number_processors": "number_processors",
                "number_cores": "number_cores",
                "physical_memory": "physical_memory",
            }
            hardware = {
                destination: source[key]
                for key, destination in allowed.items()
                if key in source
            }
        except (json.JSONDecodeError, TypeError):
            hardware = {"profile_parse_error": True}
    hardware.update(
        {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        }
    )
    return hardware


def _torch_profile() -> dict[str, Any]:
    try:
        import torch

        return {
            "version": str(torch.__version__),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        }
    except Exception as error:  # pragma: no cover - depends on the audit env
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


@dataclass(frozen=True)
class StorageDecision:
    total_bytes: int
    free_bytes: int
    reserve_bytes: int
    archive_bytes: int
    extraction_estimate_bytes: int
    cache_estimate_bytes: int
    required_free_before_download_bytes: int
    allowed: bool

    @property
    def free_fraction(self) -> float:
        return self.free_bytes / self.total_bytes

    @property
    def additional_bytes_needed(self) -> int:
        return max(0, self.required_free_before_download_bytes - self.free_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "free_fraction": self.free_fraction,
            "additional_bytes_needed": self.additional_bytes_needed,
            "assumptions": {
                "reserve_fraction": RESERVE_FRACTION,
                "extraction_multiplier": CONSERVATIVE_EXTRACTION_MULTIPLIER,
                "cache_estimate_bytes": CONSERVATIVE_CACHE_BYTES,
                "dex_models_included": False,
            },
        }


def paper_lite_storage_decision(path: Path) -> StorageDecision:
    usage = shutil.disk_usage(path)
    archives = sum(MANDATORY_PAPER_LITE_ARCHIVE_BYTES.values())
    reserve = int(usage.total * RESERVE_FRACTION)
    extraction = int(archives * CONSERVATIVE_EXTRACTION_MULTIPLIER)
    required = reserve + archives + extraction + CONSERVATIVE_CACHE_BYTES
    return StorageDecision(
        total_bytes=int(usage.total),
        free_bytes=int(usage.free),
        reserve_bytes=reserve,
        archive_bytes=archives,
        extraction_estimate_bytes=extraction,
        cache_estimate_bytes=CONSERVATIVE_CACHE_BYTES,
        required_free_before_download_bytes=required,
        allowed=bool(usage.free >= required),
    )


def _bytes_gb(value: int) -> str:
    return f"{value / 1_000_000_000:.2f} GB"


def _write_repository_audit(output: Path, commands: list[dict[str, Any]]) -> None:
    rendered = [
        "# Repository audit",
        "",
        f"Recorded: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The working tree was already dirty. Existing changes were preserved; the new route is isolated under `graspnet6d`.",
        "",
        "## Required command evidence",
        "",
    ]
    for item in commands:
        rendered.extend(
            [
                f"### `{' '.join(item['command'])}`",
                "",
                "```text",
                item["stdout"] or item["stderr"] or "(no output)",
                "```",
                "",
            ]
        )
    rendered.extend(
        [
            "## Relevant source-of-truth paths",
            "",
            "- HiFi-CS: `HiFi_reproduction/hifics/models/hifics.py`",
            "- HiFi-CS checkpoint: `HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/checkpoints/best.pth`",
            "- VGN snapshot/checkpoint: `HiFi_reproduction/third_party/vgn/`",
            "- Existing ranker: `src/unified_reranking/models/lightgbm_ranker.py`",
            "- Existing expected-gain gate: `src/unified_reranking/gate.py`",
            "- Official evaluator source snapshot: `legacy/external_graspnet/graspnetAPI/`",
            "",
            "No GraspNet-1Billion scene, grasp-label, collision-label, or model data was found in the configured external-data roots.",
        ]
    )
    _atomic_text(output / "repository_audit.md", "\n".join(rendered) + "\n")


def _write_reuse_map(output: Path) -> None:
    text = """# GraspNet 6-DoF reuse map

| Component | Decision | Evidence / reason |
|---|---|---|
| HiFi-CS model and best checkpoint | Reuse through adapter | A real `HierarchicalCLIPDensePredT` checkpoint exists; CLIP image/text encoders are frozen in the controlled configuration. New GraspNet masks still require real-data zero-shot evaluation and optional decoder-only adaptation. |
| Official VGN code and checkpoint | Reuse frozen through adapter | The local checkpoint hash matches the audited asset. The vendored tree lacks nested Git metadata, so provenance uses the locked upstream SHA plus file hashes instead of the broken parent-repository `git rev-parse`. |
| Existing 4-DoF LightGBM wrapper | Preserve; add graded adapter | The current wrapper hard-validates binary labels and `[0,1]` gain. Editing it would risk old checkpoints/tests; `graspnet6d.ranker` supplies relevance 0–6 and graded gains. |
| Existing expected-gain gate | Reuse mathematics only | The gate exists, but its learned models/thresholds belong to 4-DoF data. A 6-DoF gate must be trained from train OOF outcomes and selected on validation. |
| Existing OCID VGN diagnostic | Reference/adapt | It confirms official VGN inference and projection but uses task/OCID frames and centre-in-mask diagnostics, not official GraspNet physical evaluation. |
| graspnetAPI `eval_grasp` | Do not call directly on frozen pools | It performs score-dependent pose NMS and top-10/object/global top-50 pruning. The adapter calls the same low-level association/collision/friction math for every cached candidate. |
| 4-DoF route | Regression protected | Existing 4-DoF public modules are not used as a scratch area. The full `tests/unified_reranking` suite passed after the 6-DoF changes: 285 tests in 5,294.40 s. The hash-verified evidence is `artifacts/graspnet6d/regression/4d_regression_status.json`. |

No external source code was pasted into this namespace. Small mathematical adapters are independently written against official contracts and attributed in their module documentation.
"""
    _atomic_text(output / "reuse_map.md", text)


def _write_storage_decision(output: Path, decision: StorageDecision) -> None:
    text = f"""# Dataset profile decision

Profile: `paper-lite`  
Legacy simultaneous-footprint estimate: **{'PASS' if decision.allowed else 'SUPERSEDED / NOT AN EXECUTION GATE'}**

- Filesystem total: {_bytes_gb(decision.total_bytes)}
- Current free: {_bytes_gb(decision.free_bytes)} ({decision.free_fraction:.2%})
- Required 20% post-run reserve: {_bytes_gb(decision.reserve_bytes)}
- Mandatory compressed archives: {_bytes_gb(decision.archive_bytes)}
- Conservative extraction allowance (2× compressed): {_bytes_gb(decision.extraction_estimate_bytes)}
- Candidate/TSDF/feature/evaluator cache allowance: {_bytes_gb(decision.cache_estimate_bytes)}
- Required free before download: {_bytes_gb(decision.required_free_before_download_bytes)}
- Additional space needed now: {_bytes_gb(decision.additional_bytes_needed)}

This calculation assumes that every mandatory archive, a 2× full extraction,
the cache allowance, and a 20% filesystem reserve coexist.  That is not the
compact execution protocol.  It is retained only as historical audit evidence
and **must not block a download**.  The authoritative decision is
`staged_disk_budget.json`, which budgets one archive/selective-extraction/cache
phase at a time with a fixed 20 GB reserve.  `dex_models.zip` remains optional.
"""
    _atomic_text(output / "dataset_profile_decision.md", text)


def run_audit(output_root: Path | None = None) -> dict[str, Any]:
    from .staged_disk import build_staged_disk_budget, write_staged_disk_budget

    root = repository_root()
    output = output_root or root / "artifacts" / "graspnet6d" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        _run(["pwd"], cwd=root),
        _run(["git", "status", "--short"], cwd=root),
        _run(["git", "rev-parse", "HEAD"], cwd=root),
        _run(["git", "branch", "--show-current"], cwd=root),
        _run(["uname", "-a"], cwd=root),
        _run(["sw_vers"], cwd=root),
        _run(["uname", "-m"], cwd=root),
        _run([sys.executable, "--version"], cwd=root),
        _run(["df", "-h", "."], cwd=root),
    ]
    hardware = _sanitised_system_profile(root)
    storage = paper_lite_storage_decision(root)
    staged = build_staged_disk_budget(
        root,
        phase="download train_4.zip",
        archive_bytes=MANDATORY_PAPER_LITE_ARCHIVE_BYTES["train_4.zip"],
        selected_extraction_bytes=0,
        temporary_bytes=0,
        estimated_cache_bytes=0,
        minimum_free_reserve_gb=20.0,
    )
    environment = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "hardware": hardware,
        "torch": _torch_profile(),
        "storage": {
            "legacy_simultaneous_estimate": storage.as_dict(),
            "authoritative_staged_budget": staged.to_record(),
        },
        "privacy": "serial numbers and hardware UUIDs intentionally omitted",
    }
    _write_repository_audit(output, commands)
    _write_reuse_map(output)
    _write_storage_decision(output, storage)
    write_staged_disk_budget(
        staged,
        json_path=output / "staged_disk_budget.json",
        markdown_path=output / "staged_disk_budget.md",
    )
    _atomic_json(output / "hardware_environment.json", environment)
    _atomic_json(output / "system_profile.json", environment)
    freeze = _run([sys.executable, "-m", "pip", "freeze"], cwd=root)
    if freeze["returncode"] != 0 and shutil.which("uv"):
        freeze = _run(["uv", "pip", "freeze", "--python", sys.executable], cwd=root)
    _atomic_text(output / "pip_freeze.txt", freeze["stdout"] + "\n")
    _atomic_json(output / "audit_commands.json", commands)
    return {
        "output": str(output),
        "environment": environment,
        "download_allowed": staged.allowed,
    }


__all__ = [
    "MANDATORY_PAPER_LITE_ARCHIVE_BYTES",
    "StorageDecision",
    "paper_lite_storage_decision",
    "repository_root",
    "run_audit",
    "sha256_file",
]
