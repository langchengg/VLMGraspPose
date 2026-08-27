"""P0 source, environment, baseline, and candidate-contract audit."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .hashing import atomic_json, atomic_text, canonical_sha256, sha256_file
from .test_access_guard import append_access_log


EXPECTED_BASELINE = {
    "CROG-native": {"N": 7675, "J@1 numerator": 6848},
    "HiFi-CS→G1": {"N": 7675, "J@1 numerator": 3647},
    "HiFi-CS→C1": {"N": 7675, "J@1 numerator": 3363},
}


def _command(*args: str, cwd: Path | None = None) -> dict[str, Any]:
    process = subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "command": list(args),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _git_state(root: Path) -> dict[str, Any]:
    head = _command("git", "rev-parse", "HEAD", cwd=root)
    branch = _command("git", "branch", "--show-current", cwd=root)
    status = _command("git", "status", "--short", cwd=root)
    return {
        "commit": head["stdout"].strip(),
        "branch": branch["stdout"].strip() or "DETACHED",
        "dirty": bool(status["stdout"].strip()),
        "status_short": status["stdout"].splitlines(),
    }


def capture_environment(repo_root: Path, run_dir: Path) -> None:
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    )
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch": None,
        "mps_built": None,
        "mps_available": None,
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            }
        )
    except Exception as error:  # pragma: no cover - environment-specific
        environment["torch_error"] = f"{type(error).__name__}: {error}"
    atomic_json(run_dir / "00_audit" / "environment.json", environment)
    atomic_text(run_dir / "environment.txt", "\n".join(packages) + "\n")
    git = _git_state(repo_root)
    atomic_json(run_dir / "00_audit" / "git_state.json", git)
    atomic_text(
        run_dir / "git_state.txt",
        f"commit={git['commit']}\nbranch={git['branch']}\ndirty={git['dirty']}\n"
        + "\n".join(git["status_short"])
        + "\n",
    )
    system = {
        "hardware": _command("system_profiler", "SPHardwareDataType"),
        "software": _command("system_profiler", "SPSoftwareDataType"),
        "disk": _command("df", "-h", str(repo_root)),
        "processes": _command("ps", "-axo", "pid,etime,%cpu,%mem,command"),
    }
    atomic_json(run_dir / "00_audit" / "system_snapshot.json", system)


def _parquet_descriptor(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema_arrow.names,
    }


def audit_fair_source(run_dir: Path, fair_run: Path, modular_run: Path) -> dict[str, Any]:
    required = {
        "paired_manifest": fair_run / "01_manifest/paired_manifest.parquet",
        "crog_candidates": fair_run / "02_predictions/crog_native_predictions.parquet",
        "g1_candidates": fair_run / "02_predictions/g1_native_predictions.parquet",
        "c1_candidates": fair_run / "02_predictions/c1_native_predictions.parquet",
        "canonical_candidates_labels_locked": fair_run / "03_canonical/canonical_candidates.parquet",
        "main_results": fair_run / "04_metrics/main_results.csv",
        "evaluator": fair_run / "config/canonical_evaluator.py",
        "decoder": fair_run / "config/native_decoder_contract.json",
        "source_hashes": fair_run / "00_audit/source_run_hashes.json",
        "g1_selected": modular_run / "selected_configs/G1.json",
        "c1_selected": modular_run / "selected_configs/C1.json",
        "train_samples": modular_run / "manifests/train_samples.parquet",
        "validation_samples": modular_run / "manifests/validation_samples.parquet",
        "test_samples": modular_run / "manifests/test_samples.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required source artifacts: {missing}")

    descriptors = {
        name: (
            _parquet_descriptor(path)
            if path.suffix == ".parquet"
            else {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        for name, path in required.items()
    }
    append_access_log(
        run_dir,
        {
            "event": "prelock_audit",
            "allowed": ["parquet_schema_metadata", "aggregate_main_results"],
            "denied": ["candidate_level_test_labels", "per_sample_test_outcomes"],
        },
    )

    prediction_names = ("crog_candidates", "g1_candidates", "c1_candidates")
    candidate_contract: dict[str, Any] = {}
    for name in prediction_names:
        descriptor = descriptors[name]
        forbidden = {
            "candidate_success",
            "diagnostic_gt_index",
            "diagnostic_iou",
            "diagnostic_angle_error_deg",
        }.intersection(descriptor["columns"])
        if forbidden:
            raise ValueError(f"inference candidate table contains labels: {name}: {forbidden}")
        table = pq.read_table(required[name], columns=["sample_id", "candidate_id", "native_rank"])
        rows = table.to_pydict()
        sample_ids = [str(value) for value in rows["sample_id"]]
        keys = list(zip(sample_ids, map(str, rows["candidate_id"]), strict=True))
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate candidate identity in {name}")
        candidate_contract[name] = {
            "candidate_rows": len(keys),
            "paired_samples_with_candidates": len(set(sample_ids)),
            "maximum_native_rank": max(map(int, rows["native_rank"]), default=0),
            "identity_sequence_sha256": canonical_sha256(keys),
        }

    with required["main_results"].open(newline="", encoding="utf-8") as stream:
        aggregates = list(csv.DictReader(stream))
    by_method = {row["Method"]: row for row in aggregates}
    mismatches: list[str] = []
    for method, expected in EXPECTED_BASELINE.items():
        row = by_method.get(method)
        if row is None:
            mismatches.append(f"missing aggregate method {method}")
            continue
        for key, value in expected.items():
            if int(row[key]) != value:
                mismatches.append(
                    f"{method} {key}: observed={row[key]} expected={value}"
                )
    if descriptors["paired_manifest"]["rows"] != 7675:
        mismatches.append("paired manifest row count is not 7675")

    selected_hashes = {}
    for route in ("G1", "C1"):
        selected_path = required[f"{route.lower()}_selected"]
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        checkpoint = Path(selected["finetuned_checkpoint"]).resolve()
        observed = sha256_file(checkpoint)
        expected = str(selected["finetuned_checkpoint_sha256"])
        selected_hashes[route] = {
            "checkpoint": str(checkpoint),
            "observed_sha256": observed,
            "expected_sha256": expected,
            "match": observed == expected,
        }
        if observed != expected:
            mismatches.append(f"{route} checkpoint hash mismatch")

    audit = {
        "status": "PASS" if not mismatches else "FAIL",
        "fair_run": str(fair_run.resolve()),
        "modular_run": str(modular_run.resolve()),
        "artifacts": descriptors,
        "candidate_contract": candidate_contract,
        "aggregate_rows": aggregates,
        "checkpoint_hashes": selected_hashes,
        "mismatches": mismatches,
        "test_candidate_labels_accessed": False,
    }
    atomic_json(run_dir / "00_audit" / "source_run_hashes.json", descriptors)
    atomic_json(run_dir / "00_audit" / "fair_source_audit.json", audit)

    report = [
        "# Fair baseline audit",
        "",
        f"Status: **{audit['status']}**.",
        "",
        "The pre-lock audit read only inference-table schema/identity columns and the",
        "published aggregate baseline. Candidate-level test labels and per-sample test",
        "outcomes remain locked.",
        "",
        f"- Paired test rows: {descriptors['paired_manifest']['rows']:,}",
        f"- CROG candidate rows: {candidate_contract['crog_candidates']['candidate_rows']:,}",
        f"- G1 candidate rows: {candidate_contract['g1_candidates']['candidate_rows']:,}",
        f"- C1 candidate rows: {candidate_contract['c1_candidates']['candidate_rows']:,}",
        f"- Evaluator SHA-256: `{descriptors['evaluator']['sha256']}`",
        f"- Decoder SHA-256: `{descriptors['decoder']['sha256']}`",
        "",
        "This audit verifies provenance and published numerators; it does not claim a",
        "fresh candidate-label recomputation before the formal lock.",
    ]
    if mismatches:
        report.extend(["", "## Mismatches", "", *[f"- {item}" for item in mismatches]])
    atomic_text(run_dir / "00_audit" / "FAIR_BASELINE_AUDIT.md", "\n".join(report) + "\n")
    if mismatches:
        atomic_text(
            run_dir / "00_audit" / "BASELINE_MISMATCH_REPORT.md",
            "# Baseline mismatch report\n\n" + "\n".join(f"- {item}" for item in mismatches) + "\n",
        )
    _write_p0_reports(run_dir, fair_run, modular_run, audit)
    return audit


def _write_p0_reports(
    run_dir: Path,
    fair_run: Path,
    modular_run: Path,
    audit: dict[str, Any],
) -> None:
    """Materialise the complete P0 narrative from locked machine-readable evidence."""

    source_audit = fair_run / "00_audit" / "SOURCE_AND_ENVIRONMENT_AUDIT.md"
    checkpoint_registry = fair_run / "00_audit" / "checkpoint_registry.csv"
    geometry_contract = fair_run / "09_reports" / "GEOMETRY_CONTRACT.md"
    evaluator_validation = fair_run / "04_metrics" / "evaluator_validation_report.md"
    for path in (source_audit, checkpoint_registry, geometry_contract, evaluator_validation):
        if not path.is_file():
            raise FileNotFoundError(path)

    registry_text = checkpoint_registry.read_text(encoding="utf-8")
    atomic_text(run_dir / "00_audit" / "checkpoint_registry.csv", registry_text)

    packages = json.loads((run_dir / "00_audit" / "environment.json").read_text(encoding="utf-8"))
    git = json.loads((run_dir / "00_audit" / "git_state.json").read_text(encoding="utf-8"))
    source_report = [
        "# Source and environment audit",
        "",
        f"Status: **{audit['status']}**. The immutable baseline source is `{fair_run.resolve()}`.",
        "The development source is the modular Train/Validation/Test manifest run at",
        f"`{modular_run.resolve()}`. The current dirty worktree is recorded rather than cleaned.",
        "",
        "## Verified execution environment",
        "",
        f"- Python executable: `{packages['executable']}`",
        f"- Python: `{packages['python'].splitlines()[0]}`",
        f"- Platform: `{packages['platform']}` / `{packages['machine']}`",
        f"- PyTorch: `{packages.get('torch')}`; MPS built={packages.get('mps_built')}, available={packages.get('mps_available')}",
        f"- Git commit: `{git['commit']}`; branch `{git['branch']}`; dirty={git['dirty']}",
        "- Formal reranker computation is CPU-first because exact-host benchmarks were faster and",
        "  CPU avoids cross-backend reproducibility ambiguity. MPS remains available for the required",
        "  G1/C1 convolutional development inference after resource-safety checks.",
        "",
        "## Locked model and decoder sources",
        "",
        source_audit.read_text(encoding="utf-8").strip(),
        "",
        "## Capability and external-reference decision",
        "",
        "The installed grasp4dof virtual environment supplies PyTorch, pandas/PyArrow,",
        "scikit-learn, SciPy/statsmodels, OpenCV, Shapely, matplotlib, and LightGBM. No plugin,",
        "package installation, or external source-code copy is required. Official references used",
        "for implementation choices include [scikit-learn grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),",
        "[LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html), and",
        "[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).",
    ]
    atomic_text(
        run_dir / "00_audit" / "SOURCE_AND_ENVIRONMENT_AUDIT.md",
        "\n".join(source_report) + "\n",
    )

    historical: dict[str, Any] = {}
    for route in ("G1", "C1"):
        path = modular_run / "formal_test" / route / "metrics.json"
        historical[route] = json.loads(path.read_text(encoding="utf-8"))
    fair_rows = {row["Method"]: row for row in audit["aggregate_rows"]}
    decoder = json.loads((fair_run / "config" / "native_decoder_contract.json").read_text(encoding="utf-8"))
    history_report = [
        "# Historical reranking and baseline-gap audit",
        "",
        "Historical G1/C1 outcomes are provenance and attribution evidence only. They are not",
        "eligible primary candidates because their decoder/selector contract differs from the fair run.",
        "",
        "| Route | Historical J@1 | Fair native J@1 | Difference (pp) |",
        "|---|---:|---:|---:|",
    ]
    for route, fair_name in (("G1", "HiFi-CS→G1"), ("C1", "HiFi-CS→C1")):
        old = float(historical[route]["j_at_1"])
        new = float(fair_rows[fair_name]["J@1"])
        history_report.append(f"| {route} | {old:.6f} | {new:.6f} | {(old-new)*100:.3f} |")
    history_report.extend(
        [
            "",
            "## Contract attribution known before reranking",
            "",
            "- Historical modular inference multiplied/gated quality with mask probability, applied",
            "  candidate mask/jaw/centre rescoring, project NMS, and used a fixed 20 px short side.",
            "- Fair inference uses raw network q after the declared Gaussian filters, q>0.2,",
            "  min-distance 20, no mask-score gating, no rescoring, no post-peak NMS, and rectangle",
            "  height equal to half the predicted jaw width.",
            "- Consequently, the large historical-to-fair gap is already a candidate-membership,",
            "  geometry, score-composition, and selector-contract change. It cannot be attributed to",
            "  a reranker. The formal 2×2 bridge will quantify compatible cells after validation lock.",
            "",
            f"Fair decoder machine contract: `{json.dumps(decoder, sort_keys=True)}`",
        ]
    )
    atomic_text(
        run_dir / "00_audit" / "HISTORICAL_RERANKING_AUDIT.md",
        "\n".join(history_report) + "\n",
    )

    graph = f"""# Candidate provenance graph

```mermaid
flowchart TD
  A[\"OCID-VLG split manifests + frozen RGB/depth/language\"] --> B[\"CROG checkpoint {registry_text.splitlines()[1].split(',')[-1][:12]}…\"]
  A --> H[\"shared frozen HiFi-CS mask/probability\"]
  H --> G[\"G1 checkpoint 5fc8cdae2578…\"]
  H --> C[\"C1 checkpoint 13addaa29f1f…\"]
  B --> P1[\"native q-ranked K=5; no added NMS\"]
  G --> P2[\"fair Gaussian decoder; raw q; no gating/rescore/NMS\"]
  C --> P3[\"fair Gaussian decoder; raw q; no gating/rescore/NMS\"]
  P1 --> F[\"immutable per-route Frozen Top-5 membership and geometry\"]
  P2 --> F
  P3 --> F
  F --> R[\"order-only reranking; candidate_id permutation only\"]
  F --> E[\"frozen same-GT evaluator after formal lock\"]
```

- Test G1/C1 pools come directly from the locked fair run.
- Development G1/C1 pools must be freshly inferred with its exact checkpoint, transform, and decoder.
- Development CROG K=5 pools are reusable only after the historical-to-fair Test geometry/score
  identity cross-check passes exactly; this is performed by the candidate-freezing stage.
- Label-bearing canonical tables are physically separate and inaccessible to model feature loaders.
"""
    atomic_text(run_dir / "00_audit" / "CANDIDATE_PROVENANCE_GRAPH.md", graph)

    evaluator_report = [
        "# Evaluator audit",
        "",
        geometry_contract.read_text(encoding="utf-8").strip(),
        "",
        evaluator_validation.read_text(encoding="utf-8").strip(),
        "",
        "## Unified-reranking controls",
        "",
        f"- Frozen evaluator SHA-256: `{audit['artifacts']['evaluator']['sha256']}`.",
        "- Success is a same-GT conjunction: raster rotated IoU strictly >0.25 and",
        "  180-degree-periodic angle error <=30 degrees.",
        "- No-output samples remain in each paired denominator.",
        "- Candidate-label and per-sample Test outcome tables remain locked until the formal lock.",
        "- Shapely/OpenCV continuous polygon geometry will be validation oracles, not semantic",
        "  replacements for the frozen raster evaluator.",
    ]
    atomic_text(run_dir / "00_audit" / "EVALUATOR_AUDIT.md", "\n".join(evaluator_report) + "\n")
