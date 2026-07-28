import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


SCHEMA_VERSION = "1.0.0"
FEATURES_FILENAME = "features.jsonl"
LABELS_FILENAME = "labels.jsonl"
PREDICTIONS_FILENAME = "predictions.jsonl"
METADATA_FILENAME = "metadata.json"
COMMIT_FILENAME = "commit_journal.jsonl"

# A ranker may read only these named inference-time feature groups.  Candidate
# ids, legacy rank, image ids, coordinates and all evaluation labels are absent.
INFERENCE_FEATURE_ALLOWLIST = (
    "q",
    "q_patch_mean",
    "q_prominence",
    "mask_consistency",
    "soft_coverage",
    "binary_coverage",
    "center_prob",
    "center_margin",
    "image_support",
    "width_compatibility",
    "width_compatibility_calibrated",
    "width_ratio",
    "width_symmetry",
    "angle_consistency",
    "depth_geometry",
    "depth_mad_m",
    "contact_depth_difference_m",
    "safety",
    "collision_proxy",
    "clearance",
)

FORBIDDEN_FEATURE_KEY_TOKENS = (
    "gt",
    "label",
    "success",
    "error",
    "iou",
    "validity",
    "failure_category",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_config(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def implementation_sha256(repo_root):
    """Hash behavior-defining source, including dirty and untracked files."""
    repo_root = Path(repo_root).resolve()
    paths = [
        repo_root / "failure_analysis" / "export_test_predictions.py",
        repo_root / "failure_analysis" / "failure_utils.py",
    ]
    paths.extend((repo_root / "model").rglob("*.py"))
    paths.extend((repo_root / "utils").rglob("*.py"))
    reranking_root = repo_root / "failure_analysis" / "reranking"
    paths.extend(reranking_root.rglob("*.py"))
    paths.extend(reranking_root.rglob("*.json"))
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def make_run_manifest(kind, repo_root, parameters):
    payload = {
        "kind": str(kind),
        "schema_version": SCHEMA_VERSION,
        "implementation_sha256": implementation_sha256(repo_root),
        "parameters": parameters,
    }
    return {**payload, "run_fingerprint": hash_config(payload)}


def validate_run_manifest(existing, expected):
    if not existing or existing.get("run_fingerprint") != expected.get("run_fingerprint"):
        raise ValueError(
            "run fingerprint mismatch; refusing to reuse output: "
            f"{None if not existing else existing.get('run_fingerprint')} != "
            f"{expected.get('run_fingerprint')}"
        )


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_metadata(
    *,
    repo_root,
    checkpoint_path,
    checkpoint_sha256,
    dataset_root,
    split,
    sample_count,
    feature_config,
    gripper_config,
    runtime,
    config_sha256=None,
    output_config=None,
    calibration=None,
    calibration_provenance=None,
):
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(repo_root),
        "implementation_sha256": implementation_sha256(repo_root),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": split,
        "sample_count": int(sample_count),
        "k": int(feature_config["candidate_generation"]["k"]),
        "peak_threshold": float(feature_config["candidate_generation"]["peak_threshold"]),
        "min_distance": int(feature_config["candidate_generation"]["min_distance"]),
        "mask_threshold": float(feature_config["mask_threshold"]),
        "feature_config_hash": hash_config(feature_config),
        "gripper_config_hash": hash_config(gripper_config),
        "config_sha256": config_sha256,
        "runtime_hash": hash_config(runtime),
        "output_config": output_config or {},
        "calibration_hash": hash_config(calibration) if calibration is not None else None,
        "calibration_provenance": calibration_provenance,
    }
    metadata = {
        **fingerprint_payload,
        "metadata_fingerprint": hash_config(fingerprint_payload),
        "sample_count": int(sample_count),
        "feature_config": feature_config,
        "gripper_config": gripper_config,
        "runtime": runtime,
        "creation_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return metadata


def prepare_output_dir(path, *, resume=False, overwrite=False):
    path = Path(path)
    managed = tuple(
        path / name
        for name in (
            FEATURES_FILENAME,
            LABELS_FILENAME,
            PREDICTIONS_FILENAME,
            METADATA_FILENAME,
            COMMIT_FILENAME,
        )
    )
    existing = [item for item in managed if item.exists()]
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if existing and not (resume or overwrite):
        names = ", ".join(str(item) for item in existing)
        raise FileExistsError(f"output exists; use --resume or explicit --overwrite: {names}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_mode(*, resume, overwrite):
    if resume:
        return "a"
    if overwrite:
        return "w"
    return "x"


def read_jsonl(path, limit=None):
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_record(handle, record):
    handle.write(canonical_json(record) + os.linesep)
    handle.flush()


def load_metadata(path):
    path = Path(path)
    if path.is_dir():
        path = path / METADATA_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def validate_resume_metadata(existing, expected):
    if existing.get("metadata_fingerprint") != expected.get("metadata_fingerprint"):
        raise ValueError(
            "metadata fingerprint mismatch; refusing to reuse cache: "
            f"{existing.get('metadata_fingerprint')} != {expected.get('metadata_fingerprint')}"
        )


def completed_sample_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {str(record["sample_id"]) for record in read_jsonl(path)}


def recover_committed_jsonl_prefix(output_dir, data_filenames):
    """Validate journal order and trim only uncommitted trailing records."""
    output_dir = Path(output_dir)
    journal_path = output_dir / COMMIT_FILENAME
    data_paths = [output_dir / name for name in data_filenames]
    if not journal_path.exists():
        if any(path.exists() and path.stat().st_size for path in data_paths):
            raise ValueError("resume refused: commit journal is missing")
        return []

    journal_bytes = journal_path.read_bytes()
    journal_lines = journal_bytes.splitlines(keepends=True)
    committed = []
    byte_offset = 0
    for index, raw_line in enumerate(journal_lines):
        if not raw_line.strip():
            byte_offset += len(raw_line)
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            is_unterminated_tail = (
                index == len(journal_lines) - 1 and not raw_line.endswith((b"\n", b"\r"))
            )
            if not is_unterminated_tail:
                raise ValueError("resume refused: commit journal is corrupt")
            with journal_path.open("r+b") as handle:
                handle.truncate(byte_offset)
            break
        committed.append(str(record["sample_id"]))
        byte_offset += len(raw_line)
    if len(committed) != len(set(committed)):
        raise ValueError("resume refused: duplicate sample_id in commit journal")
    expected_count = len(committed)
    for path in data_paths:
        if not path.exists():
            raise ValueError(f"resume refused: committed data file is missing: {path}")
        observed_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if observed_count < expected_count:
                    record = json.loads(line)
                    observed = str(record["sample_id"])
                    if observed != committed[observed_count]:
                        raise ValueError(
                            "resume refused: committed sample order mismatch in "
                            f"{path} at index {observed_count}: "
                            f"{observed} != {committed[observed_count]}"
                        )
                observed_count += 1
        if observed_count < expected_count:
            raise ValueError(
                f"resume refused: {path} has {observed_count} records but journal has {expected_count}"
            )
        if observed_count > expected_count:
            temporary = path.with_suffix(path.suffix + ".resume.tmp")
            written = 0
            with path.open("r", encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as target:
                for raw_line in source:
                    if not raw_line.strip():
                        continue
                    if written >= expected_count:
                        break
                    target.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                    written += 1
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
    return committed


def assert_no_forbidden_feature_keys(record):
    def visit(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in FORBIDDEN_FEATURE_KEY_TOKENS):
                    raise ValueError(f"forbidden inference feature key at {path or '<root>'}: {key}")
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(record)


def inference_vector(candidate, fields=INFERENCE_FEATURE_ALLOWLIST):
    """Return values/reliabilities/missing flags using only the allowlist."""
    features = candidate.get("features", {})
    values = []
    for name in fields:
        if name not in INFERENCE_FEATURE_ALLOWLIST:
            raise ValueError(f"feature is not allowlisted: {name}")
        feature = features.get(name, {})
        raw = feature.get("value")
        reliability = float(feature.get("reliability", 0.0) or 0.0)
        missing = raw is None or reliability <= 0.0
        values.extend([None if raw is None else float(raw), reliability, float(missing)])
    return values
