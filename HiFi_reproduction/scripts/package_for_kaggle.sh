#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN="$BASE_DIR/runs/hifics_ocidvlg_20260711_112921"
DEFAULT_PREFIX="$BASE_DIR/artifacts/hifi_anygrasp_inputs_hifics_ocidvlg_20260711_112921"
CONTRACT="$BASE_DIR/reports/HIFI_TO_ANYGRASP_DATA_CONTRACT.md"

DRY_RUN=false
FORCE=false
EXPECTED_SAMPLES="${HIFI_ANYGRASP_EXPECTED_SAMPLES:-7675}"
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --force) FORCE=true ;;
    --expected-samples)
      shift
      EXPECTED_SAMPLES="${1:?--expected-samples requires a positive integer}"
      ;;
    --help)
      echo "usage: package_for_kaggle.sh [--dry-run] [--force] [--expected-samples N] [RUN_DIR [OUTPUT_PREFIX [VERIFIED_SUBSET]]]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ ! "$EXPECTED_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected sample count must be a positive integer: $EXPECTED_SAMPLES" >&2
  exit 2
fi

RUN_DIR="${1:-$DEFAULT_RUN}"
OUTPUT_PREFIX="${2:-$DEFAULT_PREFIX}"
VERIFIED_SUBSET="${3:-}"

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

RUN_DIR="$(absolute_path "$RUN_DIR")"
OUTPUT_PREFIX="$(absolute_path "$OUTPUT_PREFIX")"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
if [[ -z "$VERIFIED_SUBSET" ]]; then
  VERIFIED_SUBSET="$RUN_DIR/anygrasp_verified_subset"
else
  VERIFIED_SUBSET="$(absolute_path "$VERIFIED_SUBSET")"
fi
if [[ ! -d "$VERIFIED_SUBSET" ]]; then
  echo "Verified subset not found: $VERIFIED_SUBSET" >&2
  exit 1
fi
VERIFIED_SUBSET="$(cd "$VERIFIED_SUBSET" && pwd -P)"
OUTPUT_PARENT="$(dirname "$OUTPUT_PREFIX")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd -P)"
OUTPUT_PREFIX="$OUTPUT_PARENT/$(basename "$OUTPUT_PREFIX")"

INPUT_ROOT="$RUN_DIR/anygrasp_input_predicted_mask"
MANIFEST_JSONL="$INPUT_ROOT/manifest.jsonl"
MANIFEST_CSV="$INPUT_ROOT/manifest.csv"
ARCHIVE="$OUTPUT_PREFIX.tar.gz"
ARCHIVE_SHA="$ARCHIVE.sha256"

for required in "$INPUT_ROOT" "$MANIFEST_JSONL" "$MANIFEST_CSV" "$CONTRACT"; do
  if [[ ! -e "$required" ]]; then
    echo "Required packaging input not found: $required" >&2
    exit 1
  fi
done
if [[ "$FORCE" != true && ( -e "$ARCHIVE" || -e "$ARCHIVE_SHA" ) ]]; then
  echo "Refusing to overwrite existing package; pass --force: $ARCHIVE" >&2
  exit 1
fi

export RUN_DIR INPUT_ROOT MANIFEST_JSONL MANIFEST_CSV VERIFIED_SUBSET CONTRACT
export OUTPUT_PREFIX ARCHIVE ARCHIVE_SHA DRY_RUN FORCE EXPECTED_SAMPLES
python3 - <<'PY'
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

run_dir = Path(os.environ['RUN_DIR'])
input_root = Path(os.environ['INPUT_ROOT'])
manifest_jsonl = Path(os.environ['MANIFEST_JSONL'])
manifest_csv = Path(os.environ['MANIFEST_CSV'])
verified_subset = Path(os.environ['VERIFIED_SUBSET'])
contract = Path(os.environ['CONTRACT'])
archive = Path(os.environ['ARCHIVE'])
archive_sha = Path(os.environ['ARCHIVE_SHA'])
dry_run = os.environ['DRY_RUN'] == 'true'
force = os.environ['FORCE'] == 'true'
expected_samples = int(os.environ['EXPECTED_SAMPLES'])

allowed_sample_files = (
    'color.png',
    'depth.png',
    'target_mask.png',
    'target_probability.npy',
    'language.txt',
    'intrinsics.json',
    'metadata.json',
)
safe_id = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def forbidden_oracle_content(value, key=''):
    key_lower = str(key).lower()
    if any(token in key_lower for token in ('ground_truth', 'groundtruth', 'oracle', 'gt_mask')):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {'', 'false', 'no', 'none', 'null', '0', 'predicted'}
        return bool(value)
    if isinstance(value, dict):
        return any(forbidden_oracle_content(item, child_key) for child_key, item in value.items())
    if isinstance(value, list):
        return any(forbidden_oracle_content(item) for item in value)
    if isinstance(value, str) and key_lower in {'target_source', 'mask_source', 'source'}:
        return value.lower() in {'oracle', 'ground_truth', 'groundtruth', 'gt'}
    return False

manifest_rows = []
for line_number, line in enumerate(manifest_jsonl.read_text().splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    manifest_rows.append(row)
if len(manifest_rows) != expected_samples:
    raise ValueError(
        f'Partial source manifest: expected {expected_samples} rows, '
        f'got {len(manifest_rows)}'
    )

rows = []
for line_number, row in enumerate(manifest_rows, 1):
    if row.get('ready') is True:
        if row.get('ready_for_anygrasp') is not True:
            raise ValueError(
                f'Row is ready but ready_for_anygrasp is not true on line '
                f'{line_number}'
            )
        sample_id = str(row.get('sample_id') or row.get('stable_sample_id') or '')
        if not safe_id.fullmatch(sample_id) or sample_id in {'.', '..'}:
            raise ValueError(f'Unsafe ready sample_id on line {line_number}: {sample_id!r}')
        if row.get('blockers') not in (None, []):
            raise ValueError(f'Ready sample still has blockers: {sample_id}')
        row = dict(row)
        row['sample_id'] = sample_id
        rows.append(row)
    elif row.get('ready_for_anygrasp') is True:
        raise ValueError(
            f'Row has ready_for_anygrasp true but ready is not true on line '
            f'{line_number}'
        )
if not rows:
    raise ValueError('No manifest rows are marked ready; refusing empty package')
sample_ids = [row['sample_id'] for row in rows]
if len(set(sample_ids)) != len(sample_ids):
    raise ValueError('Duplicate ready sample IDs in manifest.jsonl')

with manifest_csv.open(newline='') as handle:
    csv_rows = list(csv.DictReader(handle))
if len(csv_rows) != expected_samples:
    raise ValueError(
        f'Partial CSV manifest: expected {expected_samples} rows, got {len(csv_rows)}'
    )
csv_ready = {
    str(row.get('sample_id') or row.get('stable_sample_id') or '')
    for row in csv_rows
    if str(row.get('ready', '')).lower() in {'true', '1', 'yes'}
    and str(row.get('ready_for_anygrasp', '')).lower() in {'true', '1', 'yes'}
}
if csv_ready != set(sample_ids):
    raise ValueError('Ready rows disagree between manifest.jsonl and manifest.csv')

for sample_id in sample_ids:
    sample_dir = input_root / sample_id
    if not sample_dir.is_dir() or sample_dir.resolve().parent != input_root.resolve():
        raise ValueError(f'Ready bundle is missing or escapes input root: {sample_id}')
    names = {path.name for path in sample_dir.iterdir() if path.is_file()}
    missing = sorted(set(allowed_sample_files) - names)
    if missing:
        raise ValueError(f'Ready bundle {sample_id} is missing files: {missing}')
    forbidden_names = [
        name for name in names
        if any(token in name.lower() for token in ('ground_truth', 'groundtruth', 'oracle', 'gt_mask'))
    ]
    if forbidden_names:
        raise ValueError(f'Oracle/GT material found in main bundle {sample_id}: {forbidden_names}')
    metadata = json.loads((sample_dir / 'metadata.json').read_text())
    if forbidden_oracle_content(metadata):
        raise ValueError(f'Oracle/GT provenance found in main bundle metadata: {sample_id}')
    mask_source = str(metadata.get('mask_source', metadata.get('target_mask_source', 'predicted'))).lower()
    if mask_source != 'predicted_mask_original_resolution':
        raise ValueError(f'Main target mask is not explicitly predicted for {sample_id}: {mask_source}')
    checksums = sample_dir / 'checksums.sha256'
    if not checksums.is_file():
        raise ValueError(f'Ready source bundle lacks checksums.sha256: {sample_id}')
    covered_files = set()
    for checksum_line in checksums.read_text().splitlines():
        if not checksum_line.strip():
            continue
        parts = checksum_line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f'Invalid checksum line in {checksums}: {checksum_line}')
        expected, relative = parts
        relative = relative.lstrip('*')
        if relative in covered_files:
            raise ValueError(f'Duplicate checksum target in {sample_id}: {relative}')
        covered_files.add(relative)
        target = sample_dir / relative
        if not target.is_file() or target.resolve().parent != sample_dir.resolve():
            raise ValueError(f'Unsafe or missing checksum target in {sample_id}: {relative}')
        if sha256(target) != expected:
            raise ValueError(f'Checksum mismatch in source bundle {sample_id}: {relative}')
    missing_checksum_coverage = sorted(set(allowed_sample_files) - covered_files)
    if missing_checksum_coverage:
        raise ValueError(
            f'Ready source bundle checksum coverage missing for {sample_id}: '
            f'{missing_checksum_coverage}'
        )

subset_files = [path for path in verified_subset.rglob('*') if path.is_file()]
if not subset_files:
    raise ValueError(f'Verified subset is empty: {verified_subset}')
if any(path.is_symlink() for path in verified_subset.rglob('*')):
    raise ValueError('Verified subset contains symlinks; refusing unsafe package')

print(
    f'validated_manifest_rows={len(manifest_rows)} '
    f'expected_samples={expected_samples}'
)
print(f'validated_ready_samples={len(rows)}')
print(f'verified_subset_files={len(subset_files)}')
print(f'archive={archive}')
if dry_run:
    print('dry_run=PASS archive_not_created=true')
    sys.exit(0)

stage_parent = archive.parent
temp_dir = Path(tempfile.mkdtemp(prefix='.hifi-anygrasp-package-', dir=stage_parent))
try:
    package_name = archive.name.removesuffix('.tar.gz')
    package_root = temp_dir / package_name
    bundles_root = package_root / 'anygrasp_input_predicted_mask'
    manifests_root = package_root / 'manifests'
    checksums_root = package_root / 'checksums'
    docs_root = package_root / 'docs'
    subset_root = package_root / 'anygrasp_verified_subset'
    for path in (bundles_root, manifests_root, checksums_root, docs_root):
        path.mkdir(parents=True)

    for row in rows:
        sample_id = row['sample_id']
        source = input_root / sample_id
        target = bundles_root / sample_id
        target.mkdir()
        for name in allowed_sample_files:
            shutil.copy2(source / name, target / name)
        checksum_lines = [f'{sha256(target / name)}  {name}' for name in sorted(allowed_sample_files)]
        (target / 'checksums.sha256').write_text('\n'.join(checksum_lines) + '\n')

    ready_jsonl = manifests_root / 'manifest.ready.jsonl'
    ready_jsonl.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    fieldnames = sorted({key for row in rows for key in row})
    with (manifests_root / 'manifest.ready.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    package_manifest = {
        'schema_version': 1,
        'run_name': run_dir.name,
        'source_manifest_total_rows': len(manifest_rows),
        'expected_sample_count': expected_samples,
        'ready_sample_count': len(rows),
        'input_family': 'anygrasp_input_predicted_mask',
        'target_mask_source': 'hifi_predicted_mask_only',
        'ground_truth_in_main_export': False,
        'excluded': [
            '.venv', '.git', 'source repository', 'logs', 'predictions',
            'model-resolution probability/logit arrays', 'checkpoints', 'unrelated datasets',
        ],
        'verified_subset_location': 'anygrasp_verified_subset',
        'contract': 'docs/HIFI_TO_ANYGRASP_DATA_CONTRACT.md',
    }
    (manifests_root / 'package_manifest.json').write_text(json.dumps(package_manifest, indent=2) + '\n')
    shutil.copy2(contract, docs_root / contract.name)
    shutil.copytree(verified_subset, subset_root)

    content_files = sorted(path for path in package_root.rglob('*') if path.is_file())
    content_lines = [f'{sha256(path)}  {path.relative_to(package_root).as_posix()}' for path in content_files]
    (checksums_root / 'PACKAGE_CONTENTS.sha256').write_text('\n'.join(content_lines) + '\n')

    if any(path.is_symlink() for path in package_root.rglob('*')):
        raise ValueError('Staged package contains symlinks')
    temporary_archive = temp_dir / (archive.name + '.tmp')
    with temporary_archive.open('wb') as raw_handle:
        with gzip.GzipFile(
            filename='', mode='wb', fileobj=raw_handle, mtime=0, compresslevel=1
        ) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode='w', format=tarfile.PAX_FORMAT) as tar:
                paths = [package_root] + sorted(package_root.rglob('*'), key=lambda path: path.relative_to(temp_dir).as_posix())
                for path in paths:
                    if path.is_symlink():
                        raise ValueError(f'Refusing symlink in archive: {path}')
                    arcname = path.relative_to(temp_dir).as_posix()
                    member_path = PurePosixPath(arcname)
                    if member_path.is_absolute() or '..' in member_path.parts:
                        raise ValueError(f'Unsafe archive member: {arcname}')
                    info = tar.gettarinfo(str(path), arcname=arcname)
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = 0
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_file():
                        with path.open('rb') as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)

    with tarfile.open(temporary_archive, 'r:gz') as tar:
        for member in tar.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or '..' in member_path.parts or member.issym() or member.islnk():
                raise ValueError(f'Unsafe generated archive member: {member.name}')
    temporary_sha = temp_dir / (archive_sha.name + '.tmp')
    temporary_sha.write_text(f'{sha256(temporary_archive)}  {archive.name}\n')
    if force:
        archive.unlink(missing_ok=True)
        archive_sha.unlink(missing_ok=True)
    temporary_archive.replace(archive)
    temporary_sha.replace(archive_sha)
    print(f'package_status=DONE ready_samples={len(rows)}')
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
PY
