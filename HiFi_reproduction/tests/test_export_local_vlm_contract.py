from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grasping.reranking_v1.artifact_contract import identity_payload
from src.grasping.reranking_v1.identity import sha256_file
from src.grasping.reranking_v1.local_vlm import (
    SYSTEM_PROMPT,
    ranking_json_schema,
)
from tools.modular_reranking.export_local_vlm_contract import (
    export_contract,
)


def test_export_contract_is_exact_and_exclusive(tmp_path: Path) -> None:
    run = tmp_path / "run"
    temporary = run / "tmp"
    output = run / "local_vlm" / "contract"
    temporary.mkdir(parents=True)
    (run / ".RUN_ACTIVE").touch()

    manifest = export_contract(output_root=output, tmp_root=temporary)

    prompt = output / "system_prompt.txt"
    schema = output / "ranking_json_schema.json"
    assert prompt.read_text(encoding="utf-8") == SYSTEM_PROMPT.rstrip("\n") + "\n"
    assert json.loads(schema.read_text(encoding="utf-8")) == ranking_json_schema()
    assert manifest["system_prompt"]["sha256"] == sha256_file(prompt)
    assert manifest["ranking_json_schema"]["sha256"] == sha256_file(schema)
    assert {
        key: manifest[key] for key in identity_payload()
    } == identity_payload()
    assert manifest["ground_truth_allowed"] is False
    assert not any(temporary.iterdir())

    with pytest.raises(FileExistsError):
        export_contract(output_root=output, tmp_root=temporary)


def test_export_contract_rejects_persistent_output_below_tmp(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    temporary = run / "tmp"
    temporary.mkdir(parents=True)
    (run / ".RUN_ACTIVE").touch()
    with pytest.raises(ValueError, match="outside run tmp"):
        export_contract(
            output_root=temporary / "contract",
            tmp_root=temporary,
        )
