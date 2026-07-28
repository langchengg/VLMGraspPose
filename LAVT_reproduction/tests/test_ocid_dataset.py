from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.dataset_ocid_vlg_bert import OCIDVLGLAVTDataset, stable_sent_id
from scripts.audit_ocid_vlg import audit_manifests
from scripts.discover_data import build_manifest_rows


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        tokens = list(range(10, 10 + len(text.split())))
        return ([101] + tokens + [102]) if add_special_tokens else tokens


def make_fixture(tmp_path: Path):
    root = tmp_path / "OCID-VLG"
    sequence = root / "scene"
    (sequence / "rgb").mkdir(parents=True)
    (sequence / "seg_mask_instances_combi").mkdir()
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[2:5, 3:7] = (255, 64, 0)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 7
    Image.fromarray(rgb).save(sequence / "rgb" / "frame.png")
    Image.fromarray(mask).save(sequence / "seg_mask_instances_combi" / "frame.png")
    api = tmp_path / "api"
    api.mkdir()
    (api / "load_ocidvlg.py").write_text(
        """
from pathlib import Path
import numpy as np
from PIL import Image

class OCIDVLGDataset:
    def __init__(self, root_dir, split, transform_img=None, transform_grasp='bad',
                 with_depth=True, with_segm_mask=True, with_grasp_masks=True,
                 version='multiple'):
        assert transform_img is None
        assert transform_grasp is None
        assert with_depth is False
        assert with_segm_mask is True
        assert with_grasp_masks is False
        self.root_dir = Path(root_dir)
        self.scene_ids = ['scene,frame.png']
        self.sent_indices = [4]
        self.sentences = ['pick the orange block']
        self.objIDs = [7]
        self.rgb_paths = ['scene/rgb/frame.png']
        self.mask_paths = ['scene/seg_mask_instances_combi/frame.png']
    def __len__(self):
        return 1
    def __getitem__(self, index):
        raise AssertionError('adapter must bypass grasp-aware __getitem__')
    def get_image_from_path(self, path):
        return np.asarray(Image.open(path).convert('RGB'))
    def get_mask_from_path(self, path):
        return np.asarray(Image.open(path))
""",
        encoding="utf-8",
    )
    return root, api


def test_dataset_uses_local_official_api_without_grasp_preprocess(tmp_path):
    root, api = make_fixture(tmp_path)
    dataset = OCIDVLGLAVTDataset(
        root,
        api,
        split="train",
        image_size=(3, 4),
        max_tokens=8,
        tokenizer=FakeTokenizer(),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 3, 4)
    assert sample["target_model_resolution"].shape == (3, 4)
    assert sample["target_original_resolution"].shape == (6, 8)
    assert set(sample["target_model_resolution"].unique().tolist()) <= {0, 1}
    assert set(sample["target_original_resolution"].unique().tolist()) <= {0, 1}
    assert sample["input_ids"].shape == (1, 8)
    assert sample["attention_mask"].shape == (1, 8)
    assert sample["attention_mask"].sum().item() == 6
    assert sample["sent_id"] == stable_sent_id("scene,frame.png", 4)
    assert sample["original_size"].tolist() == [6, 8]
    assert sample["token_truncated"] is False


def test_manifest_filters_and_validates_official_records(tmp_path):
    root, api = make_fixture(tmp_path)
    sent_id = stable_sent_id("scene,frame.png", 4)
    row = {
        "dataset_index": 0,
        "sent_id": sent_id,
        "raw_question_index": 4,
        "scene_id": "scene,frame.png",
        "image_path": str((root / "scene/rgb/frame.png").resolve()),
        "mask_path": str((root / "scene/seg_mask_instances_combi/frame.png").resolve()),
        "sentence": "pick the orange block",
        "objID": 7,
        "split": "train",
        "dataset_version": "unique",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = OCIDVLGLAVTDataset(
        root, api, "train", manifest_path=manifest, tokenizer=FakeTokenizer()
    )
    assert len(dataset) == 1
    bad = dict(row, sentence="wrong")
    with pytest.raises(ValueError, match="manifest/source mismatch"):
        OCIDVLGLAVTDataset(
            root, api, "train", manifest_rows=[bad], tokenizer=FakeTokenizer()
        )


def test_token_audit_reports_truncation(tmp_path):
    root, api = make_fixture(tmp_path)
    dataset = OCIDVLGLAVTDataset(
        root, api, "train", max_tokens=4, tokenizer=FakeTokenizer()
    )
    audit = dataset.token_length_audit()
    assert audit["over_max_tokens_count"] == 1
    assert audit["truncated_sent_ids"] == [stable_sent_id("scene,frame.png", 4)]


def test_discover_builds_hifi_aligned_manifest_with_raw_id(tmp_path):
    root, _ = make_fixture(tmp_path)
    refer = root / "refer" / "unique"
    refer.mkdir(parents=True)
    source = {
        "image_filename": "scene,frame.png",
        "question_index": 4,
        "question": "pick the orange block",
        "answer": 7,
    }
    (refer / "train_expressions.json").write_text(
        json.dumps({"info": {"split": "train", "version": "unique"}, "data": [source]}),
        encoding="utf-8",
    )
    hifi = tmp_path / "ocid_vlg_train.json"
    hifi.write_text(
        json.dumps(
            [{"num": 0, "question_index": 4, "scene_id": "scene,frame.png",
              "text": "pick the orange block"}]
        ),
        encoding="utf-8",
    )
    rows = build_manifest_rows(root, "unique", "train", hifi)
    assert rows[0]["raw_question_index"] == 4
    assert rows[0]["sent_id"] == stable_sent_id("scene,frame.png", 4)
    assert rows[0]["objID"] == 7


def test_audit_reports_missing_files_without_skipping(tmp_path):
    manifests = {}
    for split, question_index in (("train", 1), ("val", 2), ("test", 3)):
        scene_id = f"{split},missing.png"
        row = {
            "dataset_index": 0,
            "sent_id": stable_sent_id(scene_id, question_index),
            "raw_question_index": question_index,
            "scene_id": scene_id,
            "image_path": str(tmp_path / split / "rgb.png"),
            "mask_path": str(tmp_path / split / "mask.png"),
            "sentence": f"find {split}",
            "objID": 1,
            "split": split,
            "dataset_version": "unique",
        }
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifests[split] = path
    report = audit_manifests(manifests, tokenizer=FakeTokenizer())
    assert report["status"] == "FAIL"
    assert report["integrity_issue_counts"]["missing_rgb"] == 3
    assert report["integrity_issue_counts"]["missing_mask"] == 3
    assert report["error_count"] == 6
