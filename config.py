"""
VLMGraspPose — Central Configuration (local Florence-2-base + RGB-D geometric + MLP stack)
====================================================================
All paths, hyperparameters, object mappings, and text templates
are defined here so every module shares a single source of truth.

Directory layout
----------------
project/
  data/
    raw/graspnet/scenes/  models/  grasp_label/  collision_label/
    splits/               ← Step 1 output (JSONL view indexes)
    metadata/             ← object_id_to_name.json, query_templates.json
  derived/                ← Steps 2–8 intermediate outputs
  models/                 ← downloaded / trained weights
  results/                ← Steps 10–11 predictions and metrics
"""

import json
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ═════════════════════════════════════════════════════════════════════
#  RAW DATA  (official GraspNet-1Billion layout)
# ═════════════════════════════════════════════════════════════════════
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "graspnet"

SCENES_DIR       = RAW_DATA_ROOT / "scenes"
OBJECT_MODELS_DIR = RAW_DATA_ROOT / "models"
GRASP_LABEL_DIR  = RAW_DATA_ROOT / "grasp_label"
COLLISION_LABEL_DIR = RAW_DATA_ROOT / "collision_label"
DEX_MODELS_DIR   = RAW_DATA_ROOT / "dex_models"        # optional

# ═════════════════════════════════════════════════════════════════════
#  METADATA
# ═════════════════════════════════════════════════════════════════════
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OBJECT_ID_TO_NAME_PATH = METADATA_DIR / "object_id_to_name.json"
QUERY_TEMPLATES_PATH   = METADATA_DIR / "query_templates.json"

# ═════════════════════════════════════════════════════════════════════
#  SPLITS  (Step 1 output)
# ═════════════════════════════════════════════════════════════════════
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# GraspNet official scene ranges
SPLIT_SCENE_RANGES = {
    "train":        (0, 90),       # scene_0000 – scene_0089
    "val":          (90, 100),     # scene_0090 – scene_0099  (held-out)
    "test_seen":    (100, 130),    # scene_0100 – scene_0129
    "test_similar": (130, 160),    # scene_0130 – scene_0159
    "test_novel":   (160, 190),    # scene_0160 – scene_0189
}

ALL_SPLITS = list(SPLIT_SCENE_RANGES.keys())
TRAIN_SPLITS = ["train"]
VAL_SPLITS   = ["val"]
TEST_SPLITS  = ["test_seen", "test_similar", "test_novel"]

# ═════════════════════════════════════════════════════════════════════
#  DERIVED OUTPUTS  (Steps 2–8)
# ═════════════════════════════════════════════════════════════════════
DERIVED_DIR = PROJECT_ROOT / "derived"

QUERIES_DIR         = DERIVED_DIR / "queries"          # Step 2
ORACLE_TARGETS_DIR  = DERIVED_DIR / "oracle_targets"   # Step 3
GROUNDING_PRED_DIR  = DERIVED_DIR / "grounding_pred"   # Step 4
POINTCLOUDS_DIR     = DERIVED_DIR / "pointclouds"      # Step 5
GRASP_CANDIDATES_DIR = DERIVED_DIR / "grasp_candidates" # Step 6
RANK_LABELS_DIR     = DERIVED_DIR / "rank_labels"      # Step 7
RANK_FEATURES_DIR   = DERIVED_DIR / "rank_features"    # Step 8

# ═════════════════════════════════════════════════════════════════════
#  MODEL WEIGHTS
# ═════════════════════════════════════════════════════════════════════
MODELS_DIR = PROJECT_ROOT / "models"

# Default local thesis stack
DEFAULT_GROUNDING = "seg"
DEFAULT_DETECTOR = "geometric"
DEFAULT_RERANKER = "mlp"

# Stage 1: Florence-2 base fine-tuned
FLORENCE2_MODEL_ID  = "microsoft/Florence-2-base-ft"
FLORENCE2_MODEL_DIR = MODELS_DIR / "florence2_base_ft"

# Stage 3: local RGB-D geometric sampler
TARGET_MIN_POINTS = 30
TABLE_PLANE_MAX_POINTS = 8000
TABLE_PLANE_DISTANCE_THRESH = 0.01

# Stage 5: trained MLP reranker
RERANKER_MLP_PATH    = MODELS_DIR / "reranker_mlp_geometric_predicted.pt"

# ═════════════════════════════════════════════════════════════════════
#  RESULTS  (Steps 10–11)
# ═════════════════════════════════════════════════════════════════════
RESULTS_DIR = PROJECT_ROOT / "results"

# ═════════════════════════════════════════════════════════════════════
#  CAMERA & VIEW SETTINGS
# ═════════════════════════════════════════════════════════════════════
CAMERA_TYPE  = "realsense"         # use one camera consistently
NUM_VIEWS    = 256                 # views per scene (0000–0255)
IMAGE_HEIGHT = 720
IMAGE_WIDTH  = 1280
VIEW_STRIDE  = 16                  # subsample: every Nth view

# ═════════════════════════════════════════════════════════════════════
#  GRASP GENERATION
# ═════════════════════════════════════════════════════════════════════
GRASP_TOP_K     = 50               # candidates per text-target sample
GRASP_MIN_WIDTH = 0.02             # metres
GRASP_MAX_WIDTH = 0.10
VOXEL_SIZE      = 0.005            # point-cloud down-sampling
NORMAL_RADIUS   = 0.02
NORMAL_MAX_NN   = 30
GEOMETRIC_MAX_POINTS_FOR_SAMPLING = 5000
GEOMETRIC_NUM_CENTER_SAMPLES = 300
GEOMETRIC_LOCAL_RADIUS = 0.035
GEOMETRIC_MIN_NEIGHBORS = 12

# ═════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (Step 8)
# ═════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "target_overlap",             # f1
    "center_alignment",           # f2
    "distance_to_target_center",  # f3
    "gripper_width_match",        # f4
    "approach_direction_score",   # f5
    "depth_stability",            # f6
    "collision_penalty",          # f7
    "boundary_penalty",           # f8
    "initial_geometric_score",    # f9
    "grounding_score",            # f10
]
FEATURE_DIM = len(FEATURE_NAMES)   # 10
FEATURE_MAX_SCENE_POINTS = 50000

# ═════════════════════════════════════════════════════════════════════
#  RERANKER TRAINING  (Step 9)
# ═════════════════════════════════════════════════════════════════════
# Rule-based weights
RULE_WEIGHTS = {
    "initial_geometric_score":  0.20,
    "target_overlap":          0.25,
    "center_alignment":        0.15,
    "gripper_width_match":     0.10,
    "depth_stability":         0.10,
    "approach_direction_score": 0.10,
    "collision_penalty":       0.07,
    "boundary_penalty":        0.03,
    "grounding_score":         0.05,
}

# MLP architecture
MLP_HIDDEN_DIMS = [64, 32]
MLP_LR          = 1e-3
MLP_EPOCHS      = 50
MLP_BATCH_SIZE  = 256

# Label thresholds
LABEL_COLLISION_THRESH = 0.5

# ═════════════════════════════════════════════════════════════════════
#  COORDINATE FRAME
# ═════════════════════════════════════════════════════════════════════
COORD_FRAME = "camera"             # all intermediates in camera frame

# ═════════════════════════════════════════════════════════════════════
#  GraspNet 88-object name map  (obj_id 0–87 → friendly name)
#  Full map lives in data/metadata/object_id_to_name.json
#  This is a convenience accessor loaded lazily.
# ═════════════════════════════════════════════════════════════════════
_OBJECT_NAME_CACHE = None

def get_object_name_map() -> dict:
    """Return {obj_id (int): friendly_name (str)} from the metadata JSON."""
    global _OBJECT_NAME_CACHE
    if _OBJECT_NAME_CACHE is None:
        if OBJECT_ID_TO_NAME_PATH.exists():
            with open(OBJECT_ID_TO_NAME_PATH) as f:
                raw = json.load(f)
            _OBJECT_NAME_CACHE = {int(k): v for k, v in raw.items()}
        else:
            # Fallback to inline minimal map
            _OBJECT_NAME_CACHE = {}
    return _OBJECT_NAME_CACHE


def get_query_templates() -> dict:
    """Return query templates from the metadata JSON."""
    if QUERY_TEMPLATES_PATH.exists():
        with open(QUERY_TEMPLATES_PATH) as f:
            return json.load(f)
    return {
        "class": [
            "pick the {obj}",
            "grasp the {obj}",
            "grab the {obj}",
            "get the {obj}",
        ]
    }


# ═════════════════════════════════════════════════════════════════════
#  Ensure core directories exist
# ═════════════════════════════════════════════════════════════════════
for _d in [
    SPLITS_DIR, METADATA_DIR,
    QUERIES_DIR, ORACLE_TARGETS_DIR, GROUNDING_PRED_DIR,
    POINTCLOUDS_DIR, GRASP_CANDIDATES_DIR,
    RANK_LABELS_DIR, RANK_FEATURES_DIR,
    MODELS_DIR, RESULTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)
