"""
VLMGraspPose — Central Configuration
=====================================
All paths, hyperparameters, object mappings, and text templates
are defined here so every module shares a single source of truth.
"""

import os
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data Paths ───────────────────────────────────────────────────────
DATA_DIRS = {
    "test_seen": PROJECT_ROOT / "test_seen",
    # Future:
    # "train": PROJECT_ROOT / "train",
    # "test_similar": PROJECT_ROOT / "test_similar",
    # "test_novel": PROJECT_ROOT / "test_novel",
}

PROCESSED_DIR = PROJECT_ROOT / "processed"
STAGE1_OUTPUT_DIR = PROJECT_ROOT / "stage1_outputs"
STAGE2_OUTPUT_DIR = PROJECT_ROOT / "stage2_outputs"
FEATURES_DIR = PROJECT_ROOT / "features"
RANKING_DATA_DIR = PROJECT_ROOT / "ranking_data"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Pre-trained Model Paths ──────────────────────────────────────────
# Stage 1: Target Grounding
FLORENCE2_MODEL_DIR = MODELS_DIR / "florence-2-base"      # HuggingFace
GDINO_MODEL_DIR = MODELS_DIR / "grounding-dino-base"      # HuggingFace
# Stage 2: Grasp Generation
GRASPNET_CHECKPOINT_DIR = MODELS_DIR / "graspnet-baseline"

# Create output directories
for d in [PROCESSED_DIR, STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR,
          FEATURES_DIR, RANKING_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_TYPE = "kinect"          # or "realsense"
NUM_VIEWS = 256                 # views per scene
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# ── GraspNet Object Name → Friendly Name ────────────────────────────
# Maps the raw .ply names (without extension) to human-readable names
# used in text queries.  Keep this in sync with the actual dataset.
OBJECT_NAME_MAP = {
    "003_cracker_box":        "cracker box",
    "005_tomato_soup_can":    "tomato soup can",
    "011_banana":             "banana",
    "012_strawberry":         "strawberry",
    "015_peach":              "peach",
    "018_plum":               "plum",
    "025_mug":                "mug",
    "032_knife":              "knife",
    "035_power_drill":        "power drill",
    "037_scissors":           "scissors",
    "044_flat_screwdriver":   "flat screwdriver",
    "057_racquetball":        "racquetball",
    "065-b_cups":             "cups",
    "072-d_toy_airplane":     "toy airplane",
    "072-f_toy_airplane":     "toy airplane",
    "072-i_toy_airplane":     "toy airplane",
    "072-j_toy_airplane":     "toy airplane",
    "dabao_sod":              "dabao sod cream",
    "darlie_toothpaste":      "toothpaste",
    "camel":                  "camel figurine",
    "large_elephant":         "elephant figurine",
    "rhinocero":              "rhinoceros figurine",
    "darlie_box":             "toothpaste box",
    "black_mouse":            "mouse",
    "dabao_facewash":         "face wash",
    "pantene":                "pantene bottle",
    "head_shoulders_supreme": "shampoo bottle",
    "head_shoulders_care":    "shampoo bottle",
}

# ── Text Query Templates ────────────────────────────────────────────
TEXT_TEMPLATES = [
    "pick the {obj}",
    "grasp the {obj}",
    "grab the {obj}",
    "get the {obj}",
]

# ── Stage 2: Grasp Generation ────────────────────────────────────────
GRASP_TOP_K = 50               # candidates per target
GRASP_MIN_WIDTH = 0.02         # metres
GRASP_MAX_WIDTH = 0.10
VOXEL_SIZE = 0.005             # for point-cloud down-sampling
NORMAL_RADIUS = 0.02           # for surface-normal estimation
NORMAL_MAX_NN = 30

# ── Stage 3: Feature Extraction ─────────────────────────────────────
FEATURE_DIM_CORE = 5           # f1 – f5
FEATURE_DIM_EXTENDED = 9       # f1 – f9

# ── Stage 4: Scoring ────────────────────────────────────────────────
# Rule-based weights  (α, β, γ, δ, ε)
RULE_WEIGHTS = {
    "f1_grasp_score": 0.30,
    "f2_in_target":   0.20,
    "f3_distance":    0.20,    # applied as (1 − f3)
    "f4_iou":         0.20,
    "f5_vlm_conf":    0.10,
}

# Label generation thresholds
LABEL_GRASP_SCORE_THRESH = 0.3   # candidate score ≥ this → potential positive
LABEL_COLLISION_THRESH = 0.5     # collision risk < this → potential positive

# ── Training ─────────────────────────────────────────────────────────
MLP_HIDDEN_DIMS = [64, 32]
MLP_LR = 1e-3
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 256

# ── Preprocess ───────────────────────────────────────────────────────
# Only use every N-th view to keep JSONL manageable during demo
VIEW_STRIDE = 16               # use views 0, 16, 32, … (16 views per scene)

# ── Coordinate Frame Convention ──────────────────────────────────────
# All intermediate products are stored in CAMERA FRAME unless noted.
# depth + K  →  point cloud in camera frame
# grasp pose →  camera frame
# projection →  image frame (for f2, f4)
COORD_FRAME = "camera"
