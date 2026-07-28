import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.grasp_metrics import (
    evaluate_candidate,
    load_raw_binary_target_mask,
    periodic_angle_difference_deg,
)


MASK_THRESHOLD = 0.35
GRASP_IOU_THRESHOLD = 0.25


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def to_jsonable(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def mask_to_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return {"size": list(mask.shape), "start_value": 0, "counts": []}
    counts = []
    current = int(flat[0])
    run_length = 1
    for value in flat[1:]:
        value = int(value)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            current = value
            run_length = 1
    counts.append(run_length)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "start_value": int(flat[0]),
        "counts": counts,
    }


def rle_to_mask(rle):
    if not rle:
        return None
    counts = np.asarray(rle.get("counts", []), dtype=np.int64)
    if counts.size == 0:
        size = rle["size"]
        return np.zeros((int(size[0]), int(size[1])), dtype=bool)
    values = (np.arange(counts.size, dtype=np.uint8) + int(rle.get("start_value", 0))) % 2
    values = np.repeat(values, counts)
    size = rle["size"]
    return values.reshape((int(size[0]), int(size[1]))).astype(bool)


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def angle_difference_deg(a, b):
    if a is None or b is None:
        return math.nan
    return periodic_angle_difference_deg(a, b)


def nearest_gt_errors(pred_grasps, gt_grasps):
    """Compatibility view of canonical deterministic best-GT diagnostics.

    The historical name is retained for old report callers, but matching is no
    longer based on nearest center and must not be used as the success rule.
    """
    if not pred_grasps or len(gt_grasps) == 0:
        return math.nan, math.nan, math.nan
    best = evaluate_candidate(pred_grasps[0], gt_grasps)["best_gt"]
    return (
        float(best["center_distance_px"]),
        float(best["angle_difference_deg"]),
        float(best["width_difference_px"]),
    )


def point_inside_mask(point, mask):
    if point is None or mask is None:
        return False
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def load_rgb(path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_gt_mask(mask_path, obj_id):
    return load_raw_binary_target_mask(mask_path, obj_id)


def draw_grasps(image, grasps, color, thickness=2):
    output = image.copy()
    for rect in grasps or []:
        if len(rect) < 5:
            continue
        center_x, center_y, width, height, theta = [float(v) for v in rect[:5]]
        box = cv2.boxPoints(((center_x, center_y), (width, height), -(theta + 180.0)))
        box = np.asarray(box, dtype=np.intp)
        cv2.polylines(output, [box], True, color, thickness)
    return output


def overlay_mask(image, mask, color, alpha=0.45):
    output = image.astype(np.float32).copy()
    if mask is None:
        return image
    mask = np.asarray(mask, dtype=bool)
    output[mask] = output[mask] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def save_case_figure(row, output_path):
    rgb = load_rgb(row["image_path"])
    gt_mask = load_gt_mask(row["mask_path"], row["obj_id"])
    pred_mask = rle_to_mask(row.get("predicted_mask_rle"))
    gt_grasps = row.get("gt_grasps", [])
    top1 = row.get("predicted_grasps_top1", [])
    top5 = row.get("predicted_grasps_top5", [])

    gt_panel = overlay_mask(rgb, gt_mask, (255, 64, 64), alpha=0.42)
    gt_panel = draw_grasps(gt_panel, gt_grasps, (30, 90, 255), thickness=2)

    pred_panel = overlay_mask(rgb, pred_mask, (32, 220, 160), alpha=0.42)
    pred_panel = draw_grasps(pred_panel, top5[1:], (255, 210, 0), thickness=1)
    pred_panel = draw_grasps(pred_panel, top1, (0, 255, 0), thickness=3)

    combined = overlay_mask(gt_panel, pred_mask, (32, 220, 160), alpha=0.30)
    combined = draw_grasps(combined, top5[1:], (255, 210, 0), thickness=1)
    combined = draw_grasps(combined, top1, (0, 255, 0), thickness=3)

    metrics = [
        f"sample: {row.get('sample_id')}",
        f"target: {row.get('target_name', '')}",
        f"IoU: {format_metric(row.get('mask_iou'))}",
        f"J@1: {bool(row.get('j1_success'))}",
        f"J@Any: {bool(row.get('jany_success'))}",
        f"category: {row.get('failure_category_primary', '')}",
        str(row.get("short_reason", "")),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(gt_panel)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_panel)
    axes[2].set_title("Prediction")
    axes[3].imshow(combined)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    prompt = str(row.get("language_instruction", ""))
    fig.suptitle(prompt[:140], fontsize=13)
    fig.text(0.02, 0.02, "\n".join(metrics), fontsize=10, family="monospace")
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_contact_sheet(image_paths, output_path, columns=4, thumb_width=360):
    if not image_paths:
        return
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(image.shape[0] * scale))))
        images.append(thumb)
    if not images:
        return
    rows = math.ceil(len(images) / columns)
    thumb_height = max(image.shape[0] for image in images)
    canvas = np.full((rows * thumb_height, columns * thumb_width, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        y = row * thumb_height
        x = col * thumb_width
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    ensure_dir(Path(output_path).parent)
    plt.imsave(output_path, canvas)


def format_metric(value, digits=3):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"
