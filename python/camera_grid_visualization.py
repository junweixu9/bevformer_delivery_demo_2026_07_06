from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import load_record, resolve_path, sha256_file


CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

MODEL_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

GRID_CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)



def load_font(size: int = 18):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def corners_3d(box: np.ndarray) -> np.ndarray:
    x, y, z, w, length, h, yaw = [float(v) for v in box[:7]]
    x_c = np.array([length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2])
    y_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_c = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (rot @ np.stack([x_c, y_c, z_c], axis=0)).T + np.array([x, y, z], dtype=np.float64)


def project_box(box: np.ndarray, lidar2img: np.ndarray, image_size: tuple[int, int]) -> np.ndarray | None:
    corners = corners_3d(box)
    homo = np.concatenate([corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    proj = (lidar2img.astype(np.float64) @ homo.T).T
    depth = proj[:, 2]
    if np.count_nonzero(depth > 1e-3) < 8:
        return None
    points = proj[:, :2] / depth[:, None]
    width, height = image_size
    if not (
        (points[:, 0] >= -100).any()
        and (points[:, 0] <= width + 100).any()
        and (points[:, 1] >= -100).any()
        and (points[:, 1] <= height + 100).any()
    ):
        return None
    return points


def draw_projected_box(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int = 3) -> None:
    xy = [(float(x), float(y)) for x, y in points]
    for i, j in EDGES:
        draw.line([xy[i], xy[j]], fill=color, width=width)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int], font) -> None:
    x = max(0.0, min(float(xy[0]), 790.0))
    y = max(0.0, min(float(xy[1]), 440.0))
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def select_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_thr: float,
    max_boxes: int,
) -> list[tuple[np.ndarray, float, str]]:
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    out: list[tuple[np.ndarray, float, str]] = []
    for idx in order:
        if float(scores[idx]) < float(score_thr):
            continue
        label = int(labels[idx])
        name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        out.append((boxes[idx], float(scores[idx]), name))
        if len(out) >= int(max_boxes):
            break
    return out


def camera_records(frame_assets: dict[str, Any], repo_root: str | Path) -> dict[str, Path]:
    record = frame_assets["assets"]["camera_images"]
    return {item["name"]: resolve_path(item["path"], repo_root) for item in record["images"]}


def load_lidar2img_by_camera(frame_assets: dict[str, Any], repo_root: str | Path) -> dict[str, np.ndarray]:
    value = load_record(frame_assets["assets"]["lidar2img"], repo_root).astype(np.float64).reshape(1, 6, 4, 4)[0]
    return {name: value[index] for index, name in enumerate(MODEL_CAMERA_ORDER)}


def wrap_text(text: str, width: int = 42) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_sidebar(sheet: Image.Image, frame_index: int, branch: str, score_thr: float, max_boxes: int, result_dir: str | Path | None) -> Image.Image:
    sidebar_w = 520
    canvas = Image.new("RGB", (sheet.width + sidebar_w, sheet.height), (250, 250, 250))
    canvas.paste(sheet, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    body_font = load_font(18)
    small_font = load_font(15)
    x0 = sheet.width + 24
    y = 24
    draw.rectangle((sheet.width, 0, canvas.width - 1, canvas.height - 1), fill=(245, 247, 250), outline=(180, 180, 180), width=2)
    draw.text((x0, y), "Final Detection Visualization", fill=(0, 0, 0), font=title_font)
    y += 46
    lines = [
        f"frame{frame_index:03d} / {branch}",
        "Final Model: W8A8 per_channel",
        "Role: AidLite/QNN board demo",
        "Input: six raw camera JPGs",
        "Preprocess: BGR->RGB normalize resize CHW pad",
        "Temporal: scene-start reset, then prev_bev recursion",
        f"Score threshold: {score_thr}",
        f"Max boxes: {max_boxes}",
        "Red: W8A8 final detections",
    ]
    for raw in lines:
        color = (0, 0, 0)
        if raw.startswith("Final"):
            color = (180, 0, 0)
        elif raw.startswith("Red"):
            color = (220, 20, 20)
        for line in wrap_text(raw, width=42):
            draw.text((x0, y), line, fill=color, font=body_font)
            y += 27
        y += 4
    if result_dir is not None:
        y += 14
        draw.text((x0, y), "Input result directory:", fill=(0, 0, 0), font=body_font)
        y += 28
        for line in wrap_text(str(result_dir), width=48):
            draw.text((x0, y), line, fill=(80, 80, 80), font=small_font)
            y += 22
    return canvas


def save_camera_grid_visualization(
    output_dir: str | Path,
    frame_index: int,
    frame_assets: dict[str, Any],
    repo_root: str | Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_thr: float = 0.25,
    max_boxes: int = 80,
    panel_width: int = 900,
    draw_labels: bool = False,
    metrics_panel: bool = True,
    result_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    label_font = load_font(15)
    cameras = camera_records(frame_assets, repo_root)
    lidar2img = load_lidar2img_by_camera(frame_assets, repo_root)
    preds = select_predictions(boxes, scores, labels, score_thr, max_boxes)

    panels = []
    per_camera = {}
    for camera_name in GRID_CAMERA_ORDER:
        image = Image.open(cameras[camera_name]).convert("RGB").resize((800, 450))
        draw = ImageDraw.Draw(image)
        count = 0
        for box, score, class_name in preds:
            points = project_box(box, lidar2img[camera_name], image.size)
            if points is None:
                continue
            draw_projected_box(draw, points, (255, 40, 40), width=3)
            if draw_labels:
                draw_label(draw, tuple(points[:, :2].min(axis=0)), f"W8A8 {class_name} {score:.2f}", (255, 180, 180), label_font)
            count += 1
        header = Image.new("RGB", (image.width, 44), (20, 20, 20))
        hdraw = ImageDraw.Draw(header)
        hdraw.text((10, 10), f"{camera_name} | W8A8 {count}", fill=(255, 255, 255), font=font)
        panel = Image.new("RGB", (image.width, image.height + header.height), (0, 0, 0))
        panel.paste(header, (0, 0))
        panel.paste(image, (0, header.height))
        panel = panel.resize((panel_width, int(panel.height * panel_width / panel.width)))
        panels.append(panel)
        per_camera[camera_name] = {"w8a8": int(count)}

    cols, rows = 3, 2
    panel_w, panel_h = panels[0].size
    title_h = 70
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h + title_h), (245, 245, 245))
    sheet_draw = ImageDraw.Draw(sheet)
    branch = "scene_start" if bool(frame_assets.get("is_scene_start", False)) else "temporal"
    token = str(frame_assets.get("sample_token", ""))[:16]
    title = f"Sample4 frame{frame_index:03d} ({branch}) token={token} | red=W8A8 final | score_thr={score_thr}"
    sheet_draw.text((16, 18), title, fill=(0, 0, 0), font=font)
    for index, panel in enumerate(panels):
        x = (index % cols) * panel_w
        y = title_h + (index // cols) * panel_h
        sheet.paste(panel, (x, y))

    if metrics_panel:
        sheet = add_sidebar(sheet, frame_index, branch, score_thr, max_boxes, result_dir)

    out_file = output_path / f"frame{frame_index:03d}_camera_grid.png"
    sheet.save(out_file)
    return {
        "path": str(out_file),
        "sha256": sha256_file(out_file),
        "mode": "camera_grid_w8a8_only",
        "score_thr": float(score_thr),
        "max_boxes": int(max_boxes),
        "candidate_count": int(len(preds)),
        "per_camera_projected_counts": per_camera,
    }


def save_camera_grid_gif(records: Sequence[dict[str, Any]], output_dir: str | Path, name: str = "sample4_camera_grid.gif", width: int = 1800, duration_ms: int = 700) -> dict[str, Any] | None:
    paths = [Path(record["path"]) for record in records if record.get("path")]
    if not paths:
        return None
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = int(image.height * width / image.width)
        frames.append(image.resize((width, height)))
    out_file = Path(output_dir) / name
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return {"path": str(out_file), "sha256": sha256_file(out_file), "frame_count": len(frames)}


def save_camera_grid_summary(output_dir: str | Path, records: Sequence[dict[str, Any]], gif_record: dict[str, Any] | None) -> dict[str, Any]:
    output_path = Path(output_dir)
    summary = {
        "status": "PASS",
        "mode": "sample4_camera_grid_w8a8_only",
        "visualizations": list(records),
        "gif": gif_record,
    }
    path = output_path / "sample4_camera_grid_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["path"] = str(path)
    summary["sha256"] = sha256_file(path)
    return summary
