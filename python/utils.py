from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

IMAGE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
IMAGE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)

SHAPES = {
    "images": (6, 3, 480, 800),
    "images_unpadded": (6, 3, 450, 800),
    "img_feat": (6, 256, 15, 25),
    "encoder_img_feat": (1, 6, 256, 15, 25),
    "can_bus": (1, 18),
    "shift": (1, 2),
    "lidar2img": (1, 6, 4, 4),
    "bev": (1, 2500, 256),
    "decoder": (1, 900, 10),
}

EXPECTED_TENSORS = {
    "backbone": {
        "inputs": {"images": 6912000},
        "outputs": {"img_feat": 576000},
    },
    "encoder_temporal": {
        "inputs": {
            "can_bus": 18,
            "img_feat": 576000,
            "lidar2img": 96,
            "shift": 2,
            "prev_bev": 640000,
        },
        "outputs": {"bev_embed": 640000},
    },
    "encoder_scene_start": {
        "inputs": {
            "can_bus": 18,
            "img_feat": 576000,
            "lidar2img": 96,
        },
        "outputs": {"bev_embed": 640000},
    },
    "decoder": {
        "inputs": {"bev_embed": 640000},
        "outputs": {"cls_scores": 9000, "bbox_preds": 9000},
    },
}

ROTATE_CENTER = (100.0, 100.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_rc(value: Any) -> int:
    return 0 if value is None else int(value)


def resolve_path(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(repo_root) / value


def load_record(record: dict[str, Any], repo_root: str | Path) -> np.ndarray:
    path = resolve_path(record["path"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    expected_sha = record.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"SHA mismatch for {path}: expected={expected_sha} actual={actual_sha}")

    dtype = "<f2" if "float16" in str(record["dtype"]).lower() else "<f4"
    shape = tuple(int(value) for value in record["shape"])
    array = np.fromfile(path, dtype=dtype)
    if array.size != math.prod(shape):
        raise RuntimeError(f"Element mismatch for {path}: expected={math.prod(shape)} actual={array.size}")
    return np.ascontiguousarray(array.reshape(shape))


def _load_camera_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for board-side camera image preprocessing.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel camera image: {path} shape={image.shape}")
    image = image.astype(np.float32)
    image = image[..., ::-1]
    image = (image - IMAGE_MEAN) / IMAGE_STD
    image = cv2.resize(image, (800, 450), interpolation=cv2.INTER_LINEAR)
    return image.transpose(2, 0, 1)


def _preprocess_one_camera(
    index: int,
    camera_name: str,
    item: dict[str, Any],
    repo_root: str | Path,
    check_sha: bool,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    path = resolve_path(item["path"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = None
    expected_sha = item.get("sha256")
    if check_sha:
        actual_sha = sha256_file(path)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"SHA mismatch for {path}: expected={expected_sha} actual={actual_sha}")
    tensor = _load_camera_image(path)
    if tensor.shape != SHAPES["images_unpadded"][1:]:
        raise ValueError(f"Unexpected preprocessed camera tensor shape: {tensor.shape}")
    return index, tensor, {
        "name": camera_name,
        "path": str(path),
        "sha_checked": bool(check_sha),
        "sha256": actual_sha or expected_sha,
    }


def preprocess_camera_images(
    record: dict[str, Any],
    repo_root: str | Path,
    check_sha: bool = False,
    num_workers: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    images_by_name = {item["name"]: item for item in record["images"]}
    camera_order = tuple(record.get("order", CAMERA_NAMES))
    if camera_order != CAMERA_NAMES:
        raise RuntimeError(f"Unexpected camera order: {camera_order}")

    tasks = [(index, name, images_by_name[name], repo_root, check_sha) for index, name in enumerate(CAMERA_NAMES)]
    if int(num_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
            results = list(executor.map(lambda args: _preprocess_one_camera(*args), tasks))
    else:
        results = [_preprocess_one_camera(*args) for args in tasks]

    padded = np.zeros(SHAPES["images"], dtype=np.float32)
    checked_images = [None] * len(CAMERA_NAMES)
    for index, tensor, image_record in results:
        padded[index, :, :450, :] = tensor
        checked_images[index] = image_record

    return np.ascontiguousarray(padded, dtype=np.float32), {
        "source": "camera_images",
        "pipeline": record.get(
            "pipeline",
            "BGR -> RGB -> normalize(mean/std) -> resize(800,450) -> CHW -> pad to 480 rows",
        ),
        "camera_order": list(CAMERA_NAMES),
        "sha_checked": bool(check_sha),
        "num_workers": int(num_workers),
        "unpadded_shape": list(SHAPES["images_unpadded"]),
        "padded_shape": list(padded.shape),
        "reference_preprocessed_sha256": record.get("reference_preprocessed_sha256"),
        "images": checked_images,
    }


def load_backbone_images(
    assets: dict[str, Any],
    repo_root: str | Path,
    check_image_sha: bool = False,
    preprocess_workers: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    if "camera_images" not in assets:
        raise KeyError("camera_images is required; this demo starts from raw camera JPG inputs.")
    return preprocess_camera_images(
        assets["camera_images"],
        repo_root,
        check_sha=check_image_sha,
        num_workers=preprocess_workers,
    )


def as_encoder_img_feat(img_feat: np.ndarray) -> np.ndarray:
    native = np.ascontiguousarray(img_feat, dtype="<f2")
    return np.ascontiguousarray(native.astype(np.float32).reshape(SHAPES["encoder_img_feat"]))


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1.0e6


def stats(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0}
    mean = float(sum(items) / len(items))
    variance = float(sum((value - mean) ** 2 for value in items) / len(items))
    return {
        "count": len(items),
        "mean": mean,
        "min": float(min(items)),
        "max": float(max(items)),
        "sum": float(sum(items)),
        "var": variance,
    }


def _torchvision_inverse_affine_matrix(center: tuple[float, float], angle: float) -> list[float]:
    rot = math.radians(angle)
    a = math.cos(rot)
    b = -math.sin(rot)
    c = math.sin(rot)
    d = math.cos(rot)
    matrix = [d, -b, 0.0, -c, a, 0.0]
    cx, cy = center
    matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    matrix[2] += cx
    matrix[5] += cy
    return matrix


def rotate_prev_bev_like_torchvision(
    previous_bev: np.ndarray,
    angle: float,
    rotate_center: tuple[float, float] = ROTATE_CENTER,
) -> np.ndarray:
    bev = np.ascontiguousarray(previous_bev, dtype=np.float32).reshape(SHAPES["bev"])
    grid = bev[0].reshape(50, 50, 256)
    height, width, channels = grid.shape
    center_f = (
        float(rotate_center[0]) - width * 0.5,
        float(rotate_center[1]) - height * 0.5,
    )
    theta = np.asarray(
        _torchvision_inverse_affine_matrix(center_f, -float(angle)),
        dtype=np.float32,
    ).reshape(2, 3)
    x_base = np.linspace(-width * 0.5 + 0.5, width * 0.5 - 0.5, width, dtype=np.float32)
    y_base = np.linspace(-height * 0.5 + 0.5, height * 0.5 - 0.5, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_base, y_base)
    src_x = (x_grid * theta[0, 0] + y_grid * theta[0, 1] + theta[0, 2]) / (0.5 * width)
    src_y = (x_grid * theta[1, 0] + y_grid * theta[1, 1] + theta[1, 2]) / (0.5 * height)
    ix = np.rint(((src_x + 1.0) * width - 1.0) * 0.5).astype(np.int64)
    iy = np.rint(((src_y + 1.0) * height - 1.0) * 0.5).astype(np.int64)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    out = np.zeros((height, width, channels), dtype=np.float32)
    out[valid] = grid[iy[valid], ix[valid]]
    return np.ascontiguousarray(out.reshape(SHAPES["bev"]), dtype=np.float32)


def rotate_prev_bev(previous_live_bev: np.ndarray, rotation_can_bus: np.ndarray) -> np.ndarray:
    angle = float(rotation_can_bus.reshape(-1)[-1])
    previous_native = np.ascontiguousarray(previous_live_bev, dtype="<f2")
    previous_semantic = previous_native.astype(np.float32).reshape(SHAPES["bev"])
    rotated = rotate_prev_bev_like_torchvision(previous_semantic, angle)
    rotated_native = np.ascontiguousarray(rotated, dtype="<f2")
    return np.ascontiguousarray(rotated_native.astype(np.float32).reshape(SHAPES["bev"]))


def save_raw_outputs(output_dir: Path, frame_index: int, cls_scores: np.ndarray, bbox_preds: np.ndarray) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cls_path = output_dir / f"frame{frame_index:03d}_cls_scores_fp32.raw"
    bbox_path = output_dir / f"frame{frame_index:03d}_bbox_preds_fp32.raw"
    np.ascontiguousarray(cls_scores, dtype="<f4").tofile(cls_path)
    np.ascontiguousarray(bbox_preds, dtype="<f4").tofile(bbox_path)
    return {
        "cls_scores": {"path": str(cls_path), "sha256": sha256_file(cls_path)},
        "bbox_preds": {"path": str(bbox_path), "sha256": sha256_file(bbox_path)},
    }


def save_final_coordinates(
    output_dir: Path,
    frame_index: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frame{frame_index:03d}_final_coordinates.npz"
    np.savez(
        path,
        boxes=np.ascontiguousarray(boxes, dtype=np.float32),
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        labels=np.ascontiguousarray(labels, dtype=np.int64),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "box_count": int(len(scores)),
    }
