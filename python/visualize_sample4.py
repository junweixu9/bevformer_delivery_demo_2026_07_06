from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from camera_grid_visualization import (
    save_camera_grid_gif,
    save_camera_grid_summary,
    save_camera_grid_visualization,
)
from utils import load_json


PACKAGE_DIR = Path(__file__).resolve().parent
DEMO_ROOT = PACKAGE_DIR.parent
REPO_ROOT = DEMO_ROOT.parent
DEFAULT_RESULTS_DIR = DEMO_ROOT / "outputs/board_sample4"
DEFAULT_MANIFEST = DEMO_ROOT / "datasets/sample4/asset_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four W8A8-only real-camera grid visualizations from sample4 outputs."
    )
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--asset_manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_count", type=int, default=4)
    parser.add_argument("--score_thr", type=float, default=0.25)
    parser.add_argument("--max_boxes", type=int, default=80)
    parser.add_argument("--no_gif", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.asset_manifest)

    visualizations: list[dict[str, Any]] = []
    for frame_index in range(args.frame_start, args.frame_start + args.frame_count):
        sample = f"sample_{frame_index:03d}"
        npz_path = results_dir / f"frame{frame_index:03d}_final_coordinates.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Missing {npz_path}. Run `python3 python/run_test.py --invoke_nums 4` first."
            )
        data = np.load(npz_path)
        record = save_camera_grid_visualization(
            output_dir,
            frame_index,
            manifest["frames"][sample],
            REPO_ROOT,
            data["boxes"],
            data["scores"],
            data["labels"],
            score_thr=args.score_thr,
            max_boxes=args.max_boxes,
            result_dir=results_dir,
        )
        visualizations.append(record)
        print(f"Camera grid saved: {record['path']}")

    gif_record = None if args.no_gif else save_camera_grid_gif(visualizations, output_dir)
    summary = save_camera_grid_summary(output_dir, visualizations, gif_record)
    print("====================================")
    print(f"Generated {len(visualizations)} W8A8-only camera grid image(s).")
    if gif_record:
        print(f"gif: {gif_record['path']}")
    print(f"summary: {summary['path']}")
    print("====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
