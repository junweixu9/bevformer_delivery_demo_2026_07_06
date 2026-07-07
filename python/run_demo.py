from __future__ import annotations

"""BEVFormer board demo command-line entry.

This file is the orchestration layer of the demo. It does not implement the
model math itself. Its responsibilities are:

1. Parse command-line arguments.
2. Resolve model/config/data/output paths.
3. Optionally run a dry-run check without loading AidLite.
4. Load the four QNN240 context files through :class:`BevFormerModel`.
5. Run the four-frame sample4 manifest and write a summary JSON.

The real model execution lives in ``bevformer.py``. Image preprocessing and
small utility functions live in ``utils.py``.
"""

import argparse
import json
import time
from pathlib import Path

from bevformer import DEFAULT_SHA256, BevFormerModel
from utils import EXPECTED_TENSORS, sha256_file


# ``python/run_demo.py`` is placed under the demo package.
# PACKAGE_DIR: .../bevformer_delivery_demo/python
# DEMO_ROOT  : .../bevformer_delivery_demo
# REPO_ROOT  : parent directory used to resolve manifest paths that start with
#              ``bevformer_delivery_demo/...`` on the board.
PACKAGE_DIR = Path(__file__).resolve().parent
DEMO_ROOT = PACKAGE_DIR.parent
REPO_ROOT = DEMO_ROOT.parent

# Default QNN240 context files. These are the board runtime model binaries.
# Command-line arguments can override each path, but the default demo uses the
# top-level ``models/`` directory.
DEFAULT_BACKBONE = DEMO_ROOT / "models/backbone_context.bin"
DEFAULT_ENCODER_TEMPORAL = DEMO_ROOT / "models/temporal_encoder_context.bin"
DEFAULT_ENCODER_SCENE_START = DEMO_ROOT / "models/scene_start_encoder_context.bin"
DEFAULT_DECODER = DEMO_ROOT / "models/decoder_context.bin"

# Default config, sample manifest, postprocess contract, and output directory.
DEFAULT_CONFIG = DEMO_ROOT / "configs/demo_config.json"
DEFAULT_MANIFEST = DEMO_ROOT / "datasets/sample4/asset_manifest.json"
DEFAULT_NMS_CONTRACT = DEMO_ROOT / "configs/nms_runtime_contract.json"
DEFAULT_OUTPUT = DEMO_ROOT / "outputs/board_sample4"

# The demo always preprocesses the six camera JPGs in parallel by default.
DEFAULT_PREPROCESS_WORKERS = 6


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface used by ``python/run_test.py``.

    ``run_test.py`` is only a thin wrapper. All actual CLI options are defined
    here so the board command remains similar to a YOLOv5-style demo command.
    """

    parser = argparse.ArgumentParser(description="Run BEVFormer strict board demo with AidLite QNN240.")

    # Path arguments. If omitted, values are resolved from demo_config.json or
    # the hard-coded defaults above.
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone_model")
    parser.add_argument("--encoder_model")
    parser.add_argument("--scene_start_encoder_model")
    parser.add_argument("--decoder_model")
    parser.add_argument("--asset_manifest")
    parser.add_argument("--nms_contract")
    parser.add_argument("--output_dir")

    # Frame range. ``--invoke_nums`` is kept as the YOLOv5-style alias for
    # "how many samples to run".
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_count", type=int, default=4)
    parser.add_argument(
        "--invoke_nums",
        type=int,
        default=None,
        help="YOLOv5-style alias for how many consecutive frames to run.",
    )

    # Output and visualization switches.
    parser.add_argument("--save_all_raw", action="store_true")
    parser.add_argument("--no_visualize", action="store_true", help="Disable camera-grid visualization image output.")
    parser.add_argument("--vis_score_thr", type=float, default=0.0)
    parser.add_argument("--vis_max_boxes", type=int, default=80)

    # SHA checking for JPG files is optional because it adds extra file I/O.
    # Model context SHA checking is always performed.
    parser.add_argument(
        "--check_image_sha",
        action="store_true",
        help="Verify every camera JPG SHA during real inference. Slower; useful for audit runs.",
    )

    # Only QNN240 is supported by this delivery package.
    parser.add_argument("--model_type", default="QNN240")

    # Dry-run mode is for host/package checks. It avoids AidLite loading and
    # therefore can run on a normal development machine.
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect config, model SHA, manifest, and scene/temporal routing without loading AidLite.",
    )
    parser.add_argument(
        "--check_raw_assets",
        action="store_true",
        help="In dry-run mode, also check that every raw asset path referenced by the selected frames exists.",
    )
    return parser.parse_args()


def require_file(name: str, path: str) -> str:
    """Return an absolute file path after existence and non-empty checks."""

    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.stat().st_size == 0:
        raise FileNotFoundError(f"{name} missing or empty: {value}")
    return str(value)


def load_config(path: str) -> dict:
    """Load the main JSON config used to find default model/data paths."""

    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return json.loads(value.read_text(encoding="utf-8"))


def demo_path(config: dict, key_path: tuple[str, ...], fallback: Path) -> str:
    """Resolve one path from ``demo_config.json``.

    ``key_path`` is a nested JSON key path such as ``("models", "backbone")``.
    Relative paths in the config are interpreted relative to ``DEMO_ROOT``.
    If the key is missing, the function returns the provided fallback path.
    """

    current = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return str(fallback)
        current = current[key]

    path = Path(str(current))
    if not path.is_absolute():
        path = DEMO_ROOT / path
    return str(path)


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve manifest asset paths.

    The sample manifest stores board-style relative paths such as
    ``bevformer_delivery_demo/datasets/...``. These are resolved under
    ``REPO_ROOT`` so the same manifest works when the board package is located
    at ``/home/aidlux/bevformer_delivery_demo``.
    """

    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _dry_run(
    *,
    backbone_model: str,
    encoder_model: str,
    scene_start_encoder_model: str,
    decoder_model: str,
    asset_manifest: str,
    nms_contract: str,
    output_dir: Path,
    frame_start: int,
    frame_count: int,
    check_raw_assets: bool,
) -> dict:
    """Validate package integrity without invoking AidLite or DSP.

    Dry-run performs three checks:
    1. All four QNN240 context files exist and match the expected SHA256.
    2. The selected frame range exists in the sample manifest.
    3. When ``--check_raw_assets`` is enabled, every referenced raw asset exists.

    This mode is useful on Windows or a normal development container where the
    AidLite module is not available.
    """

    # Four model contexts are required by the split BEVFormer pipeline:
    # backbone, scene-start encoder, temporal encoder, and decoder.
    models = {
        "backbone": backbone_model,
        "encoder_temporal": encoder_model,
        "encoder_scene_start": scene_start_encoder_model,
        "decoder": decoder_model,
    }

    model_records = {}
    for name, model in models.items():
        model_path = Path(require_file(f"{name}_model", model))
        actual_sha = sha256_file(model_path)
        expected_sha = DEFAULT_SHA256[name]
        status = "PASS" if actual_sha == expected_sha else "FAIL"
        print(f"{name.upper()}_CONTEXT_SHA_GATE={status} {model_path.name}")
        if status != "PASS":
            raise RuntimeError(f"{name} context SHA mismatch: expected={expected_sha} actual={actual_sha}")

        # Store the expected tensor contract in the dry-run summary so the
        # package can be audited without loading AidLite.
        model_records[name] = {
            "path": str(model_path),
            "sha256": actual_sha,
            "expected_tensors": EXPECTED_TENSORS[name],
        }

    manifest_path = Path(require_file("asset_manifest", asset_manifest))
    nms_path = Path(require_file("nms_contract", nms_contract))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    total_frames = int(manifest.get("total_frames", len(manifest["frames"])))
    end = min(total_frames, frame_start + frame_count)
    if frame_start < 0 or frame_start >= total_frames or end < frame_start:
        raise ValueError(f"Invalid frame range: start={frame_start} count={frame_count} total={total_frames}")

    frames = {}
    scene_start_count = 0
    temporal_count = 0
    missing_assets = []

    for frame_index in range(frame_start, end):
        sample = f"sample_{frame_index:03d}"
        frame = manifest["frames"][sample]
        is_scene_start = bool(frame.get("is_scene_start", False))
        encoder_name = "encoder_scene_start" if is_scene_start else "encoder_temporal"

        # frame000 is a scene-start frame. Later frames use temporal encoder and
        # depend on the previous frame's live bev_embed.
        if is_scene_start:
            scene_start_count += 1
        else:
            temporal_count += 1

        # Raw asset checking is optional because it touches every camera JPG and
        # auxiliary tensor path. It is recommended before packaging or upload.
        if check_raw_assets:
            for asset_name, record in frame.get("assets", {}).items():
                if asset_name == "camera_images":
                    for image_record in record.get("images", []):
                        asset_path = _resolve_repo_path(image_record["path"])
                        if not asset_path.is_file():
                            missing_assets.append({
                                "frame": sample,
                                "asset": f"camera_images/{image_record.get('name', 'UNKNOWN')}",
                                "path": str(asset_path),
                            })
                    continue

                asset_path = _resolve_repo_path(record["path"])
                if not asset_path.is_file():
                    missing_assets.append({
                        "frame": sample,
                        "asset": asset_name,
                        "path": str(asset_path),
                    })

        frames[sample] = {
            "sample_token": frame.get("sample_token"),
            "is_scene_start": is_scene_start,
            "encoder": encoder_name,
            "status": "DRY_RUN_PASS",
        }
        print(f"FRAME {frame_index:03d} DRY_RUN encoder={encoder_name}")

    if missing_assets:
        first = missing_assets[0]
        raise FileNotFoundError(f"Missing raw asset: {first['frame']} {first['asset']} {first['path']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write a machine-readable summary. This is useful for proving that the
    # package is self-consistent before running on the board.
    result = {
        "status": "DRY_RUN_PASS",
        "note": "AidLite/DSP was not invoked. Run without --dry_run on the board for real inference.",
        "manifest": str(manifest_path),
        "nms_contract": str(nms_path),
        "repo_root": str(REPO_ROOT),
        "frame_range": [int(frame_start), int(end - 1)] if end > frame_start else [],
        "completed_frames": int(end - frame_start),
        "scene_start_encoder_count": scene_start_count,
        "temporal_encoder_count": temporal_count,
        "models": model_records,
        "raw_asset_existence_checked": bool(check_raw_assets),
        "frames": frames,
    }

    result_path = output_dir / "bevformer_demo_dry_run_summary.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("====================================")
    print("BEVFormer demo status: DRY_RUN_PASS")
    print(f"frames: {result['completed_frames']}")
    print(f"scene_start_encoder: {scene_start_count}")
    print(f"temporal_encoder: {temporal_count}")
    print(f"summary: {result_path}")
    print("AidLite/DSP not invoked in dry-run mode.")
    print("====================================")
    return result


def _fmt_ms(value) -> str:
    """Format optional millisecond values for console output."""

    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


def main() -> int:
    """Run the board demo.

    The function first resolves all paths. If ``--dry_run`` is enabled, it
    stops after package checks. Otherwise it loads AidLite contexts through
    ``BevFormerModel`` and runs the selected continuous frame span.
    """

    app_start = time.perf_counter_ns()

    # 1. Parse CLI and resolve all default paths from demo_config.json.
    args = parse_args()
    config = load_config(args.config)

    backbone_model = args.backbone_model or demo_path(config, ("models", "backbone"), DEFAULT_BACKBONE)
    encoder_model = args.encoder_model or demo_path(config, ("models", "encoder_temporal"), DEFAULT_ENCODER_TEMPORAL)
    scene_start_encoder_model = args.scene_start_encoder_model or demo_path(
        config,
        ("models", "encoder_scene_start"),
        DEFAULT_ENCODER_SCENE_START,
    )
    decoder_model = args.decoder_model or demo_path(config, ("models", "decoder"), DEFAULT_DECODER)
    asset_manifest = args.asset_manifest or demo_path(config, ("inputs", "asset_manifest"), DEFAULT_MANIFEST)
    nms_contract = args.nms_contract or demo_path(config, ("postprocess", "nms_contract"), DEFAULT_NMS_CONTRACT)
    output_dir = Path(args.output_dir or demo_path(config, ("outputs", "default_dir"), DEFAULT_OUTPUT)).expanduser().resolve()

    # ``--invoke_nums`` takes precedence over ``--frame_count``.
    frame_count = args.invoke_nums if args.invoke_nums is not None else args.frame_count

    # 2. Host-side package inspection path. No AidLite import or DSP execution.
    if args.dry_run:
        _dry_run(
            backbone_model=backbone_model,
            encoder_model=encoder_model,
            scene_start_encoder_model=scene_start_encoder_model,
            decoder_model=decoder_model,
            asset_manifest=asset_manifest,
            nms_contract=nms_contract,
            output_dir=output_dir,
            frame_start=args.frame_start,
            frame_count=frame_count,
            check_raw_assets=args.check_raw_assets,
        )
        return 0

    # 3. Board-side real inference path. This requires ``import aidlite`` to
    # succeed inside BevFormerModel.
    model_load_start = time.perf_counter_ns()
    model = BevFormerModel(
        backbone_model=require_file("backbone_model", backbone_model),
        encoder_temporal_model=require_file("encoder_model", encoder_model),
        encoder_scene_start_model=require_file("scene_start_encoder_model", scene_start_encoder_model),
        decoder_model=require_file("decoder_model", decoder_model),
        model_type=args.model_type,
    )
    model_load_wall_ms = (time.perf_counter_ns() - model_load_start) / 1.0e6

    # 4. Run sample4: raw JPG preprocessing -> backbone -> encoder -> decoder
    # -> NumPy NMSFreeCoder -> NPZ/PNG/GIF outputs.
    inference_start = time.perf_counter_ns()
    result = model.run_manifest(
        manifest_path=require_file("asset_manifest", asset_manifest),
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        nms_contract_path=require_file("nms_contract", nms_contract),
        frame_start=args.frame_start,
        frame_count=frame_count,
        save_all_raw=args.save_all_raw,
        visualize=not args.no_visualize,
        vis_score_thr=args.vis_score_thr,
        vis_max_boxes=args.vis_max_boxes,
        check_image_sha=args.check_image_sha,
        preprocess_workers=DEFAULT_PREPROCESS_WORKERS,
    )
    inference_wall_ms = (time.perf_counter_ns() - inference_start) / 1.0e6

    # 5. Add application-level timing that is not part of model-only QNN invoke.
    result["application_timing_ms"] = {
        "model_load_wall_ms": model_load_wall_ms,
        "run_manifest_wall_ms": inference_wall_ms,
        "total_until_summary_write_excluded_ms": (time.perf_counter_ns() - app_start) / 1.0e6,
    }

    result_path = output_dir / "bevformer_demo_summary.json"

    # Write once to measure summary serialization overhead, then write again
    # after adding the measured value to the summary.
    summary_write_start = time.perf_counter_ns()
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_write_ms = (time.perf_counter_ns() - summary_write_start) / 1.0e6
    result["application_timing_ms"]["summary_write_ms"] = summary_write_ms
    result["application_timing_ms"]["total_until_program_end_ms"] = (time.perf_counter_ns() - app_start) / 1.0e6
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Console summary for board demo presentation.
    qnn = result.get("qnn_invoke_ms", {})
    components = result.get("component_invoke_ms", {})
    e2e = result.get("end_to_end_timing_ms", {})
    app = result.get("application_timing_ms", {})

    print("====================================")
    print(f"QNN pipeline inference {result['completed_frames']} frames:")
    print(f" --mean_invoke_time is {_fmt_ms(qnn.get('mean'))} ms")
    print(f" --max_invoke_time is {_fmt_ms(qnn.get('max'))} ms")
    print(f" --min_invoke_time is {_fmt_ms(qnn.get('min'))} ms")
    print(f" --var_invoketime is {_fmt_ms(qnn.get('var'))}")
    print("====================================")
    print(f"BEVFormer demo status: {result['status']}")
    print(f"frames: {result['completed_frames']}")
    print(f"scene_start_encoder: {result['scene_start_encoder_count']}")
    print(f"temporal_encoder: {result['temporal_encoder_count']}")
    print(f"preprocess_workers: {DEFAULT_PREPROCESS_WORKERS}")
    print("====================================")
    print("Stage timing summary (ms):")
    print(f"model_load_total_ms: {_fmt_ms(app.get('model_load_wall_ms'))}")
    print(f"manifest_load_ms: {_fmt_ms(e2e.get('manifest_and_contract_load_ms'))}")
    print(f"mean_image_preprocess_ms: {_fmt_ms(result.get('image_preprocess_ms', {}).get('mean'))}")
    print(f"mean_qnn_invoke_ms: {_fmt_ms(qnn.get('mean'))}")
    print(f"mean_postprocess_ms: {_fmt_ms(result.get('postprocess_ms', {}).get('mean'))}")
    print(f"mean_visualization_png_ms: {_fmt_ms(result.get('visualization_ms', {}).get('mean'))}")
    print(f"camera_grid_gif_ms: {_fmt_ms(e2e.get('camera_grid_gif_ms'))}")
    print(f"mean_frame_inference_ms_no_postprocess_no_visualization: {_fmt_ms(result['timing_ms'].get('mean'))}")
    print(f"complete_inference_no_visualization_ms: {_fmt_ms(e2e.get('complete_inference_no_visualization_ms'))}")
    print(f"complete_inference_with_visualization_ms: {_fmt_ms(e2e.get('complete_inference_with_visualization_ms'))}")
    print(f"program_total_ms_with_model_load_and_summary: {_fmt_ms(app.get('total_until_program_end_ms'))}")
    print(f"mean_backbone_invoke_ms: {_fmt_ms(components.get('backbone', {}).get('mean'))}")
    print(f"mean_encoder_invoke_ms: {_fmt_ms(components.get('encoder', {}).get('mean'))}")
    print(f"mean_decoder_invoke_ms: {_fmt_ms(components.get('decoder', {}).get('mean'))}")
    print(f"summary: {result_path}")
    if result.get("visualizations"):
        print(f"visualizations: {len(result['visualizations'])} image(s)")
    print(f"outputs saved in {output_dir}")
    print("====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
