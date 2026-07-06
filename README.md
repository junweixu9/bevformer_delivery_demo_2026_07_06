# BEVFormer Delivery Demo

YOLOv5s-style Python board demo for BEVFormer AidLite/QNN240 deployment.
The default demo is self-contained: it runs four continuous frames and writes
the full path from bundled raw camera JPEGs to detection arrays and real-camera grid visualization images on the board.

## What Is Included

- `python/run_test.py`: YOLOv5s-style board entry.
- `python/run_demo.py`: argument parsing and demo orchestration.
- `python/bevformer.py`: AidLite/QNN240 backbone, encoder, decoder pipeline.
- `python/portable_numpy_nmsfreecoder.py`: BEVFormer decoder postprocess.
- `python/camera_grid_visualization.py`: random10-style six-camera grid rendering.
- `python/visualize_sample4.py`: regenerates the fixed four W8A8-only camera-grid images from saved NPZ outputs.
- `datasets/sample4`: four continuous frames with bundled six-camera JPGs plus the required calibration/temporal auxiliary tensors.
- `docs/BOARD_ENVIRONMENT.md`: board runtime and Python dependency notes.

## Environment Dependencies

See `docs/BOARD_ENVIRONMENT.md` for the verified board environment and required Python/runtime dependencies. Real inference requires an AidLux board Python environment where `import aidlite` succeeds.

## Board Run

Run this on the board / Container B Python environment where `import aidlite` succeeds:

```bash
cd /home/aidlux/bevformer_delivery_demo
python3 -c "import aidlite; print('aidlite ok')"
python3 python/run_test.py
```

The default command runs four frames from `datasets/sample4`. For every frame it performs camera image preprocessing on the board before QNN inference:

```text
frame 000: scene-start encoder
frame 001: temporal encoder
frame 002: temporal encoder
frame 003: temporal encoder
```

YOLOv5s-style frame-count alias is also supported:

```bash
python3 python/run_test.py --invoke_nums 4 --output_dir outputs/smoke4
```

默认使用 6 个 worker 并行处理 6 路相机 JPG，这个值已经固定在 demo 内部，演示时不需要额外传入 worker 参数。

默认运行会检查图片文件是否存在，并从原始 JPG 做预处理；为减少板端计时开销，逐张 JPG 的 SHA 校验默认关闭。需要审计每张图片 SHA 时加：

```bash
python3 python/run_test.py --invoke_nums 4 --check_image_sha
```


## Timing Output

The default command prints fixed stage timings; no extra timing flags are needed.

- `mean_qnn_invoke_ms`: backbone + encoder + decoder QNN invoke only.
- `mean_image_preprocess_ms`: six-camera JPG decode/normalize/resize/pad.
- `mean_postprocess_ms`: NMS decode plus result saving.
- `mean_visualization_png_ms` and `camera_grid_gif_ms`: camera-grid PNG/GIF rendering.
- `complete_inference_no_visualization_ms`: full inference chain without visualization.
- `complete_inference_with_visualization_ms`: full inference chain with visualization.
- `program_total_ms_with_model_load_and_summary`: application wall time measured inside Python, including model loading and summary writing.

## Outputs

The command line prints image preprocessing timing, QNN pipeline timing, and top detection results. `frame_total_ms` includes board-side image preprocessing and QNN input/output/invoke time, and excludes NMS plus camera-grid visualization rendering. Files are
written under the selected output directory:

- `bevformer_demo_summary.json`: run summary, timing, SHA, frame routing.
- `frameXXX_final_coordinates.npz`: decoded boxes, scores, labels.
- `frameXXX_camera_grid.png`: six-camera grid with projected W8A8 3D boxes.
- `sample4_camera_grid.gif`: four-frame camera-grid animation.
- `frameXXX_cls_scores_fp32.raw` / `frameXXX_bbox_preds_fp32.raw`: saved for the final frame by default.

Visualization uses only the W8A8 board inference output and follows the original `visualize_random10.sh` camera-grid style. To regenerate the four images/GIF from saved NPZ files:

```bash
python3 python/visualize_sample4.py --results_dir outputs/smoke4
```

## Image Preprocessing

The board entry starts from original camera JPG files, not preprocessed image tensors.
For each frame, `python/utils.py` performs:

```text
cv2.imread BGR -> RGB -> normalize(mean/std) -> resize(800,450) -> CHW -> zero-pad to (6,3,480,800)
```

The preprocessing parameters match the original BEVFormer tiny image pipeline and
the manifest records the SHA of every bundled camera image.

## Model Path

The full inference chain loads four contexts:

- strict train64 W8A8 backbone
- scene-start encoder
- temporal encoder
- decoder

Scene-start frames use `scene_start_encoder_context.bin`; temporal frames use
`temporal_encoder_context.bin` with the previous live `bev_embed` as `prev_bev`.
This keeps BEVFormer temporal state explicit on the board and avoids cross-scene
state contamination.

## Data Policy

The final demo bundles only four continuous example frames in `datasets/sample4`. Each frame includes original six-camera JPG inputs and only the small calibration/temporal tensors required by the deployed contexts. It does not include any extra full-size datasets beyond this four-frame sample.

## Development Container Check

A normal development container usually does not have AidLite. Use dry-run only
for package inspection:

```bash
python3 python/run_test.py --dry_run
```

Real inference must run on the board. Camera-grid visualization can be regenerated on the board from saved NPZ outputs.
