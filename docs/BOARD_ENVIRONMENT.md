# Board Environment Dependencies

This demo is intended to run on the AidLux board environment where the AidLite QNN240 runtime is available. A normal development container can run `--dry_run`, but cannot execute the QNN contexts unless `import aidlite` succeeds.

## Verified Board Environment

The current board used for validation reports:

```text
Python 3.10.12
aidlite import: OK
numpy 1.26.4
opencv-python / cv2 4.13.0
Pillow / PIL 10.4.0
```

## Required Runtime Components

- AidLux / AidLite Python runtime with `aidlite` module.
- AidLite QNN240 plugin/runtime for `FrameworkType.TYPE_QNN240`.
- Qualcomm HTP/DSP runtime and valid board license.
- Board-side QNN context execution support for QCS8550 / HTP v73.

The Python code checks the AidLite enum contract at startup:

```text
FrameworkType.TYPE_QNN240 = 109
ImplementType.TYPE_LOCAL = 3
AccelerateType.TYPE_DSP = 3
```

## Required Python Packages

The demo imports these Python packages at runtime:

```text
numpy
cv2
PIL
aidlite
```

Package usage:

- `aidlite`: loads and invokes the four QNN240 context `.bin` files.
- `numpy`: tensor loading, dtype conversion, BEVFormer postprocess, NPZ output.
- `cv2`: board-side six-camera JPG preprocessing.
- `PIL`: camera-grid PNG/GIF visualization.

## Model/Data Files Required By Default Run

Default command:

```bash
python3 python/run_test.py --invoke_nums 4 --output_dir outputs/final_sample4
```

Required model files:

```text
models/backbone_context.bin
models/scene_start_encoder_context.bin
models/temporal_encoder_context.bin
models/decoder_context.bin
```

Required config files:

```text
configs/demo_config.json
configs/nms_runtime_contract.json
```

Required sample data:

```text
datasets/sample4/asset_manifest.json
datasets/sample4/frames/sample_000..sample_003/
```

Each sample frame contains six raw camera JPGs plus the small auxiliary tensors required by the deployed contexts.

## Quick Environment Check

Run on the board:

```bash
cd /home/aidlux/bevformer_delivery_demo
python3 -c "import aidlite, numpy, cv2; from PIL import Image; print('board env ok')"
python3 python/run_test.py --dry_run --check_raw_assets
```

The dry-run checks paths, model SHA, sample assets, and scene-start/temporal routing. It does not invoke AidLite/DSP.

## Notes

- Do not expect real inference to work in a host/container environment without AidLite.
- `configs/qnn_htp_configs/` is not required by runtime; it was only an optional encryption/audit supplement and is not included in the runtime demo package.
- Full inference timing and visualization are printed by the default command without adding extra timing flags.
