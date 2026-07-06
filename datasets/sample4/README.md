# BEVFormer Sample4 Dataset

Self-contained four-frame continuous sample for the AidLite board demo.

- Each frame contains six original nuScenes camera JPG files under `cameras/`.
- `python/utils.py` preprocesses those JPG files on the board before backbone inference.
- Small raw tensors are still included for BEVFormer metadata: `can_bus`, `shift`, `lidar2img`, and temporal rotation inputs.
- frame 000 uses `encoder_scene_start` semantics.
- frames 001-003 use `encoder_temporal` with the previous live `bev_embed`.
- Paths in `asset_manifest.json` are relative to `/home/aidlux` when copied to the board.
