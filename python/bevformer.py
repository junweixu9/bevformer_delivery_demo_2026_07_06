from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import portable_numpy_nmsfreecoder as portable_nms
from camera_grid_visualization import save_camera_grid_gif, save_camera_grid_summary, save_camera_grid_visualization
from utils import (
    EXPECTED_TENSORS,
    SHAPES,
    as_encoder_img_feat,
    elapsed_ms,
    load_backbone_images,
    load_json,
    load_record,
    normalize_rc,
    rotate_prev_bev,
    save_final_coordinates,
    save_raw_outputs,
    sha256_file,
    stats,
)


DEFAULT_SHA256 = {
    "backbone": "a1988955080440ba95892b7b484f3677948508fcdae94a3e4aca34ff5b055a76",
    "encoder_temporal": "c0950b33c725a8c899520a35e8d11826204766b11beae7693516c0b1648772fb",
    "encoder_scene_start": "cc5e9c75e517ba413279d8277f14e7f2c11752d7f06a59429bf1bde193802bfe",
    "decoder": "dacebf6428168bbe0e29410f05285a7fde454860a607c73f983149e607b78d7c",
}

NUSCENES_CLASSES = (
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


class BevFormerModel:
    def __init__(
        self,
        backbone_model: str,
        encoder_temporal_model: str,
        encoder_scene_start_model: str,
        decoder_model: str,
        model_type: str = "QNN240",
        expected_sha256: dict[str, str] | None = None,
    ):
        try:
            import aidlite
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AidLite Python runtime is not available in this environment. "
                "Use --dry_run to demonstrate the package structure in a normal container, "
                "or run without --dry_run on the board / Container B where "
                "`python3 -c \"import aidlite\"` succeeds."
            ) from exc

        if model_type.upper() != "QNN240":
            raise ValueError("This demo is pinned to QNN240 contexts")
        if (
            int(aidlite.FrameworkType.TYPE_QNN240),
            int(aidlite.ImplementType.TYPE_LOCAL),
            int(aidlite.AccelerateType.TYPE_DSP),
        ) != (109, 3, 3):
            raise RuntimeError("AidLite enum contract mismatch")

        self.aidlite = aidlite
        self.expected_sha256 = expected_sha256 or DEFAULT_SHA256
        self.interpreters: dict[str, Any] = {}
        self.model_records: dict[str, Any] = {}
        self.model_load_timing_ms: dict[str, float] = {}
        model_load_start = time.perf_counter_ns()

        for name, path in (
            ("backbone", backbone_model),
            ("encoder_temporal", encoder_temporal_model),
            ("encoder_scene_start", encoder_scene_start_model),
            ("decoder", decoder_model),
        ):
            load_start = time.perf_counter_ns()
            interpreter, record = self._create_loaded_interpreter(name, str(path))
            self.model_load_timing_ms[name] = elapsed_ms(load_start)
            self.interpreters[name] = interpreter
            self.model_records[name] = record

        self.model_load_timing_ms["total"] = elapsed_ms(model_load_start)

    def __del__(self):
        for interpreter in reversed(list(getattr(self, "interpreters", {}).values())):
            for method_name in ("destroy", "destory"):
                if hasattr(interpreter, method_name):
                    try:
                        getattr(interpreter, method_name)()
                    except Exception:
                        pass
                    break

    def _create_model(self, model_path: str) -> Any:
        try:
            return self.aidlite.Model.create_instance(model_path=model_path)
        except TypeError:
            return self.aidlite.Model.create_instance(model_path)

    def _build_interpreter(self, model: Any, config: Any) -> Any:
        for method_name in ("build_interpreter_from_model_and_config", "build_interpretper_from_model_and_config"):
            if hasattr(self.aidlite.InterpreterBuilder, method_name):
                method = getattr(self.aidlite.InterpreterBuilder, method_name)
                try:
                    return method(model=model, config=config)
                except TypeError:
                    return method(model, config)
        raise RuntimeError("No supported AidLite InterpreterBuilder method")

    @staticmethod
    def _flatten_tensor_info(groups: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if groups is None:
            return records
        for graph_index, group in enumerate(groups):
            try:
                tensors = list(group)
            except TypeError:
                tensors = [group]
            for tensor_index, info in enumerate(tensors):
                records.append({
                    "graph_index": graph_index,
                    "tensor_index": tensor_index,
                    "name": str(getattr(info, "name", "")),
                    "element_count": int(getattr(info, "element_count", -1)),
                    "shape": [int(v) for v in getattr(info, "shape", [])],
                    "element_type": str(getattr(info, "element_type", "")),
                })
        return records

    def _create_loaded_interpreter(self, name: str, model_path: str) -> tuple[Any, dict[str, Any]]:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        expected_sha = self.expected_sha256[name]
        if actual_sha != expected_sha:
            raise RuntimeError(f"{name} context SHA mismatch: expected={expected_sha} actual={actual_sha}")
        print(f"{name.upper()}_CONTEXT_SHA_GATE=PASS")

        model = self._create_model(str(path))
        config = self.aidlite.Config.create_instance()
        if model is None or config is None:
            raise RuntimeError(f"{name}: Model/Config creation failed")
        config.framework_type = self.aidlite.FrameworkType.TYPE_QNN240
        config.implement_type = self.aidlite.ImplementType.TYPE_LOCAL
        config.accelerate_type = self.aidlite.AccelerateType.TYPE_DSP
        config.qnn_shared_buffer = 0

        interpreter = self._build_interpreter(model, config)
        if interpreter is None:
            raise RuntimeError(f"{name}: interpreter creation failed")
        init_rc = normalize_rc(interpreter.init())
        load_rc = normalize_rc(interpreter.load_model())
        if init_rc != 0 or load_rc != 0:
            raise RuntimeError(f"{name}: init/load failed init={init_rc} load={load_rc}")

        inputs = self._flatten_tensor_info(interpreter.get_input_tensor_info())
        outputs = self._flatten_tensor_info(interpreter.get_output_tensor_info())
        actual_inputs = {item["name"]: item["element_count"] for item in inputs}
        actual_outputs = {item["name"]: item["element_count"] for item in outputs}
        expected = EXPECTED_TENSORS[name]
        if actual_inputs != expected["inputs"] or actual_outputs != expected["outputs"]:
            raise RuntimeError(f"{name}: tensor contract mismatch inputs={actual_inputs} outputs={actual_outputs}")

        print(f"{name.upper()}_LOAD_GATE=PASS")
        return interpreter, {
            "name": name,
            "path": str(path),
            "sha256": actual_sha,
            "inputs": inputs,
            "outputs": outputs,
        }

    def _set_input(self, interpreter: Any, name: str, value: np.ndarray) -> float:
        tensor = np.ascontiguousarray(value, dtype=np.float32)
        start = time.perf_counter_ns()
        rc = normalize_rc(interpreter.set_input_tensor(in_tensor_tag=name, input_data=tensor))
        duration = elapsed_ms(start)
        if rc != 0:
            raise RuntimeError(f"set_input_tensor failed name={name} rc={rc}")
        return duration

    def _invoke(self, interpreter: Any, name: str) -> float:
        start = time.perf_counter_ns()
        rc = normalize_rc(interpreter.invoke())
        duration = elapsed_ms(start)
        if rc != 0:
            raise RuntimeError(f"{name}: invoke failed rc={rc}")
        return duration

    def _get_output(self, interpreter: Any, name: str, shape: tuple[int, ...]) -> tuple[np.ndarray, float]:
        start = time.perf_counter_ns()
        value = interpreter.get_output_tensor(out_tensor_tag=name)
        duration = elapsed_ms(start)
        if value is None:
            raise RuntimeError(f"get_output_tensor returned None name={name}")
        array = np.asarray(value, dtype=np.float32).reshape(shape)
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name}: non-finite output")
        return np.ascontiguousarray(array, dtype=np.float32), duration

    def run_frame(
        self,
        frame_index: int,
        frame_assets: dict[str, Any],
        repo_root: str | Path,
        previous_bev: np.ndarray | None,
        check_image_sha: bool = False,
        preprocess_workers: int = 6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        frame_start = time.perf_counter_ns()
        is_scene_start = bool(frame_assets.get("is_scene_start", False))
        assets = frame_assets["assets"]
        timing: dict[str, float] = {}

        preprocess_start = time.perf_counter_ns()
        images, image_preprocess = load_backbone_images(
            assets,
            repo_root,
            check_image_sha=check_image_sha,
            preprocess_workers=preprocess_workers,
        )
        timing["image_preprocess_ms"] = elapsed_ms(preprocess_start)
        can_bus = load_record(assets["can_bus"], repo_root).astype(np.float32)
        lidar2img = load_record(assets["lidar2img"], repo_root).astype(np.float32)

        timing["backbone_set_input_ms"] = self._set_input(self.interpreters["backbone"], "images", images)
        timing["backbone_invoke_ms"] = self._invoke(self.interpreters["backbone"], "backbone")
        img_feat, timing["backbone_get_output_ms"] = self._get_output(
            self.interpreters["backbone"], "img_feat", SHAPES["img_feat"]
        )
        img_feat_encoder = as_encoder_img_feat(img_feat)

        if is_scene_start:
            encoder_name = "encoder_scene_start"
            encoder = self.interpreters[encoder_name]
            timing["encoder_set_input_ms"] = 0.0
            for tensor_name, value in (("can_bus", can_bus), ("img_feat", img_feat_encoder), ("lidar2img", lidar2img)):
                timing["encoder_set_input_ms"] += self._set_input(encoder, tensor_name, value)
        else:
            if previous_bev is None:
                raise RuntimeError(f"frame{frame_index:03d}: previous bev is missing for temporal frame")
            shift = load_record(assets["shift"], repo_root).astype(np.float32)
            rotation_can_bus = load_record(assets["rotation_can_bus"], repo_root).astype(np.float32)
            rotate_start = time.perf_counter_ns()
            prev_bev = rotate_prev_bev(previous_bev, rotation_can_bus)
            timing["prev_bev_rotate_ms"] = elapsed_ms(rotate_start)
            encoder_name = "encoder_temporal"
            encoder = self.interpreters[encoder_name]
            timing["encoder_set_input_ms"] = 0.0
            for tensor_name, value in (
                ("can_bus", can_bus),
                ("img_feat", img_feat_encoder),
                ("lidar2img", lidar2img),
                ("shift", shift),
                ("prev_bev", prev_bev),
            ):
                timing["encoder_set_input_ms"] += self._set_input(encoder, tensor_name, value)

        timing["encoder_invoke_ms"] = self._invoke(encoder, encoder_name)
        bev_embed, timing["encoder_get_output_ms"] = self._get_output(encoder, "bev_embed", SHAPES["bev"])

        decoder = self.interpreters["decoder"]
        timing["decoder_set_input_ms"] = self._set_input(decoder, "bev_embed", bev_embed)
        timing["decoder_invoke_ms"] = self._invoke(decoder, "decoder")
        cls_scores, timing["decoder_get_cls_ms"] = self._get_output(decoder, "cls_scores", SHAPES["decoder"])
        bbox_preds, timing["decoder_get_bbox_ms"] = self._get_output(decoder, "bbox_preds", SHAPES["decoder"])

        timing["qnn_invoke_ms"] = (
            timing["backbone_invoke_ms"]
            + timing["encoder_invoke_ms"]
            + timing["decoder_invoke_ms"]
        )
        timing["frame_total_ms"] = elapsed_ms(frame_start)
        frame_result = {
            "frame_index": int(frame_index),
            "sample_token": frame_assets.get("sample_token"),
            "is_scene_start": is_scene_start,
            "encoder": encoder_name,
            "image_preprocess": image_preprocess,
            "timing_ms": timing,
            "status": "PASS",
        }
        return bev_embed, cls_scores, bbox_preds, frame_result

    def run_manifest(
        self,
        manifest_path: str | Path,
        repo_root: str | Path,
        output_dir: str | Path,
        nms_contract_path: str | Path,
        frame_start: int = 0,
        frame_count: int | None = None,
        save_all_raw: bool = False,
        visualize: bool = True,
        vis_score_thr: float = 0.0,
        vis_max_boxes: int = 80,
        check_image_sha: bool = False,
        preprocess_workers: int = 6,
    ) -> dict[str, Any]:
        run_manifest_start = time.perf_counter_ns()
        manifest_load_start = time.perf_counter_ns()
        manifest = load_json(manifest_path)
        nms_contract = load_json(nms_contract_path)
        manifest_load_ms = elapsed_ms(manifest_load_start)
        total_frames = int(manifest.get("total_frames", len(manifest["frames"])))
        end = total_frames if frame_count is None else min(total_frames, frame_start + frame_count)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        previous_bev: np.ndarray | None = None
        frames: dict[str, Any] = {}
        scene_start_count = 0
        temporal_count = 0
        frame_ms: list[float] = []
        qnn_invoke_ms: list[float] = []
        backbone_invoke_ms: list[float] = []
        encoder_invoke_ms: list[float] = []
        decoder_invoke_ms: list[float] = []
        image_preprocess_ms: list[float] = []
        raw_save_ms: list[float] = []
        nms_decode_ms: list[float] = []
        final_coordinates_save_ms: list[float] = []
        postprocess_ms: list[float] = []
        visualization_ms: list[float] = []
        final_outputs: dict[str, Any] | None = None
        final_coordinates: dict[str, Any] = {}
        visualizations: dict[str, Any] = {}

        for frame_index in range(frame_start, end):
            sample = f"sample_{frame_index:03d}"
            bev_embed, cls_scores, bbox_preds, frame_record = self.run_frame(
                frame_index,
                manifest["frames"][sample],
                repo_root,
                previous_bev,
                check_image_sha=check_image_sha,
                preprocess_workers=preprocess_workers,
            )
            previous_bev = np.ascontiguousarray(bev_embed, dtype=np.float32)
            frames[sample] = frame_record
            frame_ms.append(frame_record["timing_ms"]["frame_total_ms"])
            qnn_invoke_ms.append(frame_record["timing_ms"]["qnn_invoke_ms"])
            backbone_invoke_ms.append(frame_record["timing_ms"]["backbone_invoke_ms"])
            encoder_invoke_ms.append(frame_record["timing_ms"]["encoder_invoke_ms"])
            decoder_invoke_ms.append(frame_record["timing_ms"]["decoder_invoke_ms"])
            image_preprocess_ms.append(frame_record["timing_ms"].get("image_preprocess_ms", 0.0))
            if frame_record["is_scene_start"]:
                scene_start_count += 1
            else:
                temporal_count += 1
            raw_save_value = 0.0
            if save_all_raw or frame_index == end - 1:
                raw_save_start = time.perf_counter_ns()
                final_outputs = save_raw_outputs(output_path, frame_index, cls_scores, bbox_preds)
                raw_save_value = elapsed_ms(raw_save_start)
            raw_save_ms.append(raw_save_value)

            nms_start = time.perf_counter_ns()
            boxes, scores, labels = portable_nms.decode_numpy_nmsfreecoder(
                cls_scores,
                bbox_preds,
                nms_contract,
            )
            nms_value = elapsed_ms(nms_start)
            nms_decode_ms.append(nms_value)

            final_save_start = time.perf_counter_ns()
            final_coordinates[f"frame{frame_index:03d}"] = save_final_coordinates(
                output_path,
                frame_index,
                boxes,
                scores,
                labels,
            )
            final_save_value = elapsed_ms(final_save_start)
            final_coordinates_save_ms.append(final_save_value)
            postprocess_ms.append(raw_save_value + nms_value + final_save_value)

            visualization_value = 0.0
            if visualize:
                visualization_start = time.perf_counter_ns()
                visualizations[f"frame{frame_index:03d}"] = save_camera_grid_visualization(
                    output_path,
                    frame_index,
                    manifest["frames"][sample],
                    repo_root,
                    boxes,
                    scores,
                    labels,
                    score_thr=vis_score_thr,
                    max_boxes=vis_max_boxes,
                    result_dir=output_path,
                )
                visualization_value = elapsed_ms(visualization_start)
            visualization_ms.append(visualization_value)
            frame_record["timing_ms"]["raw_save_ms"] = raw_save_value
            frame_record["timing_ms"]["nms_decode_ms"] = nms_value
            frame_record["timing_ms"]["final_coordinates_save_ms"] = final_save_value
            frame_record["timing_ms"]["postprocess_ms"] = raw_save_value + nms_value + final_save_value
            frame_record["timing_ms"]["visualization_ms"] = visualization_value
            top_detections = []
            for det_index in range(min(5, len(scores))):
                label_id = int(labels[det_index])
                class_name = NUSCENES_CLASSES[label_id] if label_id < len(NUSCENES_CLASSES) else str(label_id)
                top_detections.append({
                    "box": [float(value) for value in boxes[det_index].tolist()],
                    "score": float(scores[det_index]),
                    "label": label_id,
                    "class_name": class_name,
                })
            frame_record["detections"] = {
                "count": int(len(scores)),
                "top": top_detections,
            }
            print(
                f"FRAME {frame_index:03d} PASS "
                f"preprocess={frame_record['image_preprocess']['source']} "
                f"image_preprocess_ms={frame_record['timing_ms']['image_preprocess_ms']:.3f} "
                f"encoder={frame_record['encoder']} "
                f"qnn_invoke_ms={frame_record['timing_ms']['qnn_invoke_ms']:.3f} "
                f"postprocess_ms={frame_record['timing_ms']['postprocess_ms']:.3f} "
                f"visualization_ms={frame_record['timing_ms']['visualization_ms']:.3f} "
                f"frame_inference_ms={frame_record['timing_ms']['frame_total_ms']:.3f}"
            )
            print(f"Detected {len(scores)} BEV boxes")
            if visualize:
                print(f"Camera grid saved: {visualizations[f'frame{frame_index:03d}']['path']}")
            for rank, det in enumerate(top_detections, start=1):
                box = det["box"]
                box_text = ", ".join(f"{value:.3f}" for value in box[:7])
                print(f"{rank} [{box_text}] {det['score']:.6f} {det['class_name']}")

        camera_grid_gif = None
        camera_grid_summary = None
        gif_ms = 0.0
        if visualize and visualizations:
            gif_start = time.perf_counter_ns()
            ordered_records = [visualizations[key] for key in sorted(visualizations)]
            camera_grid_gif = save_camera_grid_gif(ordered_records, output_path)
            camera_grid_summary = save_camera_grid_summary(output_path, ordered_records, camera_grid_gif)
            gif_ms = elapsed_ms(gif_start)
            if camera_grid_gif:
                print(f"Camera grid GIF saved: {camera_grid_gif['path']}")

        total_with_visualization_ms = elapsed_ms(run_manifest_start)
        total_without_visualization_ms = (
            manifest_load_ms
            + sum(frame_ms)
            + sum(postprocess_ms)
        )
        total_visualization_ms = sum(visualization_ms) + gif_ms

        return {
            "status": "PASS",
            "manifest": str(Path(manifest_path).resolve()),
            "nms_contract": str(Path(nms_contract_path).resolve()),
            "repo_root": str(Path(repo_root).resolve()),
            "frame_range": [int(frame_start), int(end - 1)] if end > frame_start else [],
            "completed_frames": int(end - frame_start),
            "scene_start_encoder_count": scene_start_count,
            "temporal_encoder_count": temporal_count,
            "models": self.model_records,
            "model_load_timing_ms": self.model_load_timing_ms,
            "end_to_end_timing_ms": {
                "manifest_and_contract_load_ms": manifest_load_ms,
                "complete_inference_no_visualization_ms": total_without_visualization_ms,
                "visualization_total_ms": total_visualization_ms,
                "complete_inference_with_visualization_ms": total_with_visualization_ms,
                "camera_grid_gif_ms": gif_ms,
            },
            "timing_ms": stats(frame_ms),
            "qnn_invoke_ms": stats(qnn_invoke_ms),
            "component_invoke_ms": {
                "backbone": stats(backbone_invoke_ms),
                "encoder": stats(encoder_invoke_ms),
                "decoder": stats(decoder_invoke_ms),
            },
            "image_preprocess_ms": stats(image_preprocess_ms),
            "postprocess_ms": stats(postprocess_ms),
            "nms_decode_ms": stats(nms_decode_ms),
            "raw_save_ms": stats(raw_save_ms),
            "final_coordinates_save_ms": stats(final_coordinates_save_ms),
            "visualization_ms": stats(visualization_ms),
            "timing_contract": {
                "frame_total_ms": "Per-frame inference path: image preprocessing plus model input/output and QNN invoke time; excludes NMS, result saving, and camera-grid visualization rendering.",
                "qnn_invoke_ms": "Backbone + encoder + decoder invoke time only.",
                "image_preprocess_ms": "Six-camera JPG decode, RGB conversion, normalization, resize, CHW conversion, and zero padding.",
                "complete_inference_no_visualization_ms": "Manifest load plus all frame inference, NMS, and result saving; excludes PNG/GIF rendering.",
                "complete_inference_with_visualization_ms": "Full run_manifest wall time including PNG/GIF rendering; excludes Python process startup and model loading, which are reported separately.",
            },
            "frames": frames,
            "final_outputs": final_outputs,
            "final_coordinates": final_coordinates,
            "visualizations": visualizations,
            "camera_grid_gif": camera_grid_gif,
            "camera_grid_summary": camera_grid_summary,
        }
