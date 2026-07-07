#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2

from app.algorithms.base import algorithm_config
from app.application.detector_registry import DetectorRegistry
from app.model_runtime.model_type_registry import resolve_model_type
from app.model_runtime.runtime import ModelManager, ModelSpec
from app.pipeline.messages import FrameContext, InferenceResult
from app.pipeline.model_scheduler import DETECTOR_INPUTS, MODEL_ALIASES

try:
    import sophon.sail as sail
except ImportError:  # pragma: no cover - board script
    sail = None


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("full_video_alarm")


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _algorithm_instance(spec, config):
    algorithm_cls = spec.algorithm_cls
    if isinstance(algorithm_cls, str):
        module_name, class_name = algorithm_cls.split(":", 1)
        algorithm_cls = getattr(import_module(module_name), class_name)
    return algorithm_cls(config, detector_name=spec.detector_name)


def _model_section(model_key: str) -> str:
    return {
        "fall_detection": "fall_detection",
        "fight_detection": "fight_detection",
        "helmet_detection": "helmet_detection",
        "ventilator_equipment": "ventilator_detection",
        "window_door_inside": "window_door_detection",
        "window_door_outside": "window_door_detection",
    }.get(model_key, model_key)


def _resolve_model_path(config: Dict[str, Any], model_key: str) -> str:
    models = config.get("models") or {}
    model_path = models.get(model_key)
    if not model_path:
        alias_key = next((src for src, dst in MODEL_ALIASES.items() if dst == model_key and models.get(src)), None)
        model_path = models.get(alias_key) if alias_key else None
    if not model_path:
        raise ValueError(f"Missing model path for {model_key}")
    return str((ROOT / model_path).resolve() if not os.path.isabs(model_path) else model_path)


def _model_specs(config: Dict[str, Any], model_keys: Iterable[str]) -> List[ModelSpec]:
    specs = []
    for model_key in model_keys:
        model_types = config.get("model_types") or {}
        model_type = resolve_model_type(model_key, config).raw if model_key in model_types else None
        specs.append(
            ModelSpec(
                model_key=model_key,
                model_path=_resolve_model_path(config, model_key),
                model_type=model_type,
                device_id=int(config.get("bm1684x", {}).get("device_id", 0)),
                thresholds=algorithm_config(config, _model_section(model_key)),
            )
        )
    return specs


def _canonical_model_keys(config: Dict[str, Any], model_keys: Iterable[str]) -> Dict[str, str]:
    canonical_by_signature: Dict[tuple[str, str | None], str] = {}
    aliases: Dict[str, str] = {}
    model_types = config.get("model_types") or {}
    for model_key in model_keys:
        model_type = resolve_model_type(model_key, config).raw if model_key in model_types else None
        signature = (_resolve_model_path(config, model_key), model_type)
        canonical = canonical_by_signature.setdefault(signature, model_key)
        aliases[model_key] = canonical
    return aliases


def _normalize_outputs(detector_name: str, latest: Dict[str, InferenceResult]):
    inputs = DETECTOR_INPUTS[detector_name]
    if detector_name == "ventilator":
        return {
            "ventilator_equipment": latest["ventilator_equipment"].detections,
            "ventilator_helmet": latest["helmet_detection"].detections,
        }
    if len(inputs) == 1:
        return latest[inputs[0]].detections
    return {model_key: latest[model_key].detections for model_key in inputs}


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, float(fps)), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _draw_alerts(frame, detector_name: str, alerts: List[Dict[str, Any]]):
    output = frame.copy()
    for alert in alerts:
        bbox = alert.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(1, int(bbox[2]))
            y2 = y1 + max(1, int(bbox[3]))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{detector_name} #{alert.get('alert_id', '')}".strip()
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return output


def _draw_full_frame(
    frame,
    frame_number: int,
    timestamp_s: float,
    annotations: Dict[str, List[Dict[str, Any]]],
):
    output = frame.copy()
    colors = {
        "fall": (0, 0, 255),
        "fight": (0, 128, 255),
        "crowd": (255, 0, 255),
        "helmet": (0, 255, 255),
        "ventilator": (255, 128, 0),
        "window_door_inside": (0, 255, 0),
        "window_door_outside": (255, 255, 0),
    }
    active = 0
    for detector_name, alerts in annotations.items():
        color = colors.get(detector_name, (0, 0, 255))
        for alert in alerts:
            bbox = alert.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + max(1, int(bbox[2]))
                y2 = y1 + max(1, int(bbox[3]))
            active += 1
            label_parts = [detector_name]
            if alert.get("alert_id") is not None:
                label_parts.append(f"#{alert.get('alert_id')}")
            elif alert.get("track_id") is not None:
                label_parts.append(f"T{alert.get('track_id')}")
            elif alert.get("tracker_id") is not None:
                label_parts.append(f"T{alert.get('tracker_id')}")
            if alert.get("event_state"):
                label_parts.append(str(alert.get("event_state")))
            label = " ".join(label_parts)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text_y = max(20, y1 - 6)
            cv2.putText(output, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    header = f"frame {frame_number}  time {timestamp_s:.2f}s  active {active}"
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 360), 26), (0, 0, 0), -1)
    cv2.putText(output, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def _save_clip(
    output_dir: Path,
    detector_name: str,
    alerts: List[Dict[str, Any]],
    pre_frames: List[Dict[str, Any]],
    post_frames: List[Dict[str, Any]],
    trigger_frame,
    fps: float,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    first_alert = alerts[0] if alerts else {}
    alert_id = first_alert.get("alert_id", "unknown")
    base = f"{detector_name}_alert{alert_id}_{timestamp}"
    image_path = output_dir / f"{base}.jpg"
    video_path = output_dir / f"{base}.mp4"

    annotated = _draw_alerts(trigger_frame, detector_name, alerts)
    cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    frames = list(pre_frames) + list(post_frames)
    if not frames:
        frames = [{"frame": trigger_frame}]
    h, w = frames[0]["frame"].shape[:2]
    writer = _open_writer(video_path, fps=fps, size=(w, h))
    try:
        for item in frames:
            frame = item["frame"]
            if item.get("trigger"):
                frame = _draw_alerts(frame, detector_name, alerts)
            writer.write(frame)
    finally:
        writer.release()

    return {
        "detector": detector_name,
        "alert_id": alert_id,
        "alert_count": len(alerts),
        "video": str(video_path),
        "image": str(image_path),
        "frames": len(frames),
        "video_size": video_path.stat().st_size if video_path.exists() else 0,
        "image_size": image_path.stat().st_size if image_path.exists() else 0,
    }


def _resize_for_detection(frame, max_width: int, max_height: int):
    if max_width <= 0 or max_height <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _video_meta(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frame_count, width, height


class VideoFrameReader:
    def __init__(self, video_path: Path, device_id: int):
        self.video_path = video_path
        self.device_id = int(device_id)
        self.handle = None
        self.decoder = None
        self.cap = None

        if sail is not None:
            try:
                self.handle = sail.Handle(self.device_id)
                self.decoder = sail.Decoder(str(video_path), True, self.device_id)
                if self.decoder.is_opened():
                    LOGGER.info("using sail.Decoder for full video read")
                    return
                self.decoder.release()
            except Exception as exc:
                LOGGER.warning("sail.Decoder open failed, fallback to cv2.VideoCapture: %s", exc)
                self.decoder = None

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        LOGGER.info("using cv2.VideoCapture for full video read")

    def read(self):
        if self.decoder is not None:
            bmimg = sail.BMImage()
            ret = self.decoder.read(self.handle, bmimg)
            if ret != 0:
                return False, None
            frame = bmimg.asmat()
            return frame is not None, frame
        return self.cap.read()

    def release(self):
        if self.decoder is not None:
            try:
                self.decoder.release()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()


def run_all_detectors(
    detector_names: List[str],
    video_path: Path,
    config: Dict[str, Any],
    output_dir: Path,
    pre_seconds: float,
    post_seconds: float,
    max_alerts: int,
    save_alert_clips: bool = True,
    full_video_path: Path | None = None,
) -> Dict[str, Any]:
    registry = DetectorRegistry()
    algorithms = {}
    states: Dict[str, Dict[str, Any]] = {}
    for detector_name in detector_names:
        spec = registry.get(detector_name)
        algorithms[detector_name] = _algorithm_instance(spec, config)
        states[detector_name] = {}

    model_keys = []
    for detector_name in detector_names:
        for model_key in DETECTOR_INPUTS[detector_name]:
            if model_key not in model_keys:
                model_keys.append(model_key)
    model_aliases = _canonical_model_keys(config, model_keys)
    runtime_model_keys = []
    for model_key in model_keys:
        canonical = model_aliases[model_key]
        if canonical not in runtime_model_keys:
            runtime_model_keys.append(canonical)

    model_manager = ModelManager()
    model_manager.load_all(_model_specs(config, runtime_model_keys))

    fps, total_frames, _, _ = _video_meta(video_path)
    reader = VideoFrameReader(video_path, device_id=int(config.get("bm1684x", {}).get("device_id", 0)))
    max_width = int((config.get("bm1684x") or {}).get("detect_frame_max_width", 640))
    max_height = int((config.get("bm1684x") or {}).get("detect_frame_max_height", 360))
    pre_count = max(1, int(pre_seconds * fps))
    post_count = max(1, int(post_seconds * fps))
    pre_buffer = deque(maxlen=pre_count)
    pending: Dict[str, Dict[str, Any] | None] = {name: None for name in detector_names}
    saved: Dict[str, List[Dict[str, Any]]] = {name: [] for name in detector_names}
    raw_alert_events: Dict[str, int] = {name: 0 for name in detector_names}
    frame_number = 0
    started = time.time()
    full_video_writer = None

    LOGGER.info(
        "single-pass full run: detectors=%s models=%s frames=%s fps=%.2f detect_size<=%dx%d",
        detector_names,
        runtime_model_keys,
        total_frames,
        fps,
        max_width,
        max_height,
    )
    try:
        while True:
            ok, source_frame = reader.read()
            if not ok:
                break
            frame_number += 1
            frame = _resize_for_detection(source_frame, max_width=max_width, max_height=max_height)
            timestamp_s = frame_number / fps
            runtime_latest = {model_key: model_manager.infer(model_key, frame) for model_key in runtime_model_keys}
            latest = {model_key: runtime_latest[model_aliases[model_key]] for model_key in model_keys}
            if full_video_writer is None and full_video_path is not None:
                h, w = frame.shape[:2]
                full_video_writer = _open_writer(full_video_path, fps=fps, size=(w, h))
                LOGGER.info("full annotated video writer opened: %s size=%dx%d", full_video_path, w, h)
            context = FrameContext(
                stream_id=0,
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp_s,
                source_size=[frame.shape[1], frame.shape[0]],
            )
            frame_item = {"frame": frame.copy(), "frame_number": frame_number, "timestamp": timestamp_s}
            frame_annotations: Dict[str, List[Dict[str, Any]]] = {}

            for detector_name in detector_names:
                clip = pending.get(detector_name)
                if save_alert_clips and clip is not None:
                    clip["post_frames"].append(frame_item)
                    if len(clip["post_frames"]) >= post_count:
                        saved[detector_name].append(
                            _save_clip(
                                output_dir=output_dir / detector_name,
                                detector_name=detector_name,
                                alerts=clip["alerts"],
                                pre_frames=clip["pre_frames"],
                                post_frames=clip["post_frames"],
                                trigger_frame=clip["trigger_frame"],
                                fps=fps,
                            )
                        )
                        pending[detector_name] = None

                result = algorithms[detector_name].process(
                    context,
                    _normalize_outputs(detector_name, latest),
                    states[detector_name],
                )
                frame_annotations[detector_name] = [
                    dict(item)
                    for item in (result.display_alerts or result.recordable_alerts or result.detections or [])
                ]
                alerts = list(result.recordable_alerts or [])
                if not alerts:
                    continue
                raw_alert_events[detector_name] += len(alerts)
                if not save_alert_clips:
                    continue
                if pending.get(detector_name) is not None:
                    pending[detector_name]["alerts"].extend(dict(item) for item in alerts)
                    continue
                if max_alerts > 0 and len(saved[detector_name]) >= max_alerts:
                    continue
                trigger_item = dict(frame_item)
                trigger_item["trigger"] = True
                pending[detector_name] = {
                    "alerts": [dict(item) for item in alerts],
                    "pre_frames": list(pre_buffer),
                    "post_frames": [trigger_item],
                    "trigger_frame": frame.copy(),
                }
                LOGGER.info(
                    "[%s] alert frame=%d time=%.2fs count=%d",
                    detector_name,
                    frame_number,
                    timestamp_s,
                    len(alerts),
                )

            if full_video_writer is not None:
                full_video_writer.write(_draw_full_frame(frame, frame_number, timestamp_s, frame_annotations))
            pre_buffer.append(frame_item)
            if frame_number % 250 == 0:
                LOGGER.info(
                    "progress %d/%d saved=%s pending=%s elapsed=%.1fs",
                    frame_number,
                    total_frames,
                    {key: len(value) for key, value in saved.items()},
                    [key for key, value in pending.items() if value is not None],
                    time.time() - started,
                )
    finally:
        reader.release()
        if full_video_writer is not None:
            full_video_writer.release()
        model_manager.close()

    if save_alert_clips:
        for detector_name, clip in list(pending.items()):
            if clip is None:
                continue
            saved[detector_name].append(
                _save_clip(
                    output_dir=output_dir / detector_name,
                    detector_name=detector_name,
                    alerts=clip["alerts"],
                    pre_frames=clip["pre_frames"],
                    post_frames=clip["post_frames"],
                    trigger_frame=clip["trigger_frame"],
                    fps=fps,
                )
            )

    elapsed = time.time() - started
    return {
        "frames_processed": frame_number,
        "total_frames": total_frames,
        "elapsed_seconds": elapsed,
        "full_video": str(full_video_path) if full_video_path is not None else None,
        "full_video_size": full_video_path.stat().st_size if full_video_path is not None and full_video_path.exists() else 0,
        "detectors": [
            {
                "detector": detector_name,
                "raw_alert_events": raw_alert_events[detector_name],
                "saved_count": len(saved[detector_name]),
                "saved_alerts": saved[detector_name],
            }
            for detector_name in detector_names
        ],
    }

def run_detector(
    detector_name: str,
    video_path: Path,
    config: Dict[str, Any],
    output_dir: Path,
    pre_seconds: float,
    post_seconds: float,
    max_alerts: int,
) -> Dict[str, Any]:
    registry = DetectorRegistry()
    spec = registry.get(detector_name)
    model_keys = DETECTOR_INPUTS[detector_name]
    model_manager = ModelManager()
    model_manager.load_all(_model_specs(config, model_keys))
    algorithm = _algorithm_instance(spec, config)
    state: Dict[str, Any] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pre_count = max(1, int(pre_seconds * fps))
    post_count = max(1, int(post_seconds * fps))
    pre_buffer = deque(maxlen=pre_count)
    pending: List[Dict[str, Any]] = []
    saved: List[Dict[str, Any]] = []
    frame_number = 0
    started = time.time()

    LOGGER.info("[%s] start full video: frames=%s fps=%.2f", detector_name, total_frames, fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            timestamp_s = frame_number / fps
            latest = {}
            for model_key in model_keys:
                latest[model_key] = model_manager.infer(model_key, frame)

            context = FrameContext(
                stream_id=0,
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp_s,
                source_size=[frame.shape[1], frame.shape[0]],
            )
            result = algorithm.process(context, _normalize_outputs(detector_name, latest), state)
            frame_item = {"frame": frame.copy(), "frame_number": frame_number, "timestamp": timestamp_s}

            for clip in pending:
                clip["post_frames"].append(frame_item)
            completed = [clip for clip in pending if len(clip["post_frames"]) >= post_count]
            pending = [clip for clip in pending if len(clip["post_frames"]) < post_count]
            for clip in completed:
                saved.append(
                    _save_clip(
                        output_dir=output_dir,
                        detector_name=detector_name,
                        alert=clip["alert"],
                        pre_frames=clip["pre_frames"],
                        post_frames=clip["post_frames"],
                        trigger_frame=clip["trigger_frame"],
                        fps=fps,
                    )
                )

            for alert in result.recordable_alerts or []:
                if len(saved) + len(pending) >= max_alerts:
                    continue
                trigger_item = dict(frame_item)
                trigger_item["trigger"] = True
                pending.append(
                    {
                        "alert": dict(alert),
                        "pre_frames": list(pre_buffer),
                        "post_frames": [trigger_item],
                        "trigger_frame": frame.copy(),
                    }
                )
                LOGGER.info(
                    "[%s] alert frame=%d time=%.2fs alert=%s",
                    detector_name,
                    frame_number,
                    timestamp_s,
                    alert,
                )

            pre_buffer.append(frame_item)
            if frame_number % 250 == 0:
                LOGGER.info(
                    "[%s] progress %d/%d saved=%d pending=%d elapsed=%.1fs",
                    detector_name,
                    frame_number,
                    total_frames,
                    len(saved),
                    len(pending),
                    time.time() - started,
                )
    finally:
        cap.release()
        model_manager.close()

    for clip in pending:
        saved.append(
            _save_clip(
                output_dir=output_dir,
                detector_name=detector_name,
                alert=clip["alert"],
                pre_frames=clip["pre_frames"],
                post_frames=clip["post_frames"],
                trigger_frame=clip["trigger_frame"],
                fps=fps,
            )
        )

    elapsed = time.time() - started
    return {
        "detector": detector_name,
        "frames_processed": frame_number,
        "total_frames": total_frames,
        "elapsed_seconds": elapsed,
        "saved_alerts": saved,
        "saved_count": len(saved),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all algorithms over a full video and save alert clips.")
    parser.add_argument("--video", default="sichuan_test_3.mp4")
    parser.add_argument("--config", default="config_bm1684x.json")
    parser.add_argument("--output-dir", default="alarm_videos/sichuan_test_3_full_all")
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["fall", "fight", "crowd", "helmet", "ventilator", "window_door_inside", "window_door_outside"],
    )
    parser.add_argument("--pre-seconds", type=float, default=3.0)
    parser.add_argument("--post-seconds", type=float, default=3.0)
    parser.add_argument("--max-alerts-per-detector", type=int, default=0)
    parser.add_argument("--save-full-video", action="store_true", help="Save one full-length annotated result video.")
    parser.add_argument("--full-video-name", default="full_annotated.mp4")
    parser.add_argument("--skip-alert-clips", action="store_true", help="Do not save pre/post alert clips.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    video_path = (ROOT / args.video).resolve() if not os.path.isabs(args.video) else Path(args.video)
    config_path = (ROOT / args.config).resolve() if not os.path.isabs(args.config) else Path(args.config)
    output_dir = (ROOT / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_video_path = output_dir / args.full_video_name if args.save_full_video else None

    config = _load_config(config_path)
    fps, total_frames, width, height = _video_meta(video_path)
    LOGGER.info("video=%s frames=%d fps=%.2f size=%dx%d", video_path, total_frames, fps, width, height)

    summary = {
        "video": str(video_path),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(),
        "video_meta": {"fps": fps, "frames": total_frames, "width": width, "height": height},
        "detectors": [],
    }
    result = run_all_detectors(
            detector_names=args.detectors,
            video_path=video_path,
            config=config,
            output_dir=output_dir,
            pre_seconds=args.pre_seconds,
            post_seconds=args.post_seconds,
            max_alerts=max(0, int(args.max_alerts_per_detector)),
            save_alert_clips=not bool(args.skip_alert_clips),
            full_video_path=full_video_path,
        )
    summary.update(result)

    summary["finished_at"] = datetime.now().isoformat()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("summary saved: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
