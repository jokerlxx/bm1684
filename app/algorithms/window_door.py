from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from app.algorithms.base import DetectionAlgorithm, algorithm_config
from app.algorithms.support.window_door_support import (
    ALERT_CLASSES,
    WINDOW_DOOR_CONFIG,
    SimpleIOUTracker,
    WindowDoorDetectionConfirmator,
)
from app.pipeline.messages import AlgorithmResult, FrameContext


class WindowDoorDetectionAlgorithm(DetectionAlgorithm):
    detector_type = "window_door"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        super().__init__(config, detector_name=detector_name)
        window_door_config = algorithm_config(config, "window_door_detection")
        self.conf_threshold = float(window_door_config.get("conf_threshold", 0.5))
        self.iou_threshold = float(window_door_config.get("iou_threshold", 0.3))
        self.max_age = int(window_door_config.get("max_age", 10))
        self.observation_frames = int(window_door_config.get("observation_frames", 60))
        self.detection_threshold = float(window_door_config.get("detection_threshold", 0.6))
        self.cooldown_duration = float(window_door_config.get("cooldown_duration", 180))

    def _stream_state(self, state: Dict[str, Any], stream_id: int) -> Dict[str, Any]:
        streams = state.setdefault("streams", {})
        if stream_id not in streams:
            streams[stream_id] = {
                "tracker": SimpleIOUTracker(max_age=self.max_age, iou_threshold=self.iou_threshold),
                "confirmators": {},
                "continuous_alert_targets": {},
                "alarm_number": 0,
                "total_alerts": 0,
                "cooldown_start_time": 0.0,
            }
        return streams[stream_id]

    def build_disabled_result(self, frame_context: FrameContext, state: Dict[str, Any]) -> AlgorithmResult:
        stream_state = self._stream_state(state, frame_context.stream_id)
        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=[],
            display_alerts=[],
            enabled=False,
            metrics={
                "tracks": {},
                "alarm_number": stream_state["alarm_number"],
                "total_alerts": stream_state["total_alerts"],
            },
        )

    def _get_detection_type_and_config(self, cls_id: int):
        for detection_type, config in WINDOW_DOOR_CONFIG.items():
            if cls_id in config:
                return detection_type, config[cls_id]
        return None, None

    def _iou_match(self, box1, box2, threshold=0.5):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou > threshold

    def _cleanup_inactive_confirmators(self, frame_number, stream_state, inactive_threshold=60):
        to_remove = []
        for confirmator_key, confirmator in stream_state["confirmators"].items():
            if frame_number - confirmator.last_seen_frame > inactive_threshold:
                to_remove.append(confirmator_key)
        for key in to_remove:
            stream_state["confirmators"].pop(key, None)
            stream_state["continuous_alert_targets"].pop(key, None)

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        stream_state = self._stream_state(state, frame_context.stream_id)
        current_time = time.time()
        detections = [
            (list(item.bbox), float(item.confidence), int(item.class_id))
            for item in inference_outputs
            if float(item.confidence) >= self.conf_threshold
        ]
        tracked_objects = stream_state["tracker"].update(detections)

        alerts = []
        display_alerts = []
        current_tracked_ids = []

        for bbox, conf, cls_id in detections:
            if cls_id not in ALERT_CLASSES:
                continue
            detection_type, config_data = self._get_detection_type_and_config(cls_id)
            if detection_type is None:
                continue
            label, display_name, color = config_data

            track_id = None
            for tracked in tracked_objects:
                tracked_bbox = [tracked[0], tracked[1], tracked[2], tracked[3]]
                if self._iou_match(bbox, tracked_bbox):
                    track_id = int(tracked[4])
                    break
            if track_id is not None:
                current_tracked_ids.append(track_id)

            confirmator_key = f"{detection_type}_{cls_id}_{track_id}"
            if confirmator_key not in stream_state["confirmators"]:
                stream_state["confirmators"][confirmator_key] = WindowDoorDetectionConfirmator(
                    track_id=track_id,
                    target_type=display_name,
                    observation_frames=self.observation_frames,
                    detection_threshold=self.detection_threshold,
                    fps=30,
                )
            confirmator = stream_state["confirmators"][confirmator_key]
            confirmed, _ = confirmator.update(frame_context.frame_number, True)

            if confirmator.has_alerted:
                display_info = {
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": conf,
                    "label": label,
                    "display_name": display_name,
                    "color": color,
                    "detection_type": detection_type,
                    "is_recording": confirmed,
                }
                display_alerts.append(display_info)

            if not confirmed:
                continue
            if stream_state["alarm_number"] == 0:
                stream_state["alarm_number"] = 1
                stream_state["cooldown_start_time"] = current_time
                stream_state["total_alerts"] += 1
                alerts.append(
                    {
                        "alert_id": stream_state["total_alerts"],
                        "track_id": track_id,
                        "bbox": bbox,
                        "confidence": conf,
                        "label": label,
                        "display_name": display_name,
                        "color": color,
                        "detection_type": detection_type,
                        "alert_time": datetime.now(),
                    }
                )
            else:
                stream_state["alarm_number"] += 1

            stream_state["continuous_alert_targets"][confirmator_key] = {
                "bbox": bbox,
                "display_name": display_name,
                "conf": conf,
                "color": color,
                "detection_type": detection_type,
                "track_id": track_id,
            }

        if stream_state["alarm_number"] > 0 and current_time - stream_state["cooldown_start_time"] >= self.cooldown_duration:
            stream_state["alarm_number"] = 0

        self._cleanup_inactive_confirmators(frame_context.frame_number, stream_state)

        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=alerts,
            display_alerts=display_alerts,
            recordable_alerts=alerts,
            enabled=True,
            metrics={
                "tracks": {
                    track_id: {
                        "bbox": stream_state["tracker"].tracks[track_id]["bbox"],
                        "class": stream_state["tracker"].tracks[track_id]["class"],
                        "confidence": stream_state["tracker"].tracks[track_id]["confidence"],
                    }
                    for track_id in stream_state["tracker"].tracks
                },
                "continuous_alerts": dict(stream_state["continuous_alert_targets"]),
                "alarm_number": stream_state["alarm_number"],
                "cooldown_remaining_s": (
                    max(0.0, self.cooldown_duration - (current_time - stream_state["cooldown_start_time"]))
                    if stream_state["alarm_number"] > 0 else 0.0
                ),
                "total_alerts": stream_state["total_alerts"],
            },
        )
