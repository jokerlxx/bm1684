from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from app.algorithms.base import DetectionAlgorithm, algorithm_config
from app.algorithms.support.event_state import TrackEventStateMachine, timestamp_to_seconds
from app.algorithms.support.helmet_support import SimpleIOUTracker
from app.pipeline.messages import AlgorithmResult, FrameContext


class HelmetDetectionAlgorithm(DetectionAlgorithm):
    detector_type = "helmet"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        super().__init__(config, detector_name=detector_name)
        helmet_config = algorithm_config(self.config, "helmet_detection")
        self.conf_threshold = float(helmet_config.get("conf_threshold", 0.3))
        self.iou_threshold = float(helmet_config.get("iou_threshold", 0.3))
        self.max_age = int(helmet_config.get("max_age", 10))
        self.center_distance_threshold_ratio = float(helmet_config.get("center_distance_threshold_ratio", 1.6))
        self.observation_duration = float(helmet_config.get("observation_duration", 1.5))
        self.no_helmet_threshold = float(helmet_config.get("no_helmet_threshold", 0.7))
        self.alert_duration = float(helmet_config.get("alert_duration", min(1.0, self.observation_duration)))
        self.timer_reset_grace_s = float(helmet_config.get("timer_reset_grace_s", 1.0))
        self.track_timeout_s = float(helmet_config.get("track_timeout_s", 1.2))
        self.cooldown_duration = float(helmet_config.get("cooldown_duration", 180))
        self.alert_hold_seconds = float(helmet_config.get("alert_hold_seconds", 0.2))

    def _stream_state(self, state: Dict[str, Any], stream_id: int) -> Dict[str, Any]:
        streams = state.setdefault("streams", {})
        if stream_id not in streams:
            streams[stream_id] = {
                "tracker": SimpleIOUTracker(
                    max_age=self.max_age,
                    iou_threshold=self.iou_threshold,
                    center_distance_threshold_ratio=self.center_distance_threshold_ratio,
                ),
                "events": TrackEventStateMachine(
                    window_seconds=self.observation_duration,
                    threshold_ratio=self.no_helmet_threshold,
                    min_observation_seconds=self.alert_duration,
                    cooldown_seconds=self.cooldown_duration,
                    alert_hold_seconds=self.alert_hold_seconds,
                ),
                "no_helmet_timers": {},
                "start_times": {},
                "last_seen": {},
                "last_no_helmet_seen": {},
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
            recordable_alerts=[],
            enabled=False,
            metrics={
                "tracks": {},
                "alarm_number": int(stream_state["alarm_number"]),
                "total_alerts": int(stream_state["total_alerts"]),
            },
        )

    def cleanup_old_tracks(self, current_time: float, stream_state: Dict[str, Any]) -> None:
        expired_ids = [
            track_id
            for track_id, last_seen_time in list(stream_state["last_seen"].items())
            if current_time - last_seen_time > self.track_timeout_s
        ]
        grace_expired_ids = [
            track_id
            for track_id, last_seen_time in list(stream_state["last_no_helmet_seen"].items())
            if current_time - last_seen_time > self.timer_reset_grace_s
        ]

        for track_id in grace_expired_ids:
            stream_state["no_helmet_timers"].pop(track_id, None)
            stream_state["start_times"].pop(track_id, None)
            stream_state["last_no_helmet_seen"].pop(track_id, None)

        for track_id in expired_ids:
            for mapping in (
                stream_state["no_helmet_timers"],
                stream_state["start_times"],
                stream_state["last_seen"],
                stream_state["last_no_helmet_seen"],
            ):
                mapping.pop(track_id, None)

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        stream_state = self._stream_state(state, frame_context.stream_id)
        current_time = timestamp_to_seconds(frame_context.timestamp, fallback=time.time())
        event_states: TrackEventStateMachine = stream_state["events"]
        detections = []
        for item in inference_outputs:
            if int(item.class_id) not in (0, 1):
                continue
            x1, y1, x2, y2 = item.bbox
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(item.confidence), int(item.class_id)))

        tracked_objects = stream_state["tracker"].update(detections)
        helmet_alerts: List[Dict[str, Any]] = []
        display_alerts: List[Dict[str, Any]] = []
        active_track_ids = set(tracked_objects.keys())

        for track_id, track_info in tracked_objects.items():
            class_id = int(track_info["class"])
            bbox = track_info["bbox"]
            confidence = float(track_info["confidence"])
            stream_state["last_seen"][track_id] = current_time
            decision = event_states.update(track_id, current_time, hit=(class_id == 1))

            if class_id != 1:
                stream_state["no_helmet_timers"].pop(track_id, None)
                stream_state["start_times"].pop(track_id, None)
                stream_state["last_no_helmet_seen"].pop(track_id, None)
                continue

            if track_id not in stream_state["start_times"]:
                stream_state["start_times"][track_id] = current_time
                stream_state["no_helmet_timers"][track_id] = 0.0
            stream_state["last_no_helmet_seen"][track_id] = current_time

            elapsed = decision.observation_span
            stream_state["no_helmet_timers"][track_id] = elapsed
            if not decision.should_display:
                continue

            display_info = {
                "track_id": track_id,
                "bbox": list(bbox),
                "confidence": confidence,
                "duration": elapsed,
                "no_helmet_ratio": decision.hit_ratio,
                "observation_count": decision.observation_count,
                "event_state": decision.state,
                "alert_time": datetime.now(),
                "is_recording": decision.just_alerted,
            }
            display_alerts.append(display_info)

            if not decision.just_alerted:
                continue
            stream_state["cooldown_start_time"] = current_time
            stream_state["total_alerts"] += 1
            helmet_alerts.append(
                {
                    "alert_id": stream_state["total_alerts"],
                    "track_id": track_id,
                    "bbox": list(bbox),
                    "confidence": confidence,
                    "duration": elapsed,
                    "no_helmet_ratio": decision.hit_ratio,
                    "observation_count": decision.observation_count,
                    "event_state": decision.state,
                    "alert_time": datetime.now(),
                }
            )

        self.cleanup_old_tracks(current_time, stream_state)
        event_states.cleanup(active_track_ids, current_time, stale_seconds=self.track_timeout_s)
        stream_state["alarm_number"] = event_states.active_count(current_time)

        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=helmet_alerts,
            display_alerts=display_alerts,
            recordable_alerts=helmet_alerts,
            enabled=True,
            metrics={
                "tracks": tracked_objects,
                "no_helmet_timers": dict(stream_state["no_helmet_timers"]),
                "alarm_number": stream_state["alarm_number"],
                "cooldown_remaining_s": event_states.max_cooldown_remaining(current_time),
                "total_alerts": stream_state["total_alerts"],
            },
        )
