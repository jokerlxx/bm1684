from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from app.algorithms.base import DetectionAlgorithm, algorithm_config
from app.algorithms.support.event_state import NORMAL, TrackEventStateMachine, timestamp_to_seconds
from app.pipeline.messages import AlgorithmResult, FrameContext


class FallDetectionAlgorithm(DetectionAlgorithm):
    detector_type = "fall"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        super().__init__(config, detector_name=detector_name)
        fall_config = algorithm_config(config, "fall_detection")
        self.observation_duration = float(fall_config.get("observation_duration", 1.0))
        self.fall_threshold = float(fall_config.get("fall_threshold", 0.25))
        self.min_observation_duration = float(
            fall_config.get("min_observation_duration", min(0.3, self.observation_duration))
        )
        self.cooldown_duration = float(fall_config.get("cooldown_duration", 60))
        self.alert_hold_seconds = float(fall_config.get("alert_hold_seconds", 0.2))

    def _stream_state(self, state: Dict[str, Any], stream_id: int) -> Dict[str, Any]:
        streams = state.setdefault("streams", {})
        if stream_id not in streams:
            streams[stream_id] = {
                "events": TrackEventStateMachine(
                    window_seconds=self.observation_duration,
                    threshold_ratio=self.fall_threshold,
                    min_observation_seconds=self.min_observation_duration,
                    cooldown_seconds=self.cooldown_duration,
                    alert_hold_seconds=self.alert_hold_seconds,
                ),
                "alarm_number": 0,
                "cooldown_start_time": 0.0,
                "last_fall_bbox": None,
                "total_alerts": 0,
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
                "state": "NORMAL",
                "num_falls": 0,
                "alarm_number": stream_state["alarm_number"],
                "cooldown_remaining_s": 0.0,
                "total_alerts": stream_state["total_alerts"],
            },
        )

    def _build_display_alerts(self, detections, decision) -> List[Dict[str, Any]]:
        return [
            {
                "bbox": list(item.bbox),
                "confidence": float(item.confidence),
                "class_id": int(item.class_id),
                "class_name": "fall",
                "is_fallen": True,
                "event_state": decision.state,
                "fall_percentage": decision.hit_ratio,
                "observation_span": decision.observation_span,
                "observation_count": decision.observation_count,
                "is_recording": decision.just_alerted,
            }
            for item in detections
        ]

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        stream_state = self._stream_state(state, frame_context.stream_id)
        detections = [item for item in inference_outputs if int(item.class_id) == 0]
        has_fall = len(detections) > 0
        if has_fall:
            stream_state["last_fall_bbox"] = list(detections[0].bbox)

        now = timestamp_to_seconds(frame_context.timestamp, fallback=time.time())
        event_states: TrackEventStateMachine = stream_state["events"]
        peek = event_states.peek("global", now)
        record_sample = has_fall or peek.state != NORMAL
        decision = event_states.update("global", now, hit=has_fall, valid=record_sample)

        display_alerts = (
            self._build_display_alerts(detections, decision) if decision.should_display else []
        )
        recordable_alerts: List[Dict[str, Any]] = []

        if decision.just_alerted:
            stream_state["cooldown_start_time"] = now
            stream_state["total_alerts"] += 1
            recordable_alerts.append(
                {
                    "alert_id": stream_state["total_alerts"],
                    "bbox": stream_state["last_fall_bbox"] or [0, 0, 100, 100],
                    "is_fallen": True,
                    "fall_percentage": decision.hit_ratio,
                    "observation_span": decision.observation_span,
                    "observation_count": decision.observation_count,
                    "confidence": float(detections[0].confidence) if detections else 0.0,
                    "alert_time": datetime.now(),
                    "event_state": decision.state,
                    "is_recording": True,
                    "has_current_detection": has_fall,
                }
            )

        stream_state["alarm_number"] = event_states.active_count(now)

        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=recordable_alerts,
            display_alerts=display_alerts,
            recordable_alerts=recordable_alerts,
            enabled=True,
            metrics={
                "current_detections": display_alerts,
                "state": decision.state,
                "num_falls": decision.hit_count if has_fall else 0,
                "fall_buffer_size": decision.observation_count,
                "fall_count": decision.hit_count,
                "alarm_number": stream_state["alarm_number"],
                "cooldown_remaining_s": event_states.max_cooldown_remaining(now),
                "total_alerts": stream_state["total_alerts"],
            },
        )
