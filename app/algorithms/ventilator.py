from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from app.algorithms.base import DetectionAlgorithm, algorithm_config
from app.algorithms.support.event_state import TrackEventStateMachine, timestamp_to_seconds
from app.algorithms.support.ventilator_support import KalmanBoxTrackerTimeBased, associate_detections_to_trackers
from app.pipeline.messages import AlgorithmResult, FrameContext


class VentilatorDetectionAlgorithm(DetectionAlgorithm):
    detector_type = "ventilator"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        super().__init__(config, detector_name=detector_name)
        ventilator_config = algorithm_config(self.config, "ventilator_detection")
        self.equipment_conf = float(ventilator_config.get("equipment_conf", 0.4))
        self.observation_duration = float(ventilator_config.get("observation_duration", 2.0))
        self.min_observation_duration = float(
            ventilator_config.get("min_observation_duration", min(1.5, self.observation_duration))
        )
        self.pass_threshold = float(ventilator_config.get("pass_threshold", 0.2))
        self.missing_equipment_threshold = float(
            ventilator_config.get("missing_equipment_threshold", 1.0 - self.pass_threshold)
        )
        self.cooldown_duration = float(ventilator_config.get("cooldown_duration", 180))
        self.alert_hold_seconds = float(ventilator_config.get("alert_hold_seconds", 0.2))
        self.mask_iou_threshold = float(ventilator_config.get("mask_iou_threshold", 0.15))
        self.tank_distance_coefficient = float(ventilator_config.get("tank_distance_coefficient", 2.0))
        self.tank_x_offset_coefficient = float(ventilator_config.get("tank_x_offset_coefficient", 1.0))
        self.head_iou_threshold = float(ventilator_config.get("head_iou_threshold", 0.3))
        self.head_center_distance_threshold_ratio = float(
            ventilator_config.get("head_center_distance_threshold_ratio", 1.8)
        )

    def _stream_state(self, state: Dict[str, Any], stream_id: int) -> Dict[str, Any]:
        streams = state.setdefault("streams", {})
        if stream_id not in streams:
            streams[stream_id] = {
                "trackers": [],
                "events": TrackEventStateMachine(
                    window_seconds=self.observation_duration,
                    threshold_ratio=self.missing_equipment_threshold,
                    min_observation_seconds=self.min_observation_duration,
                    cooldown_seconds=self.cooldown_duration,
                    alert_hold_seconds=self.alert_hold_seconds,
                ),
                "alarm_number": 0,
                "cooldown_start_time": 0.0,
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
            enabled=False,
            metrics={
                "masks": [],
                "tanks": [],
                "persons": [],
                "total_alerts": stream_state["total_alerts"],
            },
        )

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def check_equipment_wearing(self, head_bbox, masks, tanks):
        hx1, hy1, hx2, hy2 = head_bbox
        helmet_center_x = (hx1 + hx2) / 2
        helmet_center_y = (hy1 + hy2) / 2
        helmet_width = hx2 - hx1
        helmet_height = hy2 - hy1
        has_mask = False
        best_mask_iou = 0.0
        matched_mask = None
        for mask in masks:
            iou_value = self._calculate_iou(head_bbox, mask["bbox"])
            if iou_value > best_mask_iou:
                best_mask_iou = iou_value
                matched_mask = mask
            if iou_value > self.mask_iou_threshold:
                has_mask = True
                break

        has_tank = False
        best_tank_distance = float("inf")
        matched_tank = None
        for tank in tanks:
            tx1, ty1, tx2, ty2 = tank["bbox"]
            tank_center_x = (tx1 + tx2) / 2
            tank_center_y = (ty1 + ty2) / 2
            distance = np.sqrt((tank_center_x - helmet_center_x) ** 2 + (tank_center_y - helmet_center_y) ** 2)
            max_distance = self.tank_distance_coefficient * helmet_height
            x_offset = abs(tank_center_x - helmet_center_x)
            max_x_offset = self.tank_x_offset_coefficient * helmet_width
            is_below = tank_center_y > helmet_center_y
            if distance < best_tank_distance:
                best_tank_distance = distance
                matched_tank = tank
            if distance < max_distance and x_offset < max_x_offset and is_below:
                has_tank = True
                break
        return has_mask or has_tank, {
            "has_mask": has_mask,
            "has_tank": has_tank,
            "has_equipment": has_mask or has_tank,
            "mask_iou": best_mask_iou,
            "tank_distance": best_tank_distance if best_tank_distance != float("inf") else None,
            "matched_mask": matched_mask,
            "matched_tank": matched_tank,
        }

    def _update_trackers(self, stream_state, head_boxes, masks, tanks, timestamp):
        trackers = stream_state["trackers"]
        for tracker in trackers:
            tracker.updated_in_frame = False
        tracked_boxes = np.array([tracker.predict() for tracker in trackers]) if trackers else np.empty((0, 4))
        detection_boxes = np.array(head_boxes) if head_boxes else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detection_boxes,
            tracked_boxes,
            iou_threshold=self.head_iou_threshold,
            center_distance_threshold_ratio=self.head_center_distance_threshold_ratio,
        )

        for det_idx, trk_idx in matched:
            head_bbox = head_boxes[det_idx]
            trackers[trk_idx].update(head_bbox)
            has_equipment, _ = self.check_equipment_wearing(head_bbox, masks, tanks)
            trackers[trk_idx].last_has_equipment = has_equipment
            trackers[trk_idx].updated_in_frame = True

        for det_idx in unmatched_dets:
            head_bbox = head_boxes[det_idx]
            tracker = KalmanBoxTrackerTimeBased(head_bbox, observation_duration=self.observation_duration)
            has_equipment, _ = self.check_equipment_wearing(head_bbox, masks, tanks)
            tracker.last_has_equipment = has_equipment
            tracker.updated_in_frame = True
            trackers.append(tracker)

        stream_state["trackers"] = [tracker for tracker in trackers if tracker.time_since_update < 10]

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        stream_state = self._stream_state(state, frame_context.stream_id)
        equipment_output = inference_outputs["ventilator_equipment"]
        helmet_output = inference_outputs["ventilator_helmet"]

        masks = []
        tanks = []
        for item in equipment_output:
            bbox = [int(v) for v in item.bbox]
            if int(item.class_id) == 1 and float(item.confidence) >= self.equipment_conf:
                masks.append({"bbox": bbox, "confidence": float(item.confidence)})
            elif int(item.class_id) == 0 and float(item.confidence) >= self.equipment_conf:
                tanks.append({"bbox": bbox, "confidence": float(item.confidence)})

        head_boxes = [list(map(int, item.bbox)) for item in helmet_output if float(item.confidence) >= 0.3]
        self._update_trackers(stream_state, head_boxes, masks, tanks, frame_context.timestamp)
        current_time = timestamp_to_seconds(frame_context.timestamp, fallback=time.time())
        event_states: TrackEventStateMachine = stream_state["events"]

        alerts = []
        display_alerts = []
        persons = []
        active_tracker_ids = {tracker.id for tracker in stream_state["trackers"]}
        for tracker in stream_state["trackers"]:
            if getattr(tracker, "updated_in_frame", False):
                decision = event_states.update(
                    tracker.id,
                    current_time,
                    hit=not bool(getattr(tracker, "last_has_equipment", False)),
                )
            else:
                decision = event_states.peek(tracker.id, current_time)
            mask_wearing_rate = max(0.0, min(1.0, 1.0 - decision.hit_ratio))
            check_completed = decision.observation_span >= self.min_observation_duration
            check_passed = check_completed and mask_wearing_rate >= self.pass_threshold
            persons.append(
                {
                    "tracker_id": tracker.id,
                    "bbox": tracker.get_state(),
                    "mask_wearing_rate": mask_wearing_rate,
                    "observation_span": decision.observation_span,
                    "observation_count": decision.observation_count,
                    "check_completed": check_completed,
                    "check_passed": check_passed,
                    "alarm_triggered": decision.is_confirmed,
                    "event_state": decision.state,
                }
            )
            if not decision.should_display:
                continue
            display_info = {
                "tracker_id": tracker.id,
                "bbox": tracker.get_state(),
                "has_mask": False,
                "mask_wearing_rate": mask_wearing_rate,
                "observation_span": decision.observation_span,
                "observation_count": decision.observation_count,
                "event_state": decision.state,
                "is_recording": decision.just_alerted,
            }
            display_alerts.append(display_info)

            if not decision.just_alerted:
                continue
            stream_state["cooldown_start_time"] = current_time
            stream_state["total_alerts"] += 1
            tracker.alert_sent = True
            alerts.append(
                {
                    "tracker_id": tracker.id,
                    "bbox": tracker.get_state(),
                    "has_mask": False,
                    "mask_wearing_rate": mask_wearing_rate,
                    "observation_span": decision.observation_span,
                    "observation_count": decision.observation_count,
                    "event_state": decision.state,
                    "alert_id": stream_state["total_alerts"],
                    "alarm_time": frame_context.timestamp,
                }
            )

        event_states.cleanup(active_tracker_ids, current_time, stale_seconds=self.observation_duration + 2.0)
        stream_state["alarm_number"] = event_states.active_count(current_time)

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
                "masks": masks,
                "tanks": tanks,
                "persons": persons,
                "alarm_number": stream_state["alarm_number"],
                "cooldown_remaining_s": event_states.max_cooldown_remaining(current_time),
                "total_alerts": stream_state["total_alerts"],
            },
        )
