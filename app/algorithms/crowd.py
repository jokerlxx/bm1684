from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from app.algorithms.base import DetectionAlgorithm, algorithm_config
from app.algorithms.support.event_state import TrackEventStateMachine, timestamp_to_seconds
from app.pipeline.messages import AlgorithmResult, FrameContext


def _expand_range_bbox(
    bbox: List[float],
    expand_ratio: float,
    min_pixels: float,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    expand_x = max(width * float(expand_ratio), float(min_pixels))
    expand_y = max(height * float(expand_ratio), float(min_pixels))
    rx1 = x1 - expand_x
    ry1 = y1 - expand_y
    rx2 = x2 + expand_x
    ry2 = y2 + expand_y
    if frame_width is not None and frame_width > 0:
        rx1 = max(0.0, rx1)
        rx2 = min(float(frame_width) - 1.0, rx2)
    if frame_height is not None and frame_height > 0:
        ry1 = max(0.0, ry1)
        ry2 = min(float(frame_height) - 1.0, ry2)
    return [rx1, ry1, rx2, ry2]


def _boxes_overlap(box_a: List[float], box_b: List[float]) -> bool:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _connected_components(count: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)

    groups: Dict[int, List[int]] = {}
    for index in range(count):
        root = find(index)
        groups.setdefault(root, []).append(index)
    return list(groups.values())


class CrowdDetectionAlgorithm(DetectionAlgorithm):
    detector_type = "crowd"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        super().__init__(config, detector_name=detector_name)
        crowd_config = algorithm_config(self.config, "crowd_detection")
        self.person_conf = float(crowd_config.get("person_conf", 0.5))
        self.min_crowd_count = max(2, int(crowd_config.get("min_crowd_count", 2)))
        self.range_expand_ratio = max(0.1, float(crowd_config.get("range_expand_ratio", 1.5)))
        self.range_expand_pixels_min = max(1.0, float(crowd_config.get("range_expand_pixels_min", 30.0)))
        self.observation_duration = float(crowd_config.get("observation_duration", 2.0))
        self.stability_ratio = float(crowd_config.get("stability_ratio", 0.6))
        self.min_observation_duration = float(
            crowd_config.get(
                "min_observation_duration",
                min(float(crowd_config.get("stability_duration", 1.0)), self.observation_duration),
            )
        )
        self.alert_hold_seconds = float(crowd_config.get("alert_hold_seconds", 0.2))
        self.cooldown_duration = float(crowd_config.get("cooldown_duration", 30))
        self.spatial_distance_threshold = float(crowd_config.get("spatial_distance", 150))
        self.cluster_track_timeout_s = float(
            crowd_config.get("cluster_track_timeout_s", max(self.observation_duration + 2.0, self.cooldown_duration))
        )
        self.detection_buffer_size = int(crowd_config.get("detection_buffer_size", 5))
        self.body_extend_ratio = float(crowd_config.get("body_extend_ratio", 3.0))

    def _stream_state(self, state: Dict[str, Any], stream_id: int) -> Dict[str, Any]:
        streams = state.setdefault("streams", {})
        if stream_id not in streams:
            streams[stream_id] = {
                "detection_buffer": deque(maxlen=self.detection_buffer_size),
                "events": TrackEventStateMachine(
                    window_seconds=self.observation_duration,
                    threshold_ratio=self.stability_ratio,
                    min_observation_seconds=self.min_observation_duration,
                    cooldown_seconds=self.cooldown_duration,
                    alert_hold_seconds=self.alert_hold_seconds,
                ),
                "cluster_tracks": {},
                "next_cluster_track_id": 1,
                "crowd_detection_start_time": None,
                "crowd_confirmed": False,
                "crowd_stability_frames": 0,
                "current_crowd_start_time": None,
                "last_save_time": 0.0,
                "last_recorded_positions": [],
            }
        return streams[stream_id]

    def build_disabled_result(self, frame_context: FrameContext, state: Dict[str, Any]) -> AlgorithmResult:
        total_alerts = int(state.get("total_alerts", 0))
        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=[],
            display_alerts=[],
            recordable_alerts=[],
            enabled=False,
            metrics={"total_alerts": total_alerts},
        )

    def _stretch_body_bbox(self, bbox: List[float], frame_height: float | None = None) -> List[float]:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        height = max(1.0, y2 - y1)
        stretched_y2 = y2 + height * self.body_extend_ratio
        if frame_height is not None and frame_height > 0:
            stretched_y2 = min(float(frame_height) - 1.0, stretched_y2)
        return [x1, y1, x2, max(y2, stretched_y2)]

    def detect_crowd_gathering(
        self,
        detections: List[Dict[str, Any]],
        frame_height: float | None = None,
        frame_width: float | None = None,
    ) -> Dict[str, Any]:
        if len(detections) < self.min_crowd_count:
            return {"is_crowd": False, "clusters": {}, "warnings": [], "n_clusters": 0, "n_noise": 0}

        range_boxes = [
            _expand_range_bbox(
                detection["bbox"],
                self.range_expand_ratio,
                self.range_expand_pixels_min,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for detection in detections
        ]
        person_boxes = [[float(value) for value in detection["bbox"][:4]] for detection in detections]

        edges: List[Tuple[int, int]] = []
        for anchor_idx, range_box in enumerate(range_boxes):
            for other_idx, other_box in enumerate(person_boxes):
                if anchor_idx == other_idx:
                    continue
                if _boxes_overlap(range_box, other_box):
                    left, right = sorted((anchor_idx, other_idx))
                    edges.append((left, right))

        groups = _connected_components(len(detections), edges)
        clusters = {}
        warnings = []
        cluster_id = 0
        for member_indices in groups:
            if len(member_indices) < self.min_crowd_count:
                continue
            member_indices = sorted(member_indices)
            cluster_detections = [detections[index] for index in member_indices]
            cluster_boxes = [person_boxes[index] for index in member_indices]
            min_x = min(box[0] for box in cluster_boxes)
            min_y = min(box[1] for box in cluster_boxes)
            max_x = max(box[2] for box in cluster_boxes)
            max_y = max(box[3] for box in cluster_boxes)
            head_bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            body_bbox = self._stretch_body_bbox(head_bbox, frame_height=frame_height)
            centers = [
                [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
                for box in cluster_boxes
            ]
            center = np.mean(np.array(centers, dtype=float), axis=0).tolist()
            clusters[cluster_id] = {
                "points": centers,
                "detections": cluster_detections,
                "count": len(member_indices),
                "bbox": body_bbox,
                "head_bbox": head_bbox,
            }
            warnings.append(
                {
                    "cluster_id": int(cluster_id),
                    "count": len(member_indices),
                    "bbox": body_bbox,
                    "head_bbox": head_bbox,
                    "center": center,
                }
            )
            cluster_id += 1

        return {
            "is_crowd": bool(warnings),
            "clusters": clusters,
            "warnings": warnings,
            "n_clusters": len(warnings),
            "n_noise": max(0, len(detections) - sum(item["count"] for item in warnings)),
        }

    def record_positions(self, stream_state: Dict[str, Any], warnings: List[Dict[str, Any]]) -> None:
        stream_state["last_recorded_positions"] = [np.array(item["center"]) for item in warnings if "center" in item][-10:]

    def _assign_cluster_event_ids(self, stream_state: Dict[str, Any], warnings: List[Dict[str, Any]], current_time: float) -> None:
        tracks = stream_state["cluster_tracks"]
        for warning in warnings:
            center = np.array(warning.get("center") or [0.0, 0.0], dtype=float)
            best_id = None
            best_distance = float("inf")
            for cluster_id, track in tracks.items():
                track_center = np.array(track.get("center") or [0.0, 0.0], dtype=float)
                distance = float(np.linalg.norm(center - track_center))
                if distance < best_distance:
                    best_distance = distance
                    best_id = cluster_id
            if best_id is None or best_distance > self.spatial_distance_threshold:
                best_id = int(stream_state["next_cluster_track_id"])
                stream_state["next_cluster_track_id"] += 1
                tracks[best_id] = {"center": center.tolist(), "last_seen": current_time}
            else:
                previous = np.array(tracks[best_id].get("center") or center, dtype=float)
                tracks[best_id]["center"] = (previous * 0.7 + center * 0.3).tolist()
                tracks[best_id]["last_seen"] = current_time
            warning["cluster_id"] = int(best_id)
            warning["event_id"] = int(best_id)

        for cluster_id, track in list(tracks.items()):
            if current_time - float(track.get("last_seen", current_time)) > self.cluster_track_timeout_s:
                tracks.pop(cluster_id, None)

    def _process_crowd_state(
        self,
        stream_id: int,
        crowd_result: Dict[str, Any],
        current_time: float,
        frame_number: int,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        del frame_number
        stream_state = self._stream_state(state, stream_id)
        total_alerts = int(state.get("total_alerts", 0))
        warnings = list(crowd_result.get("warnings") or [])
        self._assign_cluster_event_ids(stream_state, warnings, current_time)
        stream_state["detection_buffer"].append(bool(warnings))
        event_states: TrackEventStateMachine = stream_state["events"]

        crowd_alerts = []
        display_alerts = []
        current_event_ids = {warning["event_id"] for warning in warnings}
        decisions = {}
        for event_id in list(event_states.events.keys()):
            if event_id not in current_event_ids:
                decisions[event_id] = event_states.update(event_id, current_time, hit=False)
        for warning in warnings:
            decision = event_states.update(warning["event_id"], current_time, hit=True)
            decisions[warning["event_id"]] = decision
            if decision.just_alerted:
                total_alerts += 1
                state["total_alerts"] = total_alerts
                stream_state["last_save_time"] = current_time
                self.record_positions(stream_state, [warning])
                crowd_alerts.append(
                    {
                        "alert_id": total_alerts,
                        "cluster_id": warning["cluster_id"],
                        "count": warning["count"],
                        "bbox": warning["bbox"],
                        "center": warning["center"],
                        "event_state": decision.state,
                        "crowd_ratio": decision.hit_ratio,
                    }
                )

            if decision.should_display:
                display_alerts.append(
                    {
                        "cluster_id": warning["cluster_id"],
                        "count": warning["count"],
                        "bbox": warning["bbox"],
                        "center": warning["center"],
                        "is_confirmed": True,
                        "event_state": decision.state,
                        "crowd_ratio": decision.hit_ratio,
                        "is_recording": decision.just_alerted,
                        "info": dict(warning),
                    }
                )

        event_states.cleanup(current_event_ids, current_time, stale_seconds=self.cluster_track_timeout_s)
        current_decisions = [decisions[event_id] for event_id in current_event_ids if event_id in decisions]
        stream_state["crowd_confirmed"] = any(decision.is_confirmed for decision in current_decisions)
        stream_state["crowd_stability_frames"] = max((decision.hit_count for decision in current_decisions), default=0)
        if current_decisions and stream_state["crowd_detection_start_time"] is None:
            stream_state["crowd_detection_start_time"] = current_time
        if not current_decisions:
            stream_state["crowd_detection_start_time"] = None
            stream_state["crowd_confirmed"] = False
            stream_state["crowd_stability_frames"] = 0

        return {
            "crowd_alerts": crowd_alerts,
            "display_alerts": display_alerts,
            "is_stable_crowd": any(
                decision.observation_count > 0 and decision.hit_ratio >= self.stability_ratio
                for decision in current_decisions
            ),
            "crowd_confirmed": stream_state["crowd_confirmed"],
            "crowd_stability_frames": stream_state["crowd_stability_frames"],
            "detection_buffer": list(stream_state["detection_buffer"]),
        }

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        detections = [
            {
                "bbox": list(item.bbox),
                "confidence": float(item.confidence),
                "class_id": int(item.class_id),
                "class_name": item.class_name or "head",
            }
            for item in inference_outputs
            if int(item.class_id) in (0, 1) and float(item.confidence) >= self.person_conf
        ]
        frame_height = None
        frame_width = None
        frame_shape = getattr(frame_context.frame, "shape", None)
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_height = float(frame_shape[0])
            frame_width = float(frame_shape[1])
        elif frame_context.source_size and len(frame_context.source_size) >= 2:
            frame_width = float(frame_context.source_size[0])
            frame_height = float(frame_context.source_size[1])
        crowd_result = self.detect_crowd_gathering(
            detections,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        current_time = timestamp_to_seconds(frame_context.timestamp)
        result = self._process_crowd_state(
            frame_context.stream_id,
            crowd_result,
            current_time,
            frame_context.frame_number,
            state,
        )
        stream_state = self._stream_state(state, frame_context.stream_id)
        event_states: TrackEventStateMachine = stream_state["events"]
        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=result["crowd_alerts"],
            display_alerts=result["display_alerts"],
            recordable_alerts=result["crowd_alerts"],
            enabled=True,
            metrics={
                "total_alerts": int(state.get("total_alerts", 0)),
                "is_stable_crowd": result["is_stable_crowd"],
                "crowd_confirmed": result["crowd_confirmed"],
                "crowd_stability_frames": result["crowd_stability_frames"],
                "alarm_number": 1 if result["crowd_confirmed"] else 0,
                "cooldown_remaining_s": event_states.max_cooldown_remaining(current_time),
                "detection_buffer": result["detection_buffer"],
            },
        )
