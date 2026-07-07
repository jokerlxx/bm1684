from __future__ import annotations

from collections import deque

import numpy as np


WINDOW_DOOR_CONFIG = {
    "window": {
        0: ("window_open", "窗户开", (0, 0, 255)),
        1: ("window_close", "窗户关", (0, 255, 0)),
    },
    "door": {
        2: ("door_close", "门关", (0, 255, 0)),
        3: ("door_open", "门开", (0, 0, 255)),
    },
}

ALERT_CLASSES = [0, 3]


class SimpleIOUTracker:
    def __init__(self, max_age=10, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        current_ids = {}
        matched_detections = set()
        for track_id, track in list(self.tracks.items()):
            track["age"] += 1
            if track["age"] > self.max_age:
                del self.tracks[track_id]
                continue
            best_iou = 0.0
            best_detection_idx = -1
            for det_idx, (det_bbox, det_conf, det_class) in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                iou = self._calculate_iou(track["bbox"], det_bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = det_idx
            if best_detection_idx != -1:
                det_bbox, det_conf, det_class = detections[best_detection_idx]
                self.tracks[track_id] = {"bbox": det_bbox, "age": 0, "class": det_class, "confidence": det_conf}
                current_ids[track_id] = self.tracks[track_id]
                matched_detections.add(best_detection_idx)
            else:
                current_ids[track_id] = track

        for det_idx, (det_bbox, det_conf, det_class) in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {"bbox": det_bbox, "age": 0, "class": det_class, "confidence": det_conf}
            current_ids[track_id] = self.tracks[track_id]

        tracked_objects = []
        for track_id, track_info in current_ids.items():
            bbox = track_info["bbox"]
            tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        return np.array(tracked_objects) if tracked_objects else np.empty((0, 5))

    def _calculate_iou(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0


class WindowDoorDetectionConfirmator:
    def __init__(self, track_id, target_type, observation_frames=60, detection_threshold=0.6, fps=30):
        self.track_id = track_id
        self.target_type = target_type
        self.observation_frames = observation_frames
        self.detection_threshold = detection_threshold
        self.fps = fps
        self.observing = False
        self.detection_window = deque(maxlen=observation_frames)
        self.has_alerted = False
        self.in_post_alert_monitoring = False
        self.recovery_window_frames = int(5.0 * fps)
        self.recovery_window = deque(maxlen=self.recovery_window_frames)
        self.last_seen_frame = 0

    def update(self, frame_number, detected):
        self.last_seen_frame = frame_number
        if self.has_alerted and not self.in_post_alert_monitoring:
            self.in_post_alert_monitoring = True
            self.recovery_window.clear()
        if self.in_post_alert_monitoring:
            normal_detected = not detected
            self.recovery_window.append(normal_detected)
            if len(self.recovery_window) >= self.recovery_window_frames:
                normal_frames = sum(self.recovery_window)
                normal_percentage = normal_frames / len(self.recovery_window)
                if normal_percentage >= 0.7:
                    self.reset()
                    return False, f"{self.target_type}已恢复正常"
            return False, f"{self.target_type}恢复监控中"
        if not self.observing:
            if detected:
                self.observing = True
                self.detection_window.clear()
                self.detection_window.append(True)
                return False, f"检测到{self.target_type}，开始观察"
            return False, f"未检测到{self.target_type}"
        self.detection_window.append(detected)
        frames_observed = len(self.detection_window)
        detection_frames = sum(self.detection_window)
        current_detection_percentage = detection_frames / frames_observed if frames_observed > 0 else 0.0
        if frames_observed >= self.observation_frames:
            if current_detection_percentage >= self.detection_threshold:
                self.has_alerted = True
                self.observing = False
                return True, f"{self.target_type}确认！(检测率: {current_detection_percentage:.1%})"
            self.observing = False
            return False, f"{self.target_type}阈值未达到(检测率: {current_detection_percentage:.1%})"
        return False, f"观察中({frames_observed}/{self.observation_frames}帧, 检测率: {current_detection_percentage:.1%})"

    def reset(self):
        self.observing = False
        self.detection_window.clear()
        self.has_alerted = False
        self.in_post_alert_monitoring = False
        self.recovery_window.clear()

    def get_status(self):
        return {
            "track_id": self.track_id,
            "target_type": self.target_type,
            "observing": self.observing,
            "has_alerted": self.has_alerted,
            "in_post_alert_monitoring": self.in_post_alert_monitoring,
            "frames_observed": len(self.detection_window),
            "detection_percentage": sum(self.detection_window) / len(self.detection_window) if len(self.detection_window) > 0 else 0.0,
        }

