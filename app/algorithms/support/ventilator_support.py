from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanBoxTrackerTimeBased:
    count = 0

    def __init__(self, bbox, observation_duration=10.0):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
            dtype=np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10
        initial_state = self._bbox_to_state(bbox)
        self.kf.statePost = initial_state.copy()
        self.kf.statePre = initial_state.copy()
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPre = np.eye(7, dtype=np.float32)
        self._last_bbox = self._state_to_bbox(initial_state)
        self._predicted_bbox = list(self._last_bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTrackerTimeBased.count
        KalmanBoxTrackerTimeBased.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.observation_duration = observation_duration
        self.mask_observations = deque()
        self.pass_threshold = 0.2
        self.fail_threshold = 0.8
        self.check_completed = False
        self.check_passed = False
        self.alarm_triggered = False
        self.alarm_time = None
        self.alert_sent = False

    @staticmethod
    def _state_to_bbox(state):
        x1, y1, w, h = state[0][0], state[1][0], state[2][0], state[3][0]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return [float(x1), float(y1), float(x1 + w), float(y1 + h)]

    @staticmethod
    def _bbox_to_state(bbox, velocity=None):
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        vx, vy, vw = velocity or (0.0, 0.0, 0.0)
        return np.array([[x1], [y1], [w], [h], [vx], [vy], [vw]], dtype=np.float32)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        self.kf.correct(np.array([[x1], [y1], [w], [h]], dtype=np.float32))
        state = self._bbox_to_state([x1, y1, x2, y2])
        self.kf.statePost = state.copy()
        self.kf.statePre = state.copy()
        self._last_bbox = self._state_to_bbox(state)
        self._predicted_bbox = list(self._last_bbox)

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self._predicted_bbox = list(self._last_bbox)
        return list(self._predicted_bbox)

    def get_state(self):
        if getattr(self, "_last_bbox", None) is not None:
            return list(self._last_bbox)
        state = self.kf.statePre if self.time_since_update > 0 else self.kf.statePost
        return self._state_to_bbox(state)

    def _cleanup_old_observations(self, current_time):
        cutoff_time = current_time - timedelta(seconds=self.observation_duration)
        while self.mask_observations and self.mask_observations[0][0] < cutoff_time:
            self.mask_observations.popleft()

    def update_mask_status(self, has_equipment, timestamp):
        self._cleanup_old_observations(timestamp)
        self.mask_observations.append((timestamp, has_equipment))
        if self.mask_observations:
            pass_count = sum(1 for _, equipment in self.mask_observations if equipment)
            total_count = len(self.mask_observations)
            pass_rate = pass_count / total_count
            if len(self.mask_observations) > 1:
                earliest_time = self.mask_observations[0][0]
                observation_span = (timestamp - earliest_time).total_seconds()
            else:
                observation_span = 0.0
            min_duration = self.observation_duration - 1.0
            min_frames = int(min_duration * 25)
            if (observation_span >= min_duration or total_count >= min_frames) and not self.check_completed:
                if pass_rate >= self.pass_threshold:
                    self.check_passed = True
                    self.check_completed = True
                else:
                    self.check_passed = False
                    self.check_completed = True
                    if not self.alarm_triggered:
                        self.alarm_triggered = True
                        self.alarm_time = datetime.now()
        return self.check_passed, self.alarm_triggered

    @property
    def mask_history(self):
        return [has_equipment for _, has_equipment in self.mask_observations]

    def get_stats(self):
        if self.mask_observations:
            earliest_time = self.mask_observations[0][0]
            latest_time = self.mask_observations[-1][0]
            observation_span = (latest_time - earliest_time).total_seconds()
            pass_count = sum(1 for _, equipment in self.mask_observations if equipment)
            total_count = len(self.mask_observations)
            pass_rate = pass_count / total_count
        else:
            observation_span = 0.0
            pass_rate = 0.0
            total_count = 0
        return {
            "tracker_id": self.id,
            "mask_wearing_rate": pass_rate,
            "observation_span": observation_span,
            "observation_count": total_count,
            "check_completed": self.check_completed,
            "alarm_triggered": self.alarm_triggered,
        }


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    wh = w * h
    return wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)


def _center_distance_norm(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    scale = max(abs(ax2 - ax1), abs(ay2 - ay1), abs(bx2 - bx1), abs(by2 - by1), 1.0)
    return float(np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / scale)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, center_distance_threshold_ratio=1.8):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    if min(iou_matrix.shape) > 0:
        matrix = (iou_matrix > iou_threshold).astype(np.int32)
        if matrix.sum(1).max() == 1 and matrix.sum(0).max() == 1:
            matched_indices = np.stack(np.where(matrix), axis=1)
        else:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = set(range(len(detections)))
    unmatched_trackers = set(range(len(trackers)))
    matches = []
    for match in matched_indices:
        det_idx = int(match[0])
        trk_idx = int(match[1])
        if iou_matrix[match[0], match[1]] < iou_threshold:
            continue
        matches.append([det_idx, trk_idx])
        unmatched_detections.discard(det_idx)
        unmatched_trackers.discard(trk_idx)

    center_candidates = []
    for det_idx in unmatched_detections:
        for trk_idx in unmatched_trackers:
            norm = _center_distance_norm(detections[det_idx], trackers[trk_idx])
            if norm <= center_distance_threshold_ratio:
                center_candidates.append((norm, int(det_idx), int(trk_idx)))
    for _, det_idx, trk_idx in sorted(center_candidates):
        if det_idx not in unmatched_detections or trk_idx not in unmatched_trackers:
            continue
        matches.append([det_idx, trk_idx])
        unmatched_detections.discard(det_idx)
        unmatched_trackers.discard(trk_idx)

    if not matches:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)
    return matches, np.array(sorted(unmatched_detections), dtype=int), np.array(sorted(unmatched_trackers), dtype=int)
