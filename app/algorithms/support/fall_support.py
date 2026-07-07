from __future__ import annotations

from collections import deque
from datetime import timedelta

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """Track a single bbox with a Kalman filter."""

    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))


class Sort:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for index, tracker in enumerate(trks):
            pos = self.trackers[index].predict()[0]
            tracker[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(index)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for index in reversed(to_del):
            self.trackers.pop(index)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)
        for match in matched:
            self.trackers[match[1]].update(dets[match[0], :])
        for index in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[index, :4]))

        ret = []
        idx = len(self.trackers)
        for tracker in reversed(self.trackers):
            bbox = tracker.get_state()[0]
            if tracker.time_since_update < 1 and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((bbox, [tracker.id + 1])).reshape(1, -1))
            idx -= 1
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(idx)
        if ret:
            return np.concatenate(ret)
        return np.empty((0, 5))

    @staticmethod
    def iou_batch(bb_test, bb_gt):
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        return wh / (
            (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
            - wh
        )

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = self.iou_batch(detections, trackers)
        if min(iou_matrix.shape) > 0:
            matrix = (iou_matrix > iou_threshold).astype(np.int32)
            if matrix.sum(1).max() == 1 and matrix.sum(0).max() == 1:
                matched_indices = np.stack(np.where(matrix), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for index, _ in enumerate(detections):
            if len(matched_indices) == 0 or index not in matched_indices[:, 0]:
                unmatched_detections.append(index)

        unmatched_trackers = []
        for index, _ in enumerate(trackers):
            if len(matched_indices) == 0 or index not in matched_indices[:, 1]:
                unmatched_trackers.append(index)

        matches = []
        for match in matched_indices:
            if iou_matrix[match[0], match[1]] < iou_threshold:
                unmatched_detections.append(match[0])
                unmatched_trackers.append(match[1])
            else:
                matches.append(match.reshape(1, 2))
        if not matches:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class PersonTrackerTimeBased:
    def __init__(self, tracker_id, observation_duration=5.0, fall_threshold=0.8):
        self.tracker_id = tracker_id
        self.observation_duration = observation_duration
        self.fall_threshold = fall_threshold
        self.fall_observations = deque()
        self.observing_fall = False
        self.has_alerted = False
        self.in_post_alert_monitoring = False
        self.alert_time = None
        self.current_fall_percentage = 0.0
        self.alert_frames = deque(maxlen=100)

    def _cleanup_old_observations(self, current_time):
        cutoff_time = current_time - timedelta(seconds=self.observation_duration)
        while self.fall_observations and self.fall_observations[0][0] < cutoff_time:
            self.fall_observations.popleft()

    def update(self, is_fallen, confidence, frame, frame_number, timestamp):
        self._cleanup_old_observations(timestamp)
        self.fall_observations.append((timestamp, is_fallen))
        if self.fall_observations:
            fall_count = sum(1 for _, fallen in self.fall_observations if fallen)
            total_count = len(self.fall_observations)
            self.current_fall_percentage = fall_count / total_count
        else:
            self.current_fall_percentage = 0.0

        if is_fallen:
            self.alert_frames.append({"frame": frame.copy(), "confidence": confidence, "frame_number": frame_number, "timestamp": timestamp})

        should_alert = False
        if not self.has_alerted and not self.in_post_alert_monitoring:
            if len(self.fall_observations) > 1:
                earliest_time = self.fall_observations[0][0]
                observation_span = (timestamp - earliest_time).total_seconds()
            else:
                observation_span = 0.0
            min_duration = self.observation_duration - 0.5
            min_frames = int(min_duration * 25)
            if observation_span >= min_duration or total_count >= min_frames:
                if self.current_fall_percentage >= self.fall_threshold:
                    self.observing_fall = True
                    should_alert = True
                    self.observing_fall = False
                else:
                    self.observing_fall = False
            else:
                self.observing_fall = False
        return should_alert

    def mark_alerted(self, timestamp):
        self.has_alerted = True
        self.alert_time = timestamp
        self.in_post_alert_monitoring = True

    def select_best_alert_frame(self):
        if not self.alert_frames:
            return None, 0.0, 0
        best_frame_data = max(self.alert_frames, key=lambda item: item["confidence"])
        return best_frame_data["frame"], best_frame_data["confidence"], best_frame_data["frame_number"]

    def get_stats(self):
        if self.fall_observations:
            earliest_time = self.fall_observations[0][0]
            latest_time = self.fall_observations[-1][0]
            observation_span = (latest_time - earliest_time).total_seconds()
        else:
            observation_span = 0.0
        return {
            "tracker_id": self.tracker_id,
            "fall_percentage": self.current_fall_percentage,
            "observation_span": observation_span,
            "observation_count": len(self.fall_observations),
            "observing": self.observing_fall,
            "has_alerted": self.has_alerted,
        }

