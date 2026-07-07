from __future__ import annotations


class SimpleIOUTracker:
    """IOU + normalized center distance tracker for helmet detections."""

    def __init__(self, max_age=10, iou_threshold=0.3, center_distance_threshold_ratio=1.6):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.center_distance_threshold_ratio = max(0.1, float(center_distance_threshold_ratio))
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        current_ids = {}
        matched_detections = set()
        matched_tracks = set()

        for track_id, track in list(self.tracks.items()):
            track["age"] += 1
            if track["age"] > self.max_age:
                del self.tracks[track_id]

        candidate_matches = []
        for track_id, track in self.tracks.items():
            for det_idx, (det_bbox, det_conf, det_class) in enumerate(detections):
                iou = self.calculate_iou(track["bbox"], det_bbox)
                norm_dist = self.calculate_normalized_center_distance(track["bbox"], det_bbox)
                if iou < self.iou_threshold and norm_dist > self.center_distance_threshold_ratio:
                    continue
                candidate_matches.append(
                    (
                        1 if track.get("class") == det_class else 0,
                        1 if iou >= self.iou_threshold else 0,
                        float(iou),
                        -float(norm_dist),
                        float(det_conf),
                        -int(track.get("age", 0)),
                        track_id,
                        det_idx,
                    )
                )

        candidate_matches.sort(reverse=True)
        for _, _, _, _, _, _, track_id, det_idx in candidate_matches:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            det_bbox, det_conf, det_class = detections[det_idx]
            self.tracks[track_id] = {"bbox": det_bbox, "age": 0, "class": det_class, "confidence": det_conf}
            current_ids[track_id] = self.tracks[track_id]
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

        for det_idx, (det_bbox, det_conf, det_class) in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {"bbox": det_bbox, "age": 0, "class": det_class, "confidence": det_conf}
            current_ids[track_id] = self.tracks[track_id]

        return current_ids

    def calculate_normalized_center_distance(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        cx1 = x1 + (w1 / 2.0)
        cy1 = y1 + (h1 / 2.0)
        cx2 = x2 + (w2 / 2.0)
        cy2 = y2 + (h2 / 2.0)
        center_distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        scale = max(float(w1), float(h1), float(w2), float(h2), 1.0)
        return center_distance / scale

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
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

