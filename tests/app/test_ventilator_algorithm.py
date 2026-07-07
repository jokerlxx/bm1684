from app.algorithms.ventilator import VentilatorDetectionAlgorithm
from app.algorithms.support.ventilator_support import KalmanBoxTrackerTimeBased, associate_detections_to_trackers
from app.pipeline.messages import DetectionBox, FrameContext


class FakeVentilatorTracker:
    def __init__(self):
        self.id = 7
        self.alert_sent = False
        self.time_since_update = 0
        self.updated_in_frame = True
        self.last_has_equipment = False

    def get_stats(self):
        return {
            "mask_wearing_rate": 0.0,
            "observation_span": 1.0,
            "observation_count": 25,
        }

    def get_state(self):
        return [10, 20, 30, 40]


def make_context(timestamp, frame_number=1):
    return FrameContext(stream_id=0, frame=None, frame_number=frame_number, timestamp=timestamp)


def make_algo():
    return VentilatorDetectionAlgorithm(
        {
            "ventilator_detection": {
                "observation_duration": 2.0,
                "min_observation_duration": 1.5,
                "pass_threshold": 0.2,
                "cooldown_duration": 180,
            }
        }
    )


def test_ventilator_display_alert_does_not_request_recording_during_cooldown(monkeypatch):
    algo = make_algo()
    monkeypatch.setattr(algo, "_update_trackers", lambda *args, **kwargs: None)
    tracker = FakeVentilatorTracker()
    state = {
        "streams": {
            0: {
                "trackers": [tracker],
                "events": algo._stream_state({}, 0)["events"],
                "alarm_number": 0,
                "cooldown_start_time": 0.0,
                "total_alerts": 0,
            }
        }
    }

    algo.process(make_context(0.0, frame_number=1), {"ventilator_equipment": [], "ventilator_helmet": []}, state)
    first_alert = algo.process(make_context(1.5, frame_number=2), {"ventilator_equipment": [], "ventilator_helmet": []}, state)
    result = algo.process(make_context(1.6, frame_number=3), {"ventilator_equipment": [], "ventilator_helmet": []}, state)

    assert len(first_alert.detections) == 1
    assert result.detections == []
    assert result.recordable_alerts == []
    assert result.display_alerts[0]["is_recording"] is False


def test_ventilator_display_alert_requests_recording_when_cooldown_allows(monkeypatch):
    algo = make_algo()
    monkeypatch.setattr(algo, "_update_trackers", lambda *args, **kwargs: None)
    state = {
        "streams": {
            0: {
                "trackers": [FakeVentilatorTracker()],
                "events": algo._stream_state({}, 0)["events"],
                "alarm_number": 0,
                "cooldown_start_time": 0.0,
                "total_alerts": 0,
            }
        }
    }

    algo.process(make_context(0.0, frame_number=1), {"ventilator_equipment": [], "ventilator_helmet": []}, state)
    result = algo.process(make_context(1.5, frame_number=2), {"ventilator_equipment": [], "ventilator_helmet": []}, state)

    assert len(result.detections) == 1
    assert result.recordable_alerts == result.detections
    assert result.display_alerts[0]["is_recording"] is True
    assert state["streams"][0]["total_alerts"] == 1


def test_ventilator_tracker_keeps_latest_detection_box_after_update():
    tracker = KalmanBoxTrackerTimeBased([10, 10, 30, 30])

    tracker.predict()
    tracker.update([25, 10, 45, 30])

    assert tracker.get_state() == [25.0, 10.0, 45.0, 30.0]


def test_ventilator_tracker_holds_last_box_when_detection_is_missing():
    tracker = KalmanBoxTrackerTimeBased([10, 10, 30, 30])
    tracker.update([25, 10, 45, 30])

    predicted = tracker.predict()

    assert predicted == [25.0, 10.0, 45.0, 30.0]
    assert tracker.get_state() == [25.0, 10.0, 45.0, 30.0]


def test_ventilator_head_tracker_matches_by_center_distance_when_iou_is_low():
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections=[[25, 10, 45, 30]],
        trackers=[[10, 10, 30, 30]],
        iou_threshold=0.3,
        center_distance_threshold_ratio=1.8,
    )

    assert matched.tolist() == [[0, 0]]
    assert unmatched_dets.tolist() == []
    assert unmatched_trks.tolist() == []


def test_ventilator_process_keeps_track_id_and_moves_bbox_with_person():
    algo = VentilatorDetectionAlgorithm({"ventilator_detection": {"head_center_distance_threshold_ratio": 1.8}})
    state = {}

    result_1 = algo.process(
        make_context(0.0, frame_number=1),
        {"ventilator_equipment": [], "ventilator_helmet": [DetectionBox([10, 10, 30, 30], 0.9, 0)]},
        state,
    )
    result_2 = algo.process(
        make_context(0.1, frame_number=2),
        {"ventilator_equipment": [], "ventilator_helmet": [DetectionBox([25, 10, 45, 30], 0.9, 0)]},
        state,
    )

    assert result_1.metrics["persons"][0]["tracker_id"] == result_2.metrics["persons"][0]["tracker_id"]
    assert result_2.metrics["persons"][0]["bbox"] == [25.0, 10.0, 45.0, 30.0]
