from app.algorithms.window_door import WindowDoorDetectionAlgorithm
from app.pipeline.messages import DetectionBox, FrameContext


def make_algo():
    return WindowDoorDetectionAlgorithm(
        {
            "window_door_detection": {
                "conf_threshold": 0.5,
                "observation_frames": 1,
                "detection_threshold": 1.0,
                "cooldown_duration": 180,
            }
        },
        detector_name="window_door_inside",
    )


def make_context(frame_number):
    return FrameContext(stream_id=0, frame=None, frame_number=frame_number, timestamp=float(frame_number))


def test_window_door_only_open_classes_trigger_alerts():
    algo = make_algo()
    state = {}
    closed = [
        DetectionBox([10, 10, 40, 40], 0.95, 1),
        DetectionBox([50, 10, 90, 50], 0.95, 2),
    ]

    result = algo.process(make_context(1), closed, state)

    assert result.detections == []
    assert result.display_alerts == []


def test_window_door_open_class_confirms_without_cleanup_error():
    algo = make_algo()
    state = {}
    open_window = [DetectionBox([10, 10, 40, 40], 0.95, 0)]

    algo.process(make_context(1), open_window, state)
    result = algo.process(make_context(2), open_window, state)

    assert len(result.detections) == 1
    assert result.detections[0]["label"] == "window_open"
    assert result.display_alerts[0]["is_recording"] is True


def test_window_door_respects_confidence_threshold():
    algo = make_algo()
    state = {}
    low_conf_open_door = [DetectionBox([10, 10, 40, 40], 0.2, 3)]

    result = algo.process(make_context(1), low_conf_open_door, state)

    assert result.detections == []
    assert result.display_alerts == []
