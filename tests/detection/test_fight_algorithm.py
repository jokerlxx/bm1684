from app.algorithms.fight import FightDetectionAlgorithm
from app.algorithms.support.event_state import NORMAL
from app.pipeline.messages import DetectionBox, FrameContext


def make_service(**overrides):
    fight_cfg = {
        "observation_duration": 1.0,
        "min_observation_duration": 0.3,
        "fight_threshold": 0.25,
        "cooldown_duration": 60,
    }
    fight_cfg.update(overrides)
    return FightDetectionAlgorithm(
        config={"fight_detection": fight_cfg},
        detector_name="fight",
    )


def run_frame(service, state, timestamp, outputs):
    context = FrameContext(
        stream_id=0,
        frame=__import__("numpy").zeros((360, 640, 3), dtype=__import__("numpy").uint8),
        frame_number=int(timestamp * 10),
        timestamp=float(timestamp),
    )
    return service.process(context, outputs, state)


def test_normal_misses_do_not_dilute_fight_ratio():
    service = make_service()
    state = {}

    first = run_frame(service, state, 0.0, [])
    second = run_frame(service, state, 0.2, [])
    third = run_frame(service, state, 0.4, [DetectionBox([10, 10, 50, 50], 0.5, 0)])

    assert first.metrics["state"] == NORMAL
    assert second.metrics["fight_buffer_size"] == 0
    assert third.metrics["fight_count"] == 1
    assert third.metrics["fight_buffer_size"] == 1


def test_fight_does_not_draw_boxes_before_window_confirmed():
    service = make_service()
    state = {}
    result = run_frame(service, state, 0.0, [DetectionBox([10, 10, 50, 50], 0.5, 0)])

    assert len(result.display_alerts) == 0


def test_fight_draws_boxes_after_window_confirmed():
    service = make_service()
    state = {}

    run_frame(service, state, 0.0, [])
    run_frame(service, state, 0.2, [])
    run_frame(service, state, 0.4, [DetectionBox([10, 10, 50, 50], 0.5, 0)])
    result = run_frame(service, state, 0.75, [DetectionBox([12, 12, 52, 52], 0.6, 0)])

    assert result.metrics["total_alerts"] == 1
    assert len(result.display_alerts) == 1
    assert result.display_alerts[0]["class_name"] == "fight"


def test_fight_alerts_after_suspecting_window():
    service = make_service()
    state = {}

    run_frame(service, state, 0.0, [])
    run_frame(service, state, 0.2, [])
    run_frame(service, state, 0.4, [DetectionBox([10, 10, 50, 50], 0.5, 0)])
    result = run_frame(service, state, 0.75, [DetectionBox([12, 12, 52, 52], 0.6, 0)])

    assert result.metrics["total_alerts"] == 1
    assert len(result.recordable_alerts) == 1
