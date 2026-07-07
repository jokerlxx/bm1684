import numpy as np

from app.algorithms.crowd import (
    CrowdDetectionAlgorithm,
    _boxes_overlap,
    _connected_components,
    _expand_range_bbox,
)
from app.pipeline.messages import DetectionBox, FrameContext


def make_service(**overrides):
    crowd_cfg = {
        "person_conf": 0.3,
        "min_crowd_count": 3,
        "range_expand_ratio": 1.5,
        "range_expand_pixels_min": 20,
        "observation_duration": 2.0,
        "stability_ratio": 0.3,
        "stability_duration": 1.0,
        "cooldown_duration": 30,
        "spatial_distance": 50,
    }
    crowd_cfg.update(overrides)
    return CrowdDetectionAlgorithm(
        config={"fps": 20, "crowd_detection": crowd_cfg},
        detector_name="crowd",
    )


def make_crowd_result(center=(100.0, 100.0)):
    return {
        "is_crowd": True,
        "clusters": {},
        "warnings": [
            {
                "cluster_id": 0,
                "count": 3,
                "bbox": [80.0, 80.0, 120.0, 120.0],
                "center": [float(center[0]), float(center[1])],
            }
        ],
        "n_clusters": 1,
        "n_noise": 0,
    }


def fill_and_confirm_stream(service, state, stream_id, base_time, center=(100.0, 100.0)):
    crowd_result = make_crowd_result(center=center)
    for idx, offset in enumerate((0.0, 0.2, 0.4, 0.6, 0.8), start=1):
        service._process_crowd_state(
            stream_id=stream_id,
            crowd_result=crowd_result,
            current_time=base_time + offset,
            frame_number=idx,
            state=state,
        )
    return service._process_crowd_state(
        stream_id=stream_id,
        crowd_result=crowd_result,
        current_time=base_time + 2.0,
        frame_number=6,
        state=state,
    )


def test_crowd_confirmation_uses_elapsed_time_instead_of_input_fps_frames():
    service = make_service()
    state = {}
    crowd_result = make_crowd_result()

    updates = []
    for idx, offset in enumerate((0.0, 0.2, 0.4, 0.6, 0.8), start=1):
        updates.append(
            service._process_crowd_state(
                stream_id=0,
                crowd_result=crowd_result,
                current_time=offset,
                frame_number=idx,
                state=state,
            )
        )

    assert updates[-1]["is_stable_crowd"] is True
    assert updates[-1]["crowd_confirmed"] is False
    assert updates[-1]["crowd_stability_frames"] >= 1

    confirmed = service._process_crowd_state(
        stream_id=0,
        crowd_result=crowd_result,
        current_time=2.0,
        frame_number=6,
        state=state,
    )

    assert confirmed["crowd_confirmed"] is True
    assert len(confirmed["crowd_alerts"]) == 1
    assert state["total_alerts"] == 1


def test_crowd_state_and_cooldown_are_isolated_per_stream():
    service = make_service()
    state = {}

    first = fill_and_confirm_stream(service, state=state, stream_id=0, base_time=0.0)
    second = fill_and_confirm_stream(service, state=state, stream_id=1, base_time=3.0)

    assert first["crowd_confirmed"] is True
    assert second["crowd_confirmed"] is True
    assert len(first["crowd_alerts"]) == 1
    assert len(second["crowd_alerts"]) == 1
    assert first["crowd_alerts"][0]["alert_id"] == 1
    assert second["crowd_alerts"][0]["alert_id"] == 2


def test_expand_range_bbox_uses_ratio_and_min_pixels():
    expanded = _expand_range_bbox([10, 10, 30, 30], expand_ratio=1.0, min_pixels=25)
    assert expanded == [-15.0, -15.0, 55.0, 55.0]


def test_boxes_overlap_requires_intersection():
    assert _boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15]) is True
    assert _boxes_overlap([0, 0, 10, 10], [20, 20, 30, 30]) is False


def test_connected_components_merges_transitive_neighbors():
    groups = _connected_components(4, [(0, 1), (1, 2)])
    assert sorted(len(group) for group in groups) == [1, 3]


def test_crowd_range_box_detects_three_close_heads():
    service = make_service(min_crowd_count=3, range_expand_ratio=1.0, range_expand_pixels_min=20)
    detections = [
        {"bbox": [0, 0, 100, 20], "confidence": 0.9, "class_id": 0},
        {"bbox": [90, 0, 190, 20], "confidence": 0.9, "class_id": 1},
        {"bbox": [180, 0, 280, 20], "confidence": 0.9, "class_id": 0},
    ]

    result = service.detect_crowd_gathering(detections, frame_height=120, frame_width=320)

    assert result["n_clusters"] == 1
    assert result["warnings"][0]["count"] == 3


def test_crowd_range_box_respects_min_crowd_count():
    service = make_service(min_crowd_count=3, range_expand_ratio=0.2, range_expand_pixels_min=5)
    detections = [
        {"bbox": [0, 0, 20, 20], "confidence": 0.9, "class_id": 0},
        {"bbox": [200, 0, 220, 20], "confidence": 0.9, "class_id": 1},
        {"bbox": [400, 0, 420, 20], "confidence": 0.9, "class_id": 0},
    ]

    result = service.detect_crowd_gathering(detections, frame_height=120, frame_width=640)

    assert result["warnings"] == []
    assert result["is_crowd"] is False


def test_crowd_range_box_allows_two_person_gathering_when_configured():
    service = make_service(min_crowd_count=2, range_expand_ratio=1.0, range_expand_pixels_min=20)
    detections = [
        {"bbox": [10, 10, 40, 40], "confidence": 0.9, "class_id": 0},
        {"bbox": [35, 10, 65, 40], "confidence": 0.9, "class_id": 1},
        {"bbox": [400, 10, 430, 40], "confidence": 0.9, "class_id": 0},
    ]

    result = service.detect_crowd_gathering(detections, frame_height=120, frame_width=640)

    assert result["n_clusters"] == 1
    assert result["warnings"][0]["count"] == 2


def test_crowd_uses_helmet_classes_and_stretches_alarm_bbox():
    service = make_service(min_crowd_count=3)
    state = {}
    outputs = [
        DetectionBox([10, 10, 20, 20], 0.9, 0),
        DetectionBox([30, 10, 40, 20], 0.9, 1),
        DetectionBox([50, 10, 60, 20], 0.9, 0),
        DetectionBox([70, 10, 80, 20], 0.9, 99),
    ]
    context = FrameContext(stream_id=0, frame=np.zeros((100, 100, 3), dtype=np.uint8), frame_number=1, timestamp=0.0)

    result = service.process(context, outputs, state)
    warning = result.metrics["detection_buffer"]
    crowd_result = service.detect_crowd_gathering(
        [
            {"bbox": [10, 10, 20, 20], "confidence": 0.9, "class_id": 0},
            {"bbox": [30, 10, 40, 20], "confidence": 0.9, "class_id": 1},
            {"bbox": [50, 10, 60, 20], "confidence": 0.9, "class_id": 0},
        ],
        frame_height=100,
        frame_width=100,
    )

    assert warning == [True]
    assert len(crowd_result["warnings"]) == 1
    assert crowd_result["warnings"][0]["head_bbox"] == [10.0, 10.0, 60.0, 20.0]
    assert crowd_result["warnings"][0]["bbox"][3] > 20.0
