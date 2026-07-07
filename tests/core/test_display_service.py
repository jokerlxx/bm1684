import queue
import time
from collections import deque

import numpy as np

from core.display_service import CELL_H, CELL_W, COMPOSITE_OUTPUT_H, COMPOSITE_OUTPUT_W, DisplayService


def make_service(stream_count=1, preview_cfg=None):
    if stream_count > 1:
        frame_queue = [queue.Queue() for _ in range(stream_count)]
    else:
        frame_queue = queue.Queue()
    service = DisplayService(
        frame_queue=frame_queue,
        result_queues={det: queue.Queue() for det in DisplayService._RENDER_DETECTORS},
        output_queue=queue.Queue(),
        control_queue=queue.Queue(),
        fps=20,
        stream_count=stream_count,
        preview_cfg=preview_cfg,
    )
    return service


def shutdown_service(service):
    service.video_buffer.shutdown(timeout=1)


def test_collect_latest_frame_batch_keeps_latest_per_stream_and_resizes():
    service = make_service(stream_count=2)
    try:
        service.frame_queues[0].put({
            "frame": np.zeros((360, 640, 3), dtype=np.uint8),
            "frame_number": 1,
            "timestamp": None,
            "stream_id": 0,
        })
        service.frame_queues[0].put({
            "frame": np.ones((360, 640, 3), dtype=np.uint8),
            "frame_number": 2,
            "timestamp": None,
            "stream_id": 0,
        })
        service.frame_queues[1].put({
            "frame": np.full((360, 640, 3), 2, dtype=np.uint8),
            "frame_number": 3,
            "timestamp": None,
            "stream_id": 1,
        })

        count = service._absorb_latest_frames()
        latest = service.latest_frames

        assert count == 2
        assert sorted(latest.keys()) == [0, 1]
        assert latest[0]["frame_number"] == 2
        assert latest[1]["frame_number"] == 3
        assert latest[0]["frame"].shape == (CELL_H, CELL_W, 3)
        assert latest[1]["frame"].shape == (CELL_H, CELL_W, 3)
        assert latest[0]["original_width"] == 640
        assert latest[0]["original_height"] == 360
        assert service._perf_window["input_frames"] == 2
        assert service._perf_window["stream_hits"] == [1, 1]
    finally:
        shutdown_service(service)


def test_collect_latest_single_stream_keeps_source_preview_resolution():
    service = make_service(stream_count=1)
    try:
        service.frame_queues[0].put({
            "frame": np.zeros((360, 640, 3), dtype=np.uint8),
            "frame_number": 1,
            "timestamp": None,
            "stream_id": 0,
        })

        count = service._absorb_latest_frames()
        latest = service.latest_frames[0]

        assert count == 1
        assert latest["frame"].shape == (360, 640, 3)
        assert latest["original_width"] == 640
        assert latest["original_height"] == 360
    finally:
        shutdown_service(service)


def test_get_result_for_frame_uses_fresh_sticky_result_only():
    service = make_service()
    try:
        service.sticky_results["helmet"] = {"value": "latest", "frame_number": 12, "timestamp": None}

        assert service._get_result_for_frame("helmet", 0, 12)["value"] == "latest"
        assert service._get_result_for_frame("helmet", 0, 25)["value"] == "latest"
        assert service._get_result_for_frame("helmet", 0, 40) is None
        assert service._get_result_for_frame("helmet", 0, 80) is None
        assert service._get_result_for_frame("fight", 0, 40) is None
    finally:
        shutdown_service(service)


def test_smooth_bbox_does_not_lag_latest_detection():
    service = make_service()
    try:
        first = service._smooth_bbox("helmet", 0, 1, [0, 0, 10, 10], "xyxy")
        second = service._smooth_bbox("helmet", 0, 1, [100, 100, 110, 110], "xyxy")

        assert first == [0.0, 0.0, 10.0, 10.0]
        assert second == [100.0, 100.0, 110.0, 110.0]
    finally:
        shutdown_service(service)


def test_render_results_fast_path_returns_original_frame_without_overlays():
    service = make_service()
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        rendered = service._render_results(
            frame,
            frame_number=5,
            timestamp=None,
            stream_id=0,
            original_size=(CELL_W, CELL_H),
        )

        assert rendered is frame
    finally:
        shutdown_service(service)


def test_render_results_uses_detection_coord_size_for_bbox_scaling():
    service = make_service()
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        service.sticky_results["helmet"] = {
            "frame_number": 5,
            "timestamp": None,
            "coord_width": 640,
            "coord_height": 360,
            "display_alerts": [
                {"track_id": 1, "bbox": [160, 90, 160, 90]},
            ],
        }
        captured = []

        def capture_box(frame_arg, bbox, chinese_text, english_text, color=(0, 0, 255)):
            del chinese_text, english_text, color
            captured.append([int(v) for v in bbox])
            return frame_arg

        service._draw_unified_alert_box = capture_box
        service._render_results(
            frame,
            frame_number=5,
            timestamp=None,
            stream_id=0,
            original_size=(1280, 720),
        )

        assert captured == [[160, 90, 320, 180]]
    finally:
        shutdown_service(service)


def test_render_results_draws_raw_model_boxes_when_enabled():
    service = make_service(preview_cfg={"raw_model_boxes_enabled": True})
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        service.sticky_results["ventilator"] = {
            "frame_number": 5,
            "timestamp": None,
            "coord_width": 640,
            "coord_height": 360,
            "raw_model_detections": {
                "ventilator_equipment": [
                    {"bbox": [160, 90, 320, 180], "confidence": 0.82, "class_id": 1}
                ],
                "helmet_detection": [
                    {"bbox": [320, 180, 480, 270], "confidence": 0.91, "class_id": 0}
                ],
            },
            "display_alerts": [],
            "detections": [],
            "persons": [],
        }
        captured = []

        def capture_box(frame_arg, bbox, label, color, coord_wh, disp_wh):
            del label, color, coord_wh, disp_wh
            captured.append([int(v) for v in bbox])
            return frame_arg

        service._draw_raw_model_box = capture_box
        service._render_results(
            frame,
            frame_number=5,
            timestamp=None,
            stream_id=0,
            original_size=(1280, 720),
        )

        assert captured == [[160, 90, 320, 180], [320, 180, 480, 270]]
    finally:
        shutdown_service(service)


def test_render_results_draws_cached_alert_without_new_detection_result():
    service = make_service(preview_cfg={"alert_box_hold_s": 1.2})
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        service.update_active_alert_box(
            "helmet",
            0,
            7,
            [10, 20, 30, 40],
            label=("未佩戴安全帽", "no helmet"),
            bbox_format="xywh",
            result={"_display_received_mono": time.monotonic()},
            coord_wh=(CELL_W, CELL_H),
        )
        captured = []

        def capture_box(frame_arg, bbox, chinese_text, english_text, color=(0, 0, 255)):
            del chinese_text, english_text, color
            captured.append([int(v) for v in bbox])
            return frame_arg

        service._draw_unified_alert_box = capture_box
        rendered = service._render_results(
            frame,
            frame_number=99,
            timestamp=None,
            stream_id=0,
            original_size=(CELL_W, CELL_H),
        )

        assert rendered is not frame
        assert captured == [[10, 20, 40, 60]]
    finally:
        shutdown_service(service)


def test_stale_cached_alert_expires_after_hold_window():
    service = make_service(preview_cfg={"alert_box_hold_s": 1.2})
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        service.update_active_alert_box(
            "helmet",
            0,
            7,
            [10, 20, 30, 40],
            label=("未佩戴安全帽", "no helmet"),
            bbox_format="xywh",
            result={"_display_received_mono": time.monotonic() - 2.0},
            coord_wh=(CELL_W, CELL_H),
        )

        rendered = service._render_results(
            frame,
            frame_number=99,
            timestamp=None,
            stream_id=0,
            original_size=(CELL_W, CELL_H),
        )

        assert rendered is frame
        assert 7 not in service.active_alerts["helmet"]
    finally:
        shutdown_service(service)


def test_draw_bilingual_label_reuses_cached_sprite():
    service = make_service()
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        out1 = service._draw_bilingual_label(frame.copy(), [10, 40, 120, 120], "", "alert")
        cache_size = len(service._label_sprite_cache)
        out2 = service._draw_bilingual_label(frame.copy(), [10, 40, 120, 120], "", "alert")

        assert cache_size == 1
        assert len(service._label_sprite_cache) == 1
        assert out1.shape == frame.shape
        assert out2.shape == frame.shape
        assert np.count_nonzero(out1) > 0
    finally:
        shutdown_service(service)

def test_composite_uses_fixed_four_grid():
    service = make_service(stream_count=4)
    try:
        for sid in range(4):
            service.latest_frames[sid] = {
                "frame": np.full((CELL_H, CELL_W, 3), sid + 1, dtype=np.uint8),
                "frame_number": sid + 1,
                "timestamp": None,
                "stream_id": sid,
                "original_width": CELL_W,
                "original_height": CELL_H,
            }

        composite = service._build_composite_frame()

        assert composite.shape == (COMPOSITE_OUTPUT_H, COMPOSITE_OUTPUT_W, 3)
        assert np.all(composite[10, 10] == 1)
        assert np.all(composite[10, CELL_W + 10] == 2)
        assert np.all(composite[CELL_H + 10, 10] == 3)
        assert np.all(composite[CELL_H + 10, CELL_W + 10] == 4)
    finally:
        shutdown_service(service)



def test_composite_reuses_rendered_cells_until_frame_or_result_changes():
    service = make_service(stream_count=2)
    try:
        for sid in range(2):
            service.latest_frames[sid] = {
                "frame": np.full((CELL_H, CELL_W, 3), sid + 1, dtype=np.uint8),
                "frame_number": 10 + sid,
                "timestamp": None,
                "stream_id": sid,
                "original_width": CELL_W,
                "original_height": CELL_H,
            }

        calls = []
        original_render = service._render_results

        def counting_render(frame, frame_number, timestamp, stream_id=0, original_size=None):
            calls.append(stream_id)
            return original_render(frame, frame_number, timestamp, stream_id, original_size)

        service._render_results = counting_render

        service._build_composite_frame()
        service._build_composite_frame()
        assert calls == [0, 1]

        service._bump_result_version("helmet", 0)
        service._build_composite_frame()
        assert calls == [0, 1, 0]
    finally:
        shutdown_service(service)

def test_clear_sticky_for_stream_clears_alert_and_smoothing_state():
    service = make_service(stream_count=2)
    try:
        service.sticky_results["helmet"][1] = {"display_alerts": []}
        service.latest_results["helmet"][1] = {"display_alerts": []}
        service.active_alerts["helmet"][1] = {7: {"bbox": [1, 2, 3, 4]}}
        service.triggered_alarms["helmet"][1] = {7}
        service._active_result_streams["helmet"].add(1)
        service._result_by_frame["helmet"] = {1: deque([(10, {"display_alerts": []})])}
        service._bbox_smooth_cache[("helmet", 1, 7)] = {"bbox": [1, 2, 3, 4]}

        service._clear_sticky_for_stream(1)

        assert 1 not in service.sticky_results["helmet"]
        assert 1 not in service.latest_results["helmet"]
        assert 1 not in service.active_alerts["helmet"]
        assert 1 not in service.triggered_alarms["helmet"]
        assert 1 not in service._active_result_streams["helmet"]
        assert 1 not in service._result_by_frame["helmet"]
        assert ("helmet", 1, 7) not in service._bbox_smooth_cache
        assert 1 not in service._result_versions["helmet"]
        assert 1 not in service._rendered_cell_cache
    finally:
        shutdown_service(service)



def test_merged_preview_draws_label_for_every_simple_box():
    service = make_service(stream_count=2, preview_cfg={"merged_simple_alert_boxes": True})
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        calls = []
        original_label = service._draw_bilingual_label

        def counting_label(*args, **kwargs):
            calls.append((args, kwargs))
            return original_label(*args, **kwargs)

        service._draw_bilingual_label = counting_label
        service._begin_alert_render_for_stream(0)
        out = service._draw_unified_alert_box(frame, [10, 40, 80, 100], "告警", "alert")
        out = service._draw_unified_alert_box(out, [90, 40, 150, 100], "告警", "alert")

        assert out is frame
        assert len(calls) == 2
        assert service._alert_draw_counts[0] == 2
        assert service._alert_label_counts[0] == 2
        assert np.count_nonzero(frame) > 0
    finally:
        shutdown_service(service)

def test_alert_box_budget_limits_preview_overlay_work():
    service = make_service(preview_cfg={"max_alert_boxes_per_stream": 2, "max_alert_labels_per_stream": 1})
    try:
        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        service._begin_alert_render_for_stream(0)

        service._draw_unified_alert_box(frame, [10, 10, 40, 40], "告警", "alert")
        service._draw_unified_alert_box(frame, [50, 10, 80, 40], "告警", "alert")
        service._draw_unified_alert_box(frame, [90, 10, 120, 40], "告警", "alert")

        assert service._alert_draw_counts[0] == 2
        assert service._alert_label_counts[0] == 1
    finally:
        shutdown_service(service)
