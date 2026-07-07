import queue
import threading
import time
from datetime import datetime

import numpy as np

from app.application.detector_registry import DetectorSpec
from app.pipeline.detector_worker import DetectorWorker
from app.pipeline.messages import AlgorithmResult, DetectionBox, InferenceResult


class DummyAlgorithm:
    def __init__(self, config, detector_name=None):
        self.detector_name = detector_name or "dummy"

    def build_disabled_result(self, frame_context, state):
        return AlgorithmResult(
            detector_type=self.detector_name,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            enabled=False,
            metrics={"disabled": True},
        )

    def process(self, frame_context, inference_outputs, state):
        return AlgorithmResult(
            detector_type=self.detector_name,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=[{"bbox": [1, 2, 3, 4], "score": 1.0}],
            display_alerts=[{"bbox": [1, 2, 3, 4]}],
            enabled=True,
            metrics={"inference_count": len(inference_outputs)},
        )


class FakeModelManager:
    def load_all(self, specs):
        self.specs = list(specs)
        return {}

    def infer(self, model_key, frame):
        return InferenceResult(
            model_key=model_key,
            detections=[DetectionBox(bbox=[1, 2, 3, 4], confidence=0.9, class_id=0)],
            timings={"total_ms": 0.5},
            raw_meta={},
        )

    def close(self):
        self.closed = True


def test_detector_worker_processes_frame_with_unified_loop():
    frame_queue = queue.Queue()
    result_queue = queue.Queue()
    control_queue = queue.Queue()
    frame_queue.put(
        {
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
            "frame_number": 1,
            "timestamp": datetime.now(),
            "stream_id": 0,
        }
    )
    spec = DetectorSpec(
        detector_name="dummy",
        model_keys=["dummy_model"],
        algorithm_cls=DummyAlgorithm,
        result_mapper=lambda result: result.detections,
        default_config_section="fight_detection",
    )
    worker = DetectorWorker(
        detector_spec=spec,
        frame_source=frame_queue,
        result_queue=result_queue,
        control_queue=control_queue,
        config={
            "models": {"dummy_model": "dummy.bmodel"},
            "model_types": {"dummy_model": "yolov8_fp16"},
            "bm1684x": {"device_id": 0},
            "fight_detection": {},
            "detection_timeshare": {"enabled": False},
        },
        model_manager=FakeModelManager(),
    )

    thread = threading.Thread(target=worker.start, daemon=True)
    thread.start()
    control_queue.put("enable")

    deadline = time.time() + 2.0
    payload = None
    while time.time() < deadline and payload is None:
        try:
            payload = result_queue.get(timeout=0.1)
        except queue.Empty:
            pass

    control_queue.put("stop")
    thread.join(timeout=2.0)

    assert payload is not None
    assert payload["detector_type"] == "dummy"
    assert payload["enabled"] is True
    assert payload["inference_count"] == 1


def test_detector_worker_model_specs_merge_advanced_algorithm_thresholds():
    spec = DetectorSpec(
        detector_name="dummy",
        model_keys=["dummy_model"],
        algorithm_cls=DummyAlgorithm,
        result_mapper=lambda result: result.detections,
        default_config_section="fight_detection",
    )
    worker = DetectorWorker(
        detector_spec=spec,
        frame_source=queue.Queue(),
        result_queue=queue.Queue(),
        control_queue=queue.Queue(),
        config={
            "models": {"dummy_model": "dummy.bmodel"},
            "model_types": {"dummy_model": "yolov8_fp16"},
            "bm1684x": {"device_id": 0},
            "fight_detection": {"conf_threshold": 0.5},
            "advanced_algorithm_params": {
                "fight_detection": {
                    "iou_threshold": 0.25,
                    "alert_hold_seconds": 0.5,
                }
            },
            "detection_timeshare": {"enabled": False},
        },
        model_manager=FakeModelManager(),
    )

    specs = worker._build_model_specs()

    assert specs[0].thresholds["conf_threshold"] == 0.5
    assert specs[0].thresholds["iou_threshold"] == 0.25
    assert specs[0].thresholds["alert_hold_seconds"] == 0.5
