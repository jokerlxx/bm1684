from app.model_runtime.runtime import InferenceRuntime, ModelManager, ModelSpec, StandardizedYoloRuntime
from app.pipeline.messages import DetectionBox, InferenceResult


class FakeRuntime(InferenceRuntime):
    def __init__(self, model_spec):
        super().__init__(model_spec)
        self.loaded = False
        self.closed = False

    def load(self):
        self.loaded = True

    def infer(self, frame):
        return InferenceResult(
            model_key=self.model_spec.model_key,
            detections=[DetectionBox(bbox=[1, 2, 3, 4], confidence=0.9, class_id=0, class_name="x")],
            timings={"total_ms": 1.2},
            raw_meta={"frame": frame},
        )

    def close(self):
        self.closed = True


def test_model_manager_loads_infers_and_closes():
    manager = ModelManager(runtime_cls=FakeRuntime)
    spec = ModelSpec(model_key="fight_detection", model_path="fight.bmodel", model_type="yolov8_fp16")

    runtime = manager.load(spec)
    result = manager.infer("fight_detection", frame="frame-1")
    manager.close()

    assert runtime.loaded is True
    assert result.model_key == "fight_detection"
    assert result.detections[0].bbox == [1, 2, 3, 4]
    assert runtime.closed is True


def test_standardized_yolo_runtime_normalizes_detector_output(monkeypatch):
    class FakeDetector:
        def get_last_timing(self):
            return {"total_ms": 3.5}

    monkeypatch.setattr("app.model_runtime.runtime.create_yolo_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(
        "app.model_runtime.runtime.run_yolo_inference",
        lambda detector, frame, conf_threshold=None, allowed_classes=None: [
            {"bbox": [10, 11, 12, 13], "confidence": 0.77, "class_id": 1}
        ],
    )

    runtime = StandardizedYoloRuntime(
        ModelSpec(
            model_key="helmet_detection",
            model_path="helmet.bmodel",
            model_type="yolov8_int8",
            device_id=0,
            thresholds={"conf_threshold": 0.4},
        )
    )
    runtime.load()
    result = runtime.infer(frame="frame-2")

    assert result.model_key == "helmet_detection"
    assert result.timings["total_ms"] == 3.5
    assert result.detections[0].bbox == [10, 11, 12, 13]
    assert result.detections[0].confidence == 0.77

