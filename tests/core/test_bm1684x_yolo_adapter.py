import numpy as np
import pytest

from core import bm1684x_yolo26_adapter
from core import bm1684x_yolov8_adapter
from core.bm1684x_yolo_adapter import YOLOResult, create_yolo_detector, run_yolo_inference


class DummyYOLOv8Detector:
    def __init__(self, *args):
        self.args = args

    def print_model_info(self):
        return None


class DummyYOLO26Detector:
    def __init__(self, *args):
        self.args = args

    def print_model_info(self):
        return None


def _config_with_model_types(model_type_by_key=None):
    values = {
        "fall_detection": "yolov8_fp16",
        "ventilator_equipment": "yolov8_fp16",
        "ventilator_helmet": "yolov8_fp16",
        "fight_detection": "yolov8_fp16",
        "crowd_person": "yolov8_fp16",
        "helmet_detection": "yolov8_int8",
        "window_door_inside": "yolov8_fp16",
        "window_door_outside": "yolov8_fp16",
    }
    if model_type_by_key:
        values.update(model_type_by_key)

    return {
        "models": {
            "fall_detection": "fall.bmodel",
            "ventilator_equipment": "vent_equipment.bmodel",
            "ventilator_helmet": "vent_helmet.bmodel",
            "fight_detection": "fight.bmodel",
            "crowd_person": "crowd.bmodel",
            "helmet_detection": "helmet.bmodel",
            "window_door_inside": "window_in.bmodel",
            "window_door_outside": "window_out.bmodel",
        },
        "model_types": values,
    }


def test_create_yolo_detector_selects_yolov8_backend_from_explicit_model_type(monkeypatch):
    monkeypatch.setattr(bm1684x_yolov8_adapter, "BM1684X_YOLOv8", DummyYOLOv8Detector)

    detector = create_yolo_detector(
        "helmet_model.bmodel",
        device_id=1,
        conf_threshold=0.22,
        iou_threshold=0.44,
        model_type="yolov8_int8",
    )

    assert isinstance(detector, DummyYOLOv8Detector)
    assert detector.args == ("helmet_model.bmodel", 1, 0.22, 0.44)


def test_create_yolo_detector_selects_yolov8_backend_from_model_key_and_config(monkeypatch):
    monkeypatch.setattr(bm1684x_yolov8_adapter, "BM1684X_YOLOv8", DummyYOLOv8Detector)

    detector = create_yolo_detector(
        "fight_model.bmodel",
        device_id=3,
        conf_threshold=0.31,
        iou_threshold=0.52,
        model_key="fight_detection",
        config=_config_with_model_types(),
    )

    assert isinstance(detector, DummyYOLOv8Detector)
    assert detector.args == ("fight_model.bmodel", 3, 0.31, 0.52)


def test_create_yolo_detector_selects_yolo26_backend_from_explicit_model_type(monkeypatch):
    monkeypatch.setattr(bm1684x_yolo26_adapter, "BM1684X_YOLO26", DummyYOLO26Detector)

    detector = create_yolo_detector(
        "crowd_model.bmodel",
        device_id=2,
        conf_threshold=0.19,
        iou_threshold=0.41,
        model_type="yolo26_int8",
    )

    assert isinstance(detector, DummyYOLO26Detector)
    assert detector.args == ("crowd_model.bmodel", 2, 0.19, 0.41)


def test_create_yolo_detector_selects_yolo26_backend_from_model_key_and_config(monkeypatch):
    monkeypatch.setattr(bm1684x_yolo26_adapter, "BM1684X_YOLO26", DummyYOLO26Detector)

    detector = create_yolo_detector(
        "fall_model.bmodel",
        device_id=4,
        conf_threshold=0.27,
        iou_threshold=0.38,
        model_key="fall_detection",
        config=_config_with_model_types({"fall_detection": "yolo26_int8"}),
    )

    assert isinstance(detector, DummyYOLO26Detector)
    assert detector.args == ("fall_model.bmodel", 4, 0.27, 0.38)


def test_create_yolo_detector_requires_model_key_and_config_or_model_type_for_bmodel():
    with pytest.raises(ValueError, match="requires model_key \\+ config or an explicit model_type"):
        create_yolo_detector("model.bmodel")


@pytest.mark.parametrize(
    ("model_key", "model_type"),
    [
        ("fight_detection", "yolov5_fp16"),
        ("window_door_inside", "yolo26_fp16"),
    ],
)
def test_create_yolo_detector_rejects_unsupported_model_type_from_config(model_key, model_type):
    config = _config_with_model_types({model_key: model_type})

    with pytest.raises(NotImplementedError, match=model_key):
        create_yolo_detector(
            "unsupported_model.bmodel",
            model_key=model_key,
            config=config,
        )


def test_run_yolo_inference_consumes_yolov8_compatible_result():
    class DummyDetector:
        def __call__(self, frame):
            return [
                YOLOResult(
                    [
                        [1.0, 2.0, 11.0, 12.0, 0.91, 1.0],
                        [5.0, 6.0, 15.0, 16.0, 0.20, 0.0],
                    ],
                    frame.shape[:2],
                )
            ]

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    detections = run_yolo_inference(
        DummyDetector(),
        frame,
        conf_threshold=0.5,
        allowed_classes=[1],
    )

    assert detections == [
        {
            "bbox": [1, 2, 11, 12],
            "confidence": 0.91,
            "class_id": 1,
        }
    ]
