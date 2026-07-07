import json
from pathlib import Path

import cv2

from app.algorithms.base import algorithm_config
from app.algorithms.helmet import HelmetDetectionAlgorithm
from app.pipeline.messages import DetectionBox, FrameContext
from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference


ROOT = Path(__file__).resolve().parents[2]
_HELMET_DETECTOR = None


def _load_config():
    with (ROOT / "config_bm1684x.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_image(relative_path):
    image = cv2.imread(str(ROOT / relative_path))
    assert image is not None, "failed to read test image: {}".format(relative_path)
    return image


def _helmet_detector(config):
    global _HELMET_DETECTOR
    if _HELMET_DETECTOR is not None:
        return _HELMET_DETECTOR

    params = algorithm_config(config, "helmet_detection")
    _HELMET_DETECTOR = create_yolo_detector(
        config["models"]["helmet_detection"],
        device_id=int(config.get("bm1684x", {}).get("device_id", 0)),
        conf_threshold=float(params.get("conf_threshold", 0.25)),
        iou_threshold=float(params.get("iou_threshold", 0.45)),
        model_key="helmet_detection",
        config=config,
    )
    return _HELMET_DETECTOR


def _helmet_detections(detector, image):
    return [
        DetectionBox.from_mapping(item)
        for item in run_yolo_inference(detector, image, conf_threshold=None)
    ]


def test_helmet_bmodel_detects_no_helmet_sample_on_board():
    config = _load_config()
    image = _read_image("img.png")
    detector = _helmet_detector(config)

    detections = _helmet_detections(detector, image)

    assert detections
    assert any(item.class_id == 1 for item in detections)
    assert max(item.confidence for item in detections if item.class_id == 1) >= 0.8


def test_helmet_bmodel_detections_trigger_alarm_on_board():
    config = _load_config()
    image = _read_image("img.png")
    detector = _helmet_detector(config)
    detections = _helmet_detections(detector, image)
    assert any(item.class_id == 1 for item in detections)

    algorithm = HelmetDetectionAlgorithm(config)
    state = {}
    result = None
    for frame_number, timestamp in enumerate((0.0, 0.5, 1.0), start=1):
        result = algorithm.process(
            FrameContext(
                stream_id=1,
                frame=image,
                frame_number=frame_number,
                timestamp=timestamp,
            ),
            detections,
            state,
        )

    assert result is not None
    assert result.recordable_alerts
    assert result.detections == result.recordable_alerts
    assert result.metrics["alarm_number"] > 0


def test_helmet_bmodel_wearing_sample_does_not_trigger_no_helmet_alarm_on_board():
    config = _load_config()
    image = _read_image("1.jpg")
    detector = _helmet_detector(config)
    detections = _helmet_detections(detector, image)
    assert detections
    assert all(item.class_id != 1 for item in detections)

    algorithm = HelmetDetectionAlgorithm(config)
    state = {}
    for frame_number, timestamp in enumerate((0.0, 0.5, 1.0, 1.5), start=1):
        result = algorithm.process(
            FrameContext(
                stream_id=1,
                frame=image,
                frame_number=frame_number,
                timestamp=timestamp,
            ),
            detections,
            state,
        )
        assert result.recordable_alerts == []
        assert result.metrics["alarm_number"] == 0
