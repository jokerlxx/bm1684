import pytest

from app.model_runtime.model_type_registry import (
    ensure_supported_model_type,
    parse_model_type,
    resolve_model_type,
    validate_model_types_config,
)


def _base_config():
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
            "_comment_models": "ignored",
        },
        "model_types": {
            "fall_detection": "yolov8_fp16",
            "ventilator_equipment": "yolov8_fp16",
            "ventilator_helmet": "yolov8_fp16",
            "fight_detection": "yolov8_fp16",
            "crowd_person": "yolov8_fp16",
            "helmet_detection": "yolov8_int8",
            "window_door_inside": "yolov8_fp16",
            "window_door_outside": "yolov8_fp16",
            "_comment_types": "ignored",
        },
    }


def test_validate_model_types_config_accepts_matching_keys():
    validate_model_types_config(_base_config())


def test_validate_model_types_config_rejects_missing_key():
    config = _base_config()
    del config["model_types"]["fight_detection"]

    with pytest.raises(ValueError, match="missing keys"):
        validate_model_types_config(config)


def test_validate_model_types_config_rejects_extra_key():
    config = _base_config()
    config["model_types"]["extra_model"] = "yolov8_fp16"

    with pytest.raises(ValueError, match="unknown keys"):
        validate_model_types_config(config)


def test_validate_model_types_config_rejects_invalid_value():
    config = _base_config()
    config["model_types"]["helmet_detection"] = "legacy"

    with pytest.raises(ValueError, match="family_precision"):
        validate_model_types_config(config)


def test_resolve_model_type_parses_family_and_precision():
    spec = resolve_model_type("helmet_detection", _base_config())

    assert spec.model_key == "helmet_detection"
    assert spec.raw == "yolov8_int8"
    assert spec.family == "yolov8"
    assert spec.precision == "int8"


def test_parse_and_support_checks_accept_yolo26_int8():
    spec = parse_model_type("crowd_person", "yolo26_int8")

    assert spec.family == "yolo26"
    assert spec.precision == "int8"

    assert ensure_supported_model_type(spec) == spec


def test_parse_and_support_checks_allow_future_family_but_reject_runtime_support():
    spec = parse_model_type("fight_detection", "yolov5_fp16")

    assert spec.family == "yolov5"
    assert spec.precision == "fp16"

    with pytest.raises(NotImplementedError, match="Supported values in this build"):
        ensure_supported_model_type(spec)
