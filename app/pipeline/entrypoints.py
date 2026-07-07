from __future__ import annotations

from app.application.detector_registry import DetectorRegistry
from app.pipeline.detector_worker import DetectorWorker


_REGISTRY = DetectorRegistry()


def run_detector_process(detector_name, frame_queue, result_queue, control_queue, config, model_overrides=None):
    worker = DetectorWorker(
        detector_spec=_REGISTRY.get(detector_name),
        frame_source=frame_queue,
        result_queue=result_queue,
        control_queue=control_queue,
        config=config,
        model_overrides=model_overrides,
        registry=_REGISTRY,
    )
    worker.start()


def run_fall_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "fall",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"fall_detection": model_path},
    )


def run_crowd_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "crowd",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"crowd_person": model_path},
    )


def run_fight_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "fight",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"fight_detection": model_path},
    )


def run_helmet_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "helmet",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"helmet_detection": model_path},
    )


def run_window_door_inside_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "window_door_inside",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"window_door_inside": model_path},
    )


def run_window_door_outside_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "window_door_outside",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={"window_door_outside": model_path},
    )


def run_window_door_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_window_door_inside_detector(model_path, frame_queue, result_queue, control_queue, config)


def run_ventilator_detector(equipment_model_path, helmet_model_path, frame_queue, result_queue, control_queue, config):
    return run_detector_process(
        "ventilator",
        frame_queue,
        result_queue,
        control_queue,
        config,
        model_overrides={
            "ventilator_equipment": equipment_model_path,
            "ventilator_helmet": helmet_model_path,
        },
    )
