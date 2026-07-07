from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class DetectorSpec:
    detector_name: str
    model_keys: List[str]
    algorithm_cls: object
    result_mapper: Callable
    default_config_section: str


def _default_result_mapper(inference_result):
    return inference_result.detections


DETECTOR_SPECS: Dict[str, DetectorSpec] = {
    "fall": DetectorSpec(
        detector_name="fall",
        model_keys=["fall_detection"],
        algorithm_cls="app.algorithms.fall:FallDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="fall_detection",
    ),
    "ventilator": DetectorSpec(
        detector_name="ventilator",
        model_keys=["ventilator_equipment", "ventilator_helmet"],
        algorithm_cls="app.algorithms.ventilator:VentilatorDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="ventilator_detection",
    ),
    "fight": DetectorSpec(
        detector_name="fight",
        model_keys=["fight_detection"],
        algorithm_cls="app.algorithms.fight:FightDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="fight_detection",
    ),
    "crowd": DetectorSpec(
        detector_name="crowd",
        model_keys=["crowd_person"],
        algorithm_cls="app.algorithms.crowd:CrowdDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="crowd_detection",
    ),
    "helmet": DetectorSpec(
        detector_name="helmet",
        model_keys=["helmet_detection"],
        algorithm_cls="app.algorithms.helmet:HelmetDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="helmet_detection",
    ),
    "window_door_inside": DetectorSpec(
        detector_name="window_door_inside",
        model_keys=["window_door_inside"],
        algorithm_cls="app.algorithms.window_door:WindowDoorDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="window_door_detection",
    ),
    "window_door_outside": DetectorSpec(
        detector_name="window_door_outside",
        model_keys=["window_door_outside"],
        algorithm_cls="app.algorithms.window_door:WindowDoorDetectionAlgorithm",
        result_mapper=_default_result_mapper,
        default_config_section="window_door_detection",
    ),
}

VALID_DETECTORS = list(DETECTOR_SPECS.keys())


class DetectorRegistry:
    def __init__(self, specs=None):
        self.specs = dict(specs or DETECTOR_SPECS)

    def get(self, detector_name: str) -> DetectorSpec:
        return self.specs[detector_name]

    def all(self) -> Dict[str, DetectorSpec]:
        return dict(self.specs)
