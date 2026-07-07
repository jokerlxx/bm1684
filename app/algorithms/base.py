from __future__ import annotations

from typing import Any, Dict

from app.pipeline.messages import AlgorithmResult, FrameContext


def algorithm_config(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    base = dict((config or {}).get(section) or {})
    advanced = ((config or {}).get("advanced_algorithm_params") or {}).get(section) or {}
    base.update(advanced)
    return base


class DetectionAlgorithm:
    detector_type = "unknown"

    def __init__(self, config: Dict[str, Any], detector_name: str | None = None):
        self.config = config or {}
        if detector_name:
            self.detector_type = detector_name

    def build_disabled_result(self, frame_context: FrameContext, state: Dict[str, Any]) -> AlgorithmResult:
        return AlgorithmResult(
            detector_type=self.detector_type,
            stream_id=frame_context.stream_id,
            frame_number=frame_context.frame_number,
            timestamp=frame_context.timestamp,
            detections=[],
            display_alerts=[],
            recordable_alerts=[],
            enabled=False,
            metrics={},
        )

    def process(self, frame_context: FrameContext, inference_outputs: Any, state: Dict[str, Any]) -> AlgorithmResult:
        raise NotImplementedError
