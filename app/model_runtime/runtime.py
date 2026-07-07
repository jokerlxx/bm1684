from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference

from app.pipeline.messages import DetectionBox, InferenceResult
from core.logging_utils import log_model_loaded


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    model_path: str
    model_type: Optional[str]
    device_id: int = 0
    thresholds: Dict[str, Any] = field(default_factory=dict)


class InferenceRuntime(ABC):
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame: Any) -> InferenceResult:
        raise NotImplementedError

    def infer_batch(self, frames: Iterable[Any]) -> list[InferenceResult]:
        return [self.infer(frame) for frame in frames]

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class StandardizedYoloRuntime(InferenceRuntime):
    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec)
        self._detector = None

    def load(self) -> None:
        conf_threshold = self.model_spec.thresholds.get("conf_threshold", 0.25)
        iou_threshold = self.model_spec.thresholds.get("iou_threshold", 0.45)
        self._detector = create_yolo_detector(
            self.model_spec.model_path,
            device_id=self.model_spec.device_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_key=self.model_spec.model_key,
            config={
                "models": {self.model_spec.model_key: self.model_spec.model_path},
                "model_types": {self.model_spec.model_key: self.model_spec.model_type},
            } if self.model_spec.model_type else None,
            model_type=self.model_spec.model_type,
        )

    def infer(self, frame: Any) -> InferenceResult:
        detections = []
        for item in run_yolo_inference(self._detector, frame, conf_threshold=None, allowed_classes=None):
            detections.append(
                DetectionBox(
                    bbox=list(item.get("bbox") or []),
                    confidence=float(item.get("confidence") or 0.0),
                    class_id=int(item.get("class_id") or 0),
                    class_name=item.get("class_name"),
                )
            )
        timings = {}
        getter = getattr(self._detector, "get_last_timing", None)
        if callable(getter):
            try:
                timings = dict(getter() or {})
            except Exception:
                timings = {}
        return InferenceResult(
            model_key=self.model_spec.model_key,
            detections=detections,
            timings=timings,
            raw_meta={"model_path": self.model_spec.model_path},
        )

    def _boxes_to_detections(self, result: Any) -> list[DetectionBox]:
        if not hasattr(result, "boxes") or result.boxes is None:
            return []
        boxes_data = result.boxes.data
        if hasattr(boxes_data, "cpu"):
            boxes_data = boxes_data.cpu().numpy()
        elif hasattr(boxes_data, "numpy"):
            boxes_data = boxes_data.numpy()
        detections = []
        for det in boxes_data:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            detections.append(
                DetectionBox(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=float(conf),
                    class_id=int(cls_id),
                )
            )
        return detections

    def _batch_size(self) -> int:
        return max(1, int(getattr(self._detector, "batch_size", 1) or 1))

    def infer_batch(self, frames: Iterable[Any]) -> list[InferenceResult]:
        frame_list = list(frames)
        if not frame_list:
            return []
        results: list[InferenceResult] = []
        batch_size = self._batch_size()
        for start in range(0, len(frame_list), batch_size):
            chunk = frame_list[start : start + batch_size]
            detector_results = self._detector(chunk) if len(chunk) > 1 else self._detector(chunk[0])
            detector_results = list(detector_results or [])
            timings = {}
            getter = getattr(self._detector, "get_last_timing", None)
            if callable(getter):
                try:
                    timings = dict(getter() or {})
                except Exception:
                    timings = {}
            for result in detector_results[: len(chunk)]:
                results.append(
                    InferenceResult(
                        model_key=self.model_spec.model_key,
                        detections=self._boxes_to_detections(result),
                        timings=timings,
                        raw_meta={"model_path": self.model_spec.model_path, "batch_size": len(chunk)},
                    )
                )
        return results

    def close(self) -> None:
        self._detector = None


class ModelManager:
    def __init__(self, runtime_cls=StandardizedYoloRuntime):
        self.runtime_cls = runtime_cls
        self._runtimes: Dict[str, InferenceRuntime] = {}

    def load(self, model_spec: ModelSpec) -> InferenceRuntime:
        runtime = self.runtime_cls(model_spec)
        runtime.load()
        log_model_loaded(model_spec.model_key, model_spec.model_path)
        self._runtimes[model_spec.model_key] = runtime
        return runtime

    def load_all(self, model_specs: Iterable[ModelSpec]) -> Dict[str, InferenceRuntime]:
        loaded = {}
        for spec in model_specs:
            loaded[spec.model_key] = self.load(spec)
        return loaded

    def infer(self, model_key: str, frame: Any) -> InferenceResult:
        return self._runtimes[model_key].infer(frame)

    def infer_batch(self, model_key: str, frames: Iterable[Any]) -> list[InferenceResult]:
        return self._runtimes[model_key].infer_batch(frames)

    def get_max_batch(self, model_key: str) -> int:
        runtime = self._runtimes.get(model_key)
        if runtime is None:
            return 1
        getter = getattr(runtime, "_batch_size", None)
        if callable(getter):
            return max(1, int(getter()))
        batch_size = getattr(runtime, "_detector", None)
        if batch_size is not None:
            return max(1, int(getattr(batch_size, "batch_size", 1) or 1))
        return 1

    def close(self) -> None:
        for runtime in self._runtimes.values():
            try:
                runtime.close()
            except Exception:
                pass
        self._runtimes.clear()
