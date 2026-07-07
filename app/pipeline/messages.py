from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DetectionBox:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "DetectionBox":
        return cls(
            bbox=list(payload.get("bbox") or []),
            confidence=float(payload.get("confidence") or 0.0),
            class_id=int(payload.get("class_id") or 0),
            class_name=payload.get("class_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameContext:
    stream_id: int
    frame: Any
    frame_number: int
    timestamp: Any
    source_size: Optional[List[int]] = None
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "FrameContext":
        source_width = payload.get("original_width")
        source_height = payload.get("original_height")
        source_size = None
        if source_width is not None and source_height is not None:
            source_size = [int(source_width), int(source_height)]
        elif payload.get("shape") is not None:
            shape = payload.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                source_size = [int(shape[1]), int(shape[0])]
        return cls(
            stream_id=int(payload.get("stream_id", 0)),
            frame=payload.get("frame"),
            frame_number=int(payload.get("frame_number", 0)),
            timestamp=payload.get("timestamp"),
            source_size=source_size,
            raw_meta=dict(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "frame": self.frame,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "source_size": self.source_size,
            "raw_meta": dict(self.raw_meta),
        }


@dataclass
class InferenceResult:
    model_key: str
    detections: List[DetectionBox]
    timings: Dict[str, Any] = field(default_factory=dict)
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "detections": [item.to_dict() for item in self.detections],
            "timings": dict(self.timings),
            "raw_meta": dict(self.raw_meta),
        }


@dataclass
class AlgorithmResult:
    detector_type: str
    stream_id: int
    frame_number: int
    timestamp: Any
    detections: List[Dict[str, Any]] = field(default_factory=list)
    display_alerts: List[Dict[str, Any]] = field(default_factory=list)
    recordable_alerts: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "detector_type": self.detector_type,
            "stream_id": self.stream_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "detections": list(self.detections),
            "display_alerts": list(self.display_alerts),
            "recordable_alerts": list(self.recordable_alerts),
            "enabled": bool(self.enabled),
        }
        payload.update(self.metrics)
        payload.update(self.extras)
        return payload


@dataclass
class AlertEvent:
    alarm_id: str
    alarm_type: str
    alarm_info: Dict[str, Any]
    video_path: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class PreviewStatus:
    transport: str
    healthy: bool
    target_fps: int
    decoder_backend: Optional[str] = None
    encoder_backend: Optional[str] = None
    scale_backend: Optional[str] = None
    compose_fps: float = 0.0
    encode_in_fps: float = 0.0
    last_frame_ts: Optional[str] = None
    last_segment_ts: Optional[str] = None
    playlist_age_ms: Optional[float] = None
    unhealthy_reason: Optional[str] = None
