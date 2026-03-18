"""
核心模块：事件类型、展示服务、BM1684X 适配器。
事件驱动微服务架构下，各模块通过队列传递事件（FrameEvent / DetectionResultEvent / AlertEvent）。
"""

from core.events import (
    EVENT_FRAME,
    EVENT_DETECTION_RESULT,
    EVENT_ALERT,
    frame_event,
    detection_result_event,
    alert_event,
)
from core.display_service import run_display_service

__all__ = [
    "EVENT_FRAME",
    "EVENT_DETECTION_RESULT",
    "EVENT_ALERT",
    "frame_event",
    "detection_result_event",
    "alert_event",
    "run_display_service",
]
