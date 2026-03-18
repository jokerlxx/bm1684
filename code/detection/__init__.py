"""
检测引擎（Detection Engine）+ 插件化 AI 模型池（Model Pool）。
从帧队列取帧，经插件化模型并行推理，将分析结果写入各检测器结果队列。
"""

from detection.fall_detector import run_fall_detector
from detection.ventilator_detector import run_ventilator_detector
from detection.fight_detector import run_fight_detector
from detection.crowd_detector import run_crowd_detector
from detection.helmet_detector import run_helmet_detector
from detection.window_door_detector import (
    run_window_door_detector,
    run_window_door_inside_detector,
    run_window_door_outside_detector,
)

MODEL_POOL = {
    "fall": run_fall_detector,
    "ventilator": run_ventilator_detector,
    "fight": run_fight_detector,
    "crowd": run_crowd_detector,
    "helmet": run_helmet_detector,
    "window_door_inside": run_window_door_inside_detector,
    "window_door_outside": run_window_door_outside_detector,
}

__all__ = [
    "MODEL_POOL",
    "run_fall_detector",
    "run_ventilator_detector",
    "run_fight_detector",
    "run_crowd_detector",
    "run_helmet_detector",
    "run_window_door_detector",
    "run_window_door_inside_detector",
    "run_window_door_outside_detector",
]
