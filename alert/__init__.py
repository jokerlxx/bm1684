"""
告警管理器（Alert Manager）：根据预设规则判断检测分析结果并触发告警。
规则与触发逻辑当前在 core.display_service 中实现（各检测器阈值、持续时间等），
触发后调用存储层保存告警图片/视频，并可向 Web 层推送告警事件。
"""

# 告警类型与展示名称映射，供前端与历史回溯使用
ALARM_TYPE_LABELS = {
    "fall": "跌倒检测",
    "ventilator": "呼吸机检测",
    "fight": "打架检测",
    "crowd": "聚集检测",
    "helmet": "安全帽检测",
    "window_door": "窗户门检测",
}

__all__ = ["ALARM_TYPE_LABELS"]
