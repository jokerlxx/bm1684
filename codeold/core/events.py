"""
事件类型定义：各模块通过队列或消息中间件传递结构化事件，实现解耦。
"""

# 事件类型常量
EVENT_FRAME = "frame"
EVENT_DETECTION_RESULT = "detection_result"
EVENT_ALERT = "alert"


def frame_event(stream_id, frame, timestamp, frame_number=None):
    """视频流接入层产生的帧事件（当前实现中 frame_data 即为此结构）。"""
    return {
        "type": EVENT_FRAME,
        "stream_id": stream_id,
        "frame": frame,
        "timestamp": timestamp,
        "frame_number": frame_number,
    }


def detection_result_event(detector_type, result, stream_id=None):
    """检测引擎产生的分析结果事件（当前 result 字典即为此结构）。"""
    return {
        "type": EVENT_DETECTION_RESULT,
        "detector_type": detector_type,
        "result": result,
        "stream_id": stream_id,
    }


def alert_event(alarm_id, alarm_type, alarm_info, video_path=None, image_path=None, timestamp=None):
    """告警管理器/存储层产生的告警事件，用于 Web 实时推送与历史回溯。"""
    import datetime as dt
    return {
        "type": EVENT_ALERT,
        "alarm_id": alarm_id,
        "alarm_type": alarm_type,
        "alarm_info": alarm_info,
        "video_path": video_path,
        "image_path": image_path,
        "timestamp": timestamp or dt.datetime.now().isoformat(),
    }
