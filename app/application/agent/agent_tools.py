"""
AI Safety Agent — 工具函数集合。

每个工具函数接收系统数据源并返回结构化结果。
所有函数均只读取数据，不修改系统状态。
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, date, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

BEIJING_TZ = timezone(timedelta(hours=8))

# 报警类型中文映射
ALARM_TYPE_LABELS: Dict[str, str] = {
    "fall": "摔倒检测",
    "ventilator": "呼吸机检测",
    "fight": "打架检测",
    "crowd": "聚集检测",
    "helmet": "安全帽检测",
    "window_door_inside": "门窗（仓内）",
    "window_door_outside": "门窗（仓外）",
}

ALARM_TYPE_ICONS: Dict[str, str] = {
    "fall": "🤸",
    "helmet": "🧢",
    "fight": "🥊",
    "crowd": "👥",
    "ventilator": "😷",
    "window_door_inside": "🪟",
    "window_door_outside": "🚪",
}


def _now_beijing() -> datetime:
    return datetime.now(BEIJING_TZ)


def _today_str() -> str:
    return _now_beijing().strftime("%Y-%m-%d")


def _alarm_type_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    return ALARM_TYPE_LABELS.get(key, key or "未知类型")


def _alarm_type_icon(raw: str) -> str:
    key = (raw or "").strip().lower()
    return ALARM_TYPE_ICONS.get(key, "📹")


def mask_rtsp_url(url: str) -> str:
    """对 RTSP 地址中的密码进行脱敏。

    Examples:
        rtsp://admin:1q2w3e4r@192.168.150.65:554/Streaming/Channels/101
        → rtsp://admin:******@192.168.150.65:554/Streaming/Channels/101
    """
    if not url or not isinstance(url, str):
        return ""
    return re.sub(r"(:)([^:@]+)(@)", r"\1******\3", url)


def _extract_channel_from_name(name: str) -> Optional[int]:
    """从报警文件名中提取通道号。

    Examples:
        helmet_ch2_20250606_153000_frame.jpg → 2
        fall_20250606_153000.mp4 → None
    """
    if not name:
        return None
    m = re.search(r"_ch(\d+)_", name)
    if m:
        return int(m.group(1))
    return None


def _extract_alarm_type_from_name(name: str) -> Optional[str]:
    """从报警文件名中提取报警类型。"""
    if not name:
        return None
    for key in ALARM_TYPE_LABELS:
        if name.lower().startswith(key):
            return key
    return None


def _is_today(ts: float) -> bool:
    """判断时间戳是否为今天（北京时间）。"""
    try:
        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ)
        return dt.date() == _now_beijing().date()
    except Exception:
        return False


def _is_date(ts: float, date_str: str) -> bool:
    """判断时间戳是否为指定日期。"""
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ)
        return dt.date() == target
    except Exception:
        return False


# ── 工具函数 ──────────────────────────────────────────────


def get_system_status_tool(
    status_handler: Callable[[], Dict[str, Any]],
    alert_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """查询系统运行状态。

    Args:
        status_handler: 系统状态获取函数（handlers["status"]）
        alert_items: 报警列表（可选，用于提取最近报警）

    Returns:
        {
            "system_running": bool,
            "detectors": {...},
            "preview": {...},
            "stream_count": int,
            "active_detector_count": int,
            "active_task_count": int,
            "recent_alarm": dict | None,
            "summary": str,
        }
    """
    try:
        status = status_handler()
    except Exception:
        return {
            "status": "error",
            "message": "无法获取系统状态",
            "system_running": False,
            "detectors": {},
            "preview": {},
            "stream_count": 0,
            "active_detector_count": 0,
            "active_task_count": 0,
            "recent_alarm": None,
            "summary": "系统状态获取失败",
        }

    detectors = status.get("detectors", {})
    active_detectors = {k: v for k, v in detectors.items() if v.get("running")}

    # 最近报警
    recent_alarm = None
    if alert_items:
        sorted_items = sorted(
            alert_items,
            key=lambda x: x.get("mtime", x.get("timestamp", 0)),
            reverse=True,
        )
        if sorted_items:
            recent = sorted_items[0]
            recent_alarm = {
                "type": _alarm_type_label(
                    _extract_alarm_type_from_name(recent.get("name", "")) or ""
                ),
                "name": recent.get("name", ""),
                "time": recent.get("time_display", ""),
                "size": recent.get("size_display", ""),
            }

    stream_count = len(status.get("streams", []))

    summary_parts = []
    if status.get("system_running"):
        summary_parts.append("系统正在运行")
    else:
        summary_parts.append("系统已停止")
    summary_parts.append(f"共 {stream_count} 路视频通道")
    summary_parts.append(f"{len(active_detectors)} 个检测器已启用")

    return {
        "status": "success",
        "system_running": bool(status.get("system_running")),
        "detectors": {
            k: {"running": bool(v.get("running")), "pid": v.get("pid")}
            for k, v in detectors.items()
        },
        "preview": status.get("preview", {}),
        "stream_count": stream_count,
        "active_detector_count": len(active_detectors),
        "active_task_count": 1 if active_detectors else 0,
        "recent_alarm": recent_alarm,
        "summary": "；".join(summary_parts),
    }


def get_alert_history_tool(
    alerts_handler: Callable[[], Dict[str, Any]],
    date: Optional[str] = None,
    alarm_type: Optional[str] = None,
    channel: Optional[int] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """查询报警事件历史。

    Args:
        alerts_handler: 报警历史获取函数（handlers["alerts_history"]）
        date: 日期筛选（"YYYY-MM-DD"）
        alarm_type: 报警类型筛选（英文key）
        channel: 通道筛选（1-4）
        limit: 最大返回数量

    Returns:
        {
            "total": int,
            "filtered": int,
            "items": [...],
            "date": str | None,
            "alarm_type": str | None,
            "channel": int | None,
        }
    """
    try:
        result = alerts_handler()
    except Exception:
        return {"status": "error", "message": "无法获取报警历史", "total": 0, "filtered": 0, "items": []}

    items = result.get("items", [])
    output_dir = result.get("output_dir", "")

    # 筛选
    filtered = []
    for item in items:
        name = item.get("name", "")
        mtime = item.get("mtime", 0)

        if date and not _is_date(mtime, date):
            continue
        if alarm_type:
            item_type = _extract_alarm_type_from_name(name)
            if item_type != alarm_type:
                continue
        if channel is not None:
            item_ch = _extract_channel_from_name(name)
            if item_ch != channel:
                continue
        filtered.append(item)

    # 补充字段
    for item in filtered:
        item["alarm_type"] = _extract_alarm_type_from_name(item.get("name", ""))
        item["alarm_type_label"] = _alarm_type_label(item.get("alarm_type", ""))
        item["channel"] = _extract_channel_from_name(item.get("name", ""))

    # 排序（最新在前）并截断
    filtered.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    filtered = filtered[:max(1, int(limit))]

    return {
        "status": "success",
        "total": len(items),
        "filtered": len(filtered),
        "items": filtered,
        "date": date,
        "alarm_type": alarm_type,
        "channel": channel,
        "output_dir": output_dir,
    }


def get_alert_statistics_tool(
    alerts_handler: Callable[[], Dict[str, Any]],
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """统计报警数据。

    Args:
        alerts_handler: 报警历史获取函数
        date: 统计日期（默认今天）

    Returns:
        {
            "date": str,
            "total": int,
            "by_type": {...},
            "by_channel": {...},
            "recent_alarms": [...],
            "top_type": str | None,
            "top_channel": int | None,
        }
    """
    target_date = date or _today_str()

    try:
        result = alerts_handler()
    except Exception:
        return {
            "status": "error",
            "message": "无法获取报警数据",
            "date": target_date,
            "total": 0,
            "by_type": {},
            "by_channel": {},
            "recent_alarms": [],
            "top_type": None,
            "top_channel": None,
        }

    items = result.get("items", [])

    # 按日期筛选
    today_items = [item for item in items if _is_date(item.get("mtime", 0), target_date)]

    by_type: Dict[str, int] = Counter()
    by_channel: Dict[int, int] = Counter()
    enriched: List[Dict[str, Any]] = []

    for item in today_items:
        name = item.get("name", "")
        a_type = _extract_alarm_type_from_name(name)
        ch = _extract_channel_from_name(name)

        by_type[_alarm_type_label(a_type)] += 1
        if ch is not None:
            by_channel[ch] += 1

        enriched.append({
            "name": item.get("name", ""),
            "type": a_type,
            "type_label": _alarm_type_label(a_type),
            "channel": ch,
            "mtime": item.get("mtime", 0),
            "time_display": item.get("time_display", ""),
            "size_display": item.get("size_display", ""),
        })

    # 最近5条
    enriched.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    recent = enriched[:5]

    top_type = by_type.most_common(1)[0][0] if by_type else None
    top_channel = by_channel.most_common(1)[0][0] if by_channel else None

    return {
        "status": "success",
        "date": target_date,
        "total": len(today_items),
        "by_type": dict(by_type),
        "by_channel": {str(k): v for k, v in by_channel.items()},
        "recent_alarms": recent,
        "top_type": top_type,
        "top_channel": top_channel,
    }


def get_tasks_tool(
    tasks_handler: Callable[[], Dict[str, Any]],
    scheduler_getter: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """查询当前任务配置。

    Args:
        tasks_handler: 任务列表获取函数（handlers["tasks_get"]）
        scheduler_getter: 调度器获取函数

    Returns:
        {
            "task_count": int,
            "tasks": [...],
        }
    """
    try:
        result = tasks_handler()
    except Exception:
        return {"status": "error", "message": "无法获取任务配置", "task_count": 0, "tasks": []}

    tasks = result.get("tasks", [])
    enriched_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        detectors = task.get("detectors", [])
        detector_labels = [_alarm_type_label(d) for d in detectors]
        enriched_tasks.append({
            "id": task.get("id", ""),
            "name": task.get("name", ""),
            "stream_index": task.get("stream_index", 1),
            "stream_label": f"通道{task.get('stream_index', 1)}",
            "detectors": detectors,
            "detector_labels": detector_labels,
            "running": bool(task.get("running")),
        })

    return {
        "status": "success",
        "task_count": len(enriched_tasks),
        "tasks": enriched_tasks,
    }


def get_video_streams_tool(
    streams_handler: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """查询视频流配置。

    Args:
        streams_handler: 视频流配置获取函数（handlers["video_streams_get"]）

    Returns:
        {
            "stream_count": int,
            "streams": [...],
            "connected_count": int,
        }
    """
    try:
        result = streams_handler()
    except Exception:
        return {"status": "error", "message": "无法获取视频流配置", "stream_count": 0, "streams": []}

    raw_streams = result.get("streams", [])
    max_streams = result.get("max_streams", 4)

    streams = []
    connected_count = 0
    for idx, s in enumerate(raw_streams):
        source = (s.get("source") or s.get("ip") or "").strip()
        is_connected = bool(source)
        if is_connected:
            connected_count += 1
        streams.append({
            "index": idx + 1,
            "name": s.get("name", f"通道{idx + 1}"),
            "source_type": s.get("source_type", "rtsp"),
            "source_type_label": "RTSP" if s.get("source_type") != "file" else "本地文件",
            "source": mask_rtsp_url(source),
            "connected": is_connected,
        })

    return {
        "status": "success",
        "stream_count": len(streams),
        "max_streams": max_streams,
        "streams": streams,
        "connected_count": connected_count,
    }


def generate_daily_report_tool(
    status_handler: Callable[[], Dict[str, Any]],
    alerts_handler: Callable[[], Dict[str, Any]],
    tasks_handler: Callable[[], Dict[str, Any]],
    streams_handler: Callable[[], Dict[str, Any]],
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """生成粮库安全巡检日报。

    整合系统状态、报警统计、任务和视频流信息。

    Args:
        status_handler: 系统状态获取函数
        alerts_handler: 报警历史获取函数
        tasks_handler: 任务列表获取函数
        streams_handler: 视频流配置获取函数
        date: 报告日期

    Returns:
        {
            "report_date": str,
            "system_status": {...},
            "alert_statistics": {...},
            "tasks": {...},
            "streams": {...},
            "risk_level": str,  # low/medium/high
        }
    """
    target_date = date or _today_str()

    system_status = get_system_status_tool(status_handler)
    alert_stats = get_alert_statistics_tool(alerts_handler, date=target_date)
    tasks = get_tasks_tool(tasks_handler)
    streams = get_video_streams_tool(streams_handler)

    # 风险评估
    total_alarms = alert_stats.get("total", 0)
    if total_alarms > 10:
        risk_level = "high"
    elif total_alarms > 5:
        risk_level = "medium"
    else:
        risk_level = "low"

    recent_items = alert_stats.get("recent_alarms", [])
    recent_list = []
    for item in recent_items[:5]:
        recent_list.append({
            "type_label": item.get("type_label", ""),
            "channel": item.get("channel"),
            "time": item.get("time_display", ""),
        })

    return {
        "status": "success",
        "report_date": target_date,
        "report_time": _now_beijing().strftime("%Y年%m月%d日 %H:%M"),
        "system_status": system_status,
        "alert_statistics": {
            "total": total_alarms,
            "by_type": alert_stats.get("by_type", {}),
            "by_channel": alert_stats.get("by_channel", {}),
            "top_type": alert_stats.get("top_type"),
            "top_channel": alert_stats.get("top_channel"),
            "recent_list": recent_list,
        },
        "tasks": {
            "task_count": tasks.get("task_count", 0),
            "tasks": tasks.get("tasks", []),
        },
        "streams": {
            "total": streams.get("stream_count", 0),
            "connected": streams.get("connected_count", 0),
        },
        "risk_level": risk_level,
    }
