"""
AI Safety Agent — 本地规则模板。

当 DeepSeek API 未配置或调用失败时，使用这些模板生成回答。
模板仅基于真实系统数据填充，不编造任何信息。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── System Status ─────────────────────────────────────────


def build_system_status_answer(data: Dict[str, Any]) -> str:
    """生成系统状态回答。"""
    if data.get("status") == "error":
        return "抱歉，当前无法获取系统状态信息，请稍后重试或检查系统是否已启动。"

    running = data.get("system_running", False)
    stream_count = data.get("stream_count", 0)
    active_detectors = data.get("active_detector_count", 0)
    detectors = data.get("detectors", {})
    recent_alarm = data.get("recent_alarm")
    preview = data.get("preview", {})

    parts = []

    if running:
        parts.append("当前系统处于**运行状态**。")
    else:
        parts.append("当前系统**已停止**，若需启用监控请手动启动系统。")

    parts.append(f"共配置 {stream_count} 路视频通道。")

    # 检测器详情
    detector_labels = {
        "fall": "摔倒检测",
        "ventilator": "呼吸机检测",
        "fight": "打架检测",
        "crowd": "聚集检测",
        "helmet": "安全帽检测",
        "window_door_inside": "门窗（仓内）",
        "window_door_outside": "门窗（仓外）",
    }
    active_list = []
    for name, info in detectors.items():
        if info.get("running"):
            label = detector_labels.get(name, name)
            active_list.append(label)

    if active_list:
        parts.append(f"已启用的检测器：{'、'.join(active_list)}。")
    else:
        parts.append("当前没有检测器处于启用状态。")

    # 预览状态
    preview_healthy = preview.get("healthy")
    if preview_healthy is False and running:
        reason = preview.get("unhealthy_reason", "未知")
        parts.append(f"⚠️ 预览服务状态异常：{reason}")
    elif running:
        parts.append("预览服务运行正常。")

    # 最近报警
    if recent_alarm:
        parts.append(
            f"最近一次告警为{recent_alarm.get('type', '未知类型')}"
            f"（{recent_alarm.get('time', '')}）。"
        )
    elif running:
        parts.append("近期暂无告警记录，系统处于安全状态。")

    if not running:
        parts.append("请点击启动按钮开启安全监控服务。若系统持续无法启动，建议检查摄像头连接和配置文件。")

    return "\n\n".join(parts)


# ── Alert History ─────────────────────────────────────────


def build_alert_history_answer(data: Dict[str, Any]) -> str:
    """生成报警历史回答。"""
    if data.get("status") == "error":
        return "抱歉，当前无法获取报警历史信息，请稍后重试。"

    total = data.get("total", 0)
    filtered = data.get("filtered", 0)
    items = data.get("items", [])
    date = data.get("date")
    alarm_type = data.get("alarm_type")
    channel = data.get("channel")

    parts = []

    # 描述筛选条件
    conditions = []
    if date:
        conditions.append(f"{date}")
    if alarm_type:
        from app.application.agent.agent_tools import _alarm_type_label
        conditions.append(_alarm_type_label(alarm_type))
    if channel is not None:
        conditions.append(f"通道{channel}")

    cond_str = "、".join(conditions) if conditions else "全部历史"

    if filtered == 0:
        parts.append(f"查询条件「{cond_str}」下暂无报警记录。")
        return "\n\n".join(parts)

    parts.append(
        f"查询条件「{cond_str}」下共 {filtered} 条报警记录"
        + (f"（全部历史共 {total} 条）" if total != filtered else "")
        + "："
    )

    for i, item in enumerate(items[:20], 1):
        type_label = item.get("alarm_type_label", "未知")
        ch = item.get("channel")
        ch_str = f" 通道{ch}" if ch else ""
        time_str = item.get("time_display", "")
        size_str = item.get("size_display", "")
        parts.append(f"{i}. {type_label}{ch_str} — {time_str}（{size_str}）")

    if filtered > 20:
        parts.append(f"……还有 {filtered - 20} 条记录，请在报警事件页面查看完整列表。")

    return "\n\n".join(parts)


# ── Alert Statistics ──────────────────────────────────────


def build_alert_statistics_answer(data: Dict[str, Any]) -> str:
    """生成报警统计回答。"""
    if data.get("status") == "error":
        return "抱歉，当前无法统计报警数据，请稍后重试。"

    date = data.get("date", "今天")
    total = data.get("total", 0)
    by_type = data.get("by_type", {})
    by_channel = data.get("by_channel", {})
    top_type = data.get("top_type")
    top_channel = data.get("top_channel")
    recent = data.get("recent_alarms", [])

    parts = []

    if total == 0:
        parts.append(f"{date}（{_date_label(date)}）系统暂无报警记录，各区域作业情况正常。")
        return "\n\n".join(parts)

    parts.append(f"{date}（{_date_label(date)}）系统共记录 **{total}** 条报警事件。")

    # 按类型统计
    if by_type:
        parts.append("**按报警类型统计：**")
        for type_name, count in sorted(by_type.items(), key=lambda x: -x[1]):
            parts.append(f"- {type_name}：{count} 条")

    # 按通道统计
    if by_channel:
        parts.append("**按通道统计：**")
        for ch, count in sorted(by_channel.items(), key=lambda x: int(x[0])):
            parts.append(f"- 通道{ch}：{count} 条")

    # 风险分析
    if total > 10:
        parts.append(
            "🔴 **风险提示：** 今日报警数量较多（超过10条），建议立即对高频报警区域进行现场复核，"
            "检查作业人员防护装备佩戴情况，必要时暂停相关区域作业并开展安全培训。"
        )
    elif total > 5:
        parts.append(
            "🟡 **提醒：** 今日报警数量处于中等水平，建议关注高频报警类型和通道，"
            "对相关区域加强巡查。"
        )
    else:
        parts.append("🟢 今日报警数量较少，整体安全形势可控。")

    # 高频类型/通道
    if top_type and top_channel:
        parts.append(f"报警主要集中在**{top_type}**和**通道{top_channel}**，建议优先复查对应区域。")

    # 最近报警
    if recent:
        parts.append("**最近报警：**")
        for item in recent[:3]:
            ch_str = f" 通道{item.get('channel')}" if item.get("channel") else ""
            parts.append(f"- {item.get('type_label', '')}{ch_str} — {item.get('time_display', '')}")

    parts.append("\n建议保留告警截图与录屏用于后续追溯，可在「报警事件」页面查看详情。")
    return "\n\n".join(parts)


# ── Tasks ─────────────────────────────────────────────────


def build_tasks_answer(data: Dict[str, Any]) -> str:
    """生成任务配置回答。"""
    if data.get("status") == "error":
        return "抱歉，当前无法获取任务配置信息，请稍后重试。"

    tasks = data.get("tasks", [])
    task_count = data.get("task_count", 0)

    if task_count == 0:
        return "当前没有配置检测任务。请在「任务管理」页面新增检测任务，选择视频通道和检测器后启用。"

    parts = [f"当前共配置 **{task_count}** 个检测任务："]
    for i, task in enumerate(tasks, 1):
        status = "✅ 运行中" if task.get("running") else "⏸ 已停止"
        parts.append(
            f"{i}. **{task.get('name', '')}**（{task.get('stream_label', '')}）— {status}\n"
            f"   检测器：{'、'.join(task.get('detector_labels', []))}"
        )

    parts.append("\n可在「任务管理」页面启动、停止或修改任务配置。")
    return "\n\n".join(parts)


# ── Video Streams ─────────────────────────────────────────


def build_video_streams_answer(data: Dict[str, Any]) -> str:
    """生成视频流配置回答。"""
    if data.get("status") == "error":
        return "抱歉，当前无法获取视频流配置信息，请稍后重试。"

    streams = data.get("streams", [])
    connected = data.get("connected_count", 0)
    total = data.get("stream_count", 0)

    parts = [f"当前系统固定配置 **{total}** 路视频通道，其中 **{connected}** 路已连接："]
    for s in streams:
        icon = "✅" if s.get("connected") else "❌"
        parts.append(
            f"{icon} **{s.get('name', '')}** — "
            f"{s.get('source_type_label', 'RTSP')} — "
            f"地址：`{s.get('source', '未配置')}`"
        )

    if connected < total:
        parts.append(f"\n⚠️ 有 {total - connected} 路通道未配置视频源，对应画面在四宫格中显示为空。可在「视频流」页面配置。")

    parts.append("\n说明：RTSP 密码已自动脱敏处理，完整地址仅管理员可见。")
    return "\n\n".join(parts)


# ── Daily Report ──────────────────────────────────────────


def build_daily_report_answer(data: Dict[str, Any]) -> str:
    """生成安全巡检日报。"""
    if data.get("status") == "error":
        return "抱歉，当前无法生成巡检报告，请稍后重试。"

    report_date = data.get("report_date", "")
    report_time = data.get("report_time", "")
    system = data.get("system_status", {})
    alert_stats = data.get("alert_statistics", {})
    tasks = data.get("tasks", {})
    streams = data.get("streams", {})
    risk_level = data.get("risk_level", "low")

    total_alarms = alert_stats.get("total", 0)
    by_type = alert_stats.get("by_type", {})
    by_channel = alert_stats.get("by_channel", {})
    system_running = system.get("system_running", False)
    stream_total = streams.get("total", 0)
    stream_connected = streams.get("connected", 0)

    lines = [
        "# 粮库AI安全巡检日报",
        "",
        f"**日期：** {report_time}",
        f"**系统状态：** {'🟢 运行正常' if system_running else '🔴 已停止'}",
        f"**监控通道：** {stream_total}路（已连接 {stream_connected} 路）",
        f"**今日报警总数：** {total_alarms}条",
        f"**风险等级：** {_risk_label(risk_level)}",
        "",
    ]

    if total_alarms > 0:
        lines.append("## 报警类型统计")
        lines.append("")
        for type_name, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"1. {type_name}：{count}条")
        lines.append("")

        if by_channel:
            lines.append("## 通道报警分布")
            lines.append("")
            for ch, count in sorted(by_channel.items(), key=lambda x: int(x[0])):
                lines.append(f"- 通道{ch}：{count}条")
            lines.append("")

    # 风险分析
    lines.append("## 风险分析")
    lines.append("")
    if not system_running:
        lines.append("当前系统已停止运行，无法进行实时安全监控。请尽快启动系统以恢复监控能力。")
    elif total_alarms == 0:
        lines.append("今日系统运行平稳，无报警事件发生，各区域作业情况正常。")
    elif total_alarms <= 3:
        lines.append("今日报警数量较少，属于正常波动范围。建议继续保持现有安全监管策略。")
    elif total_alarms <= 10:
        top_type = alert_stats.get("top_type", "")
        top_ch = alert_stats.get("top_channel", "")
        lines.append(
            f"今日报警集中在{top_type}和通道{top_ch}，表明对应区域作业人员防护装备佩戴规范性不足，"
            "存在一定安全风险，需要关注。"
        )
    else:
        lines.append(
            "今日报警数量较多，安全形势需要高度重视。建议对高频报警区域进行现场复核，"
            "检查作业人员防护装备佩戴情况，必要时暂停相关区域作业并开展安全培训。"
        )
    lines.append("")

    # 处置建议
    lines.append("## 处置建议")
    lines.append("")
    idx = 1
    if not system_running:
        lines.append(f"{idx}. 立即启动AI安全监控系统；")
        idx += 1
    if total_alarms > 0:
        top_ch = alert_stats.get("top_channel")
        if top_ch:
            lines.append(f"{idx}. 对通道{top_ch}对应区域进行现场复核；")
            idx += 1
        lines.append(f"{idx}. 提醒作业人员规范佩戴安全帽和呼吸机；")
        idx += 1
    if total_alarms > 5:
        lines.append(f"{idx}. 对高频告警区域加强安全培训；")
        idx += 1
    if stream_connected < stream_total:
        lines.append(f"{idx}. 检查未连接通道的视频源配置；")
        idx += 1
    lines.append(f"{idx}. 保留告警截图和录屏用于后续追溯；")
    idx += 1
    lines.append(f"{idx}. 定期检查存储空间，确保告警文件正常留存。")
    lines.append("")

    lines.append("---")
    lines.append(f"*本报告由粮库AI安全监控平台自动生成，时间：{report_time}*")

    return "\n".join(lines)


# ── General Help ──────────────────────────────────────────


def build_general_help_answer() -> str:
    """生成通用帮助信息。"""
    return (
        "您好！我是**粮库安全监管智能体**，可以帮您完成以下操作：\n\n"
        "1. **查询系统状态** — 了解当前系统是否运行、哪些检测器已启用\n"
        "   示例：「系统运行正常吗？」\n\n"
        "2. **查询报警事件** — 查看最近的报警记录\n"
        "   示例：「今天有哪些报警？」「最近一次报警是什么？」\n\n"
        "3. **统计报警数据** — 按类型、通道统计今日报警\n"
        "   示例：「今天报警多少次？」「统计本周报警情况」\n\n"
        "4. **查询任务配置** — 了解当前检测任务的通道和检测器\n"
        "   示例：「当前配置了哪些检测任务？」\n\n"
        "5. **查询视频流** — 查看各通道的视频源连接状态\n"
        "   示例：「当前有哪些视频通道？」\n\n"
        "6. **生成巡检报告** — 自动生成每日安全巡检日报\n"
        "   示例：「生成今天的安全巡检报告」\n\n"
        "请直接输入您的问题，我会基于系统真实数据为您解答。"
    )


# ── Unknown Intent ────────────────────────────────────────


def build_unknown_intent_answer() -> str:
    """生成未知意图回答（引导用户使用已知功能）。"""
    return (
        "抱歉，我没有完全理解您的问题。您可以尝试以下方式提问：\n\n"
        "- 「当前系统运行正常吗？」\n"
        "- 「今天有哪些报警？」\n"
        "- 「生成今天的安全巡检报告」\n"
        "- 「当前配置了哪些检测任务？」\n"
        "- 「当前有哪些视频通道？」\n"
        "- 「最近一次报警是什么？」\n\n"
        "或者直接告诉我您的需求，我会尽力帮您解决。"
    )


# ── Error ─────────────────────────────────────────────────


def build_error_answer(error_msg: str = "") -> str:
    """生成错误回答。"""
    return (
        "AI安全助手处理失败，请稍后重试。"
        + (f"\n\n错误详情：{error_msg}" if error_msg else "")
    )


# ── Helper ────────────────────────────────────────────────


def _date_label(date_str: str) -> str:
    """将日期字符串转为人类可读标签。"""
    from datetime import date as dt_date

    try:
        d = dt_date.fromisoformat(date_str)
        today = dt_date.today()
        if d == today:
            return "今日"
        elif d == today - dt_date.resolution:
            return "昨日"
        return f"{d.month}月{d.day}日"
    except Exception:
        return date_str


def _risk_label(level: str) -> str:
    mapping = {"high": "🔴 高风险", "medium": "🟡 中等风险", "low": "🟢 低风险"}
    return mapping.get(level, level)
