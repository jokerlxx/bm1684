"""
AI Safety Agent — 粮库安全监管智能体核心。

职责：
1. 接收用户自然语言消息
2. 识别意图（规则匹配）
3. 调用对应工具函数获取真实系统数据
4. 优先使用 DeepSeek API 润色回答
5. DeepSeek 不可用时回退本地模板
6. 返回统一的 JSON 格式结果
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logging_utils import log_ai_assistant_response

from app.application.agent.agent_tools import (
    generate_daily_report_tool,
    get_alert_history_tool,
    get_alert_statistics_tool,
    get_system_status_tool,
    get_tasks_tool,
    get_video_streams_tool,
)
from app.application.agent.llm_client import LLMClient, SYSTEM_PROMPT
from app.application.agent.prompt_templates import (
    build_alert_history_answer,
    build_alert_statistics_answer,
    build_daily_report_answer,
    build_error_answer,
    build_general_help_answer,
    build_system_status_answer,
    build_tasks_answer,
    build_unknown_intent_answer,
    build_video_streams_answer,
)

logger = logging.getLogger(__name__)

# 意图关键词匹配规则（顺序影响优先级）
INTENT_RULES: List[Tuple[str, List[str]]] = [
    # 报告（含"今天/今日"且含"报警/告警"时已被alert_statistics抢先，
    # 但纯"报告/巡检/日报/总结"仍走generate_report）
    ("generate_report", ["报告", "日报", "巡检报告", "总结"]),
    # 统计优先于一般报警查询（"今天/今日"+"报警/告警/事件/异常" → alert_statistics）
    ("alert_statistics", ["今天", "今日", "统计", "多少次", "多少条"]),
    # 报警查询
    ("query_alerts", ["报警", "告警", "事件", "异常"]),
    # 系统状态
    ("query_system_status", ["状态", "运行", "正常吗", "系统情况", "是否正常"]),
    # 任务配置
    ("query_tasks", ["任务", "检测器", "配置了什么"]),
    # 视频流
    ("query_video_streams", ["视频流", "视频通道", "摄像头", "rtsp"]),
]

# "今天/今日/最近" 与 "报警/告警" 同时出现 → alert_statistics
_INTENT_OVERRIDE_PAIRS: List[Tuple[List[str], List[str], str]] = [
    (
        ["今天", "今日", "本周", "本月", "最近"],
        ["报警", "告警", "事件", "异常", "多少次", "多少条"],
        "alert_statistics",
    ),
    (
        ["今天", "今日", "本周", "本月"],
        ["报告", "日报", "巡检"],
        "generate_report",
    ),
]


def _recognize_intent(message: str) -> str:
    """基于关键词规则识别用户意图。

    优先级：
    1. 组合匹配（如"今天"+"报警" → alert_statistics）
    2. 单关键词匹配（按 INTENT_RULES 顺序）
    3. 默认 → general_help
    """
    msg_lower = message.lower().strip()

    # 1. 组合匹配
    for prefix_keys, suffix_keys, intent in _INTENT_OVERRIDE_PAIRS:
        has_prefix = any(kw in msg_lower for kw in prefix_keys)
        has_suffix = any(kw in msg_lower for kw in suffix_keys)
        if has_prefix and has_suffix:
            return intent

    # 2. 单关键词匹配
    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in msg_lower:
                return intent

    # 3. 默认
    return "general_help"


def _build_llm_prompt(intent: str, question: str, tool_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """构建发送给 DeepSeek 的消息列表。"""
    data_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    # 截断过长数据
    if len(data_json) > 4000:
        data_json = data_json[:4000] + "\n... (truncated)"

    intent_labels = {
        "query_system_status": "系统状态查询",
        "query_alerts": "报警事件查询",
        "alert_statistics": "报警统计",
        "query_tasks": "任务配置查询",
        "query_video_streams": "视频流配置查询",
        "generate_report": "生成安全巡检日报",
        "general_help": "通用帮助",
    }

    user_content = (
        f"用户问题：{question}\n"
        f"意图：{intent_labels.get(intent, intent)}\n"
        f"工具名称：{intent}\n"
        f"系统数据：\n{data_json}\n"
        f"\n请基于以上真实数据回答用户问题。不要编造任何数据中不存在的信息。"
        f"回答应简洁、准确、面向粮库安全管理人员。如有安全风险，请给出分析和建议。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class SafetyAgent:
    """粮库安全监管智能体。"""

    def __init__(
        self,
        status_handler: Optional[Callable[[], Dict[str, Any]]] = None,
        alerts_handler: Optional[Callable[[], Dict[str, Any]]] = None,
        tasks_handler: Optional[Callable[[], Dict[str, Any]]] = None,
        streams_handler: Optional[Callable[[], Dict[str, Any]]] = None,
        scheduler_getter: Optional[Callable[[], Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.status_handler = status_handler or (lambda: {})
        self.alerts_handler = alerts_handler or (lambda: {"items": []})
        self.tasks_handler = tasks_handler or (lambda: {"tasks": []})
        self.streams_handler = streams_handler or (lambda: {"streams": []})
        self.scheduler_getter = scheduler_getter
        self.config = config or {}
        self.llm = LLMClient(config=self.config)

    def _call_tool(self, intent: str, message: str) -> Tuple[str, Dict[str, Any]]:
        """根据意图调用对应工具函数。

        Returns:
            (tool_name, tool_result)
        """
        # 从消息中提取日期
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", message)
        date_arg = date_match.group(1) if date_match else None

        # 从消息中提取通道号
        channel_match = re.search(r"通道\s*(\d)", message)
        channel_arg = int(channel_match.group(1)) if channel_match else None

        # 从消息中提取报警类型
        alarm_type_map = {
            "安全帽": "helmet",
            "摔倒": "fall",
            "跌倒": "fall",
            "打架": "fight",
            "聚集": "crowd",
            "呼吸机": "ventilator",
            "门窗": None,  # 细分
            "仓内": "window_door_inside",
            "仓外": "window_door_outside",
        }
        alarm_type_arg = None
        for kw, atype in alarm_type_map.items():
            if kw in message and atype:
                alarm_type_arg = atype
                break

        if intent == "query_system_status":
            alert_items = self.alerts_handler().get("items", [])
            return "get_system_status", get_system_status_tool(
                self.status_handler, alert_items=alert_items
            )

        elif intent == "query_alerts":
            return "get_alert_history", get_alert_history_tool(
                self.alerts_handler,
                date=date_arg,
                alarm_type=alarm_type_arg,
                channel=channel_arg,
                limit=50,
            )

        elif intent == "alert_statistics":
            return "get_alert_statistics", get_alert_statistics_tool(
                self.alerts_handler, date=date_arg
            )

        elif intent == "query_tasks":
            return "get_tasks", get_tasks_tool(
                self.tasks_handler, scheduler_getter=self.scheduler_getter
            )

        elif intent == "query_video_streams":
            return "get_video_streams", get_video_streams_tool(self.streams_handler)

        elif intent == "generate_report":
            return "generate_daily_report", generate_daily_report_tool(
                self.status_handler,
                self.alerts_handler,
                self.tasks_handler,
                self.streams_handler,
                date=date_arg,
            )

        else:
            return "none", {}

    def _try_llm_answer(self, intent: str, message: str, tool_result: Dict[str, Any]) -> Optional[str]:
        """尝试通过 DeepSeek API 生成回答。

        Returns:
            成功：回答字符串
            失败：None
        """
        llm_messages = _build_llm_prompt(intent, message, tool_result)
        try:
            answer = self.llm.chat(llm_messages)
            if answer:
                return answer
        except Exception as exc:
            logger.warning("LLM fallback triggered: %s", exc)
        return None

    def _local_answer(self, intent: str, tool_result: Dict[str, Any]) -> str:
        """使用本地模板生成回答。"""
        builders = {
            "query_system_status": build_system_status_answer,
            "query_alerts": build_alert_history_answer,
            "alert_statistics": build_alert_statistics_answer,
            "query_tasks": build_tasks_answer,
            "query_video_streams": build_video_streams_answer,
            "generate_report": build_daily_report_answer,
            "general_help": lambda _d: build_general_help_answer(),
            "unknown": lambda _d: build_unknown_intent_answer(),
        }
        builder = builders.get(intent, builders["unknown"])
        try:
            return builder(tool_result)
        except Exception as exc:
            logger.exception("Local template builder failed for intent=%s: %s", intent, exc)
            return build_error_answer(str(exc))

    def chat(self, message: str) -> Dict[str, Any]:
        """处理用户消息，返回统一 JSON 响应。

        Args:
            message: 用户自然语言消息

        Returns:
            {
                "success": bool,
                "answer": str,
                "intent": str,
                "tool_used": str | None,
                "data": dict,
            }
        """
        if not message or not message.strip():
            return {
                "success": True,
                "answer": "您好！请问有什么可以帮助您的？\n\n您可以尝试问：「系统运行正常吗？」「今天有哪些报警？」「生成巡检报告」等。",
                "intent": "general_help",
                "tool_used": None,
                "data": {},
            }

        start = time.monotonic()
        # Step 1: 意图识别
        intent = _recognize_intent(message)
        logger.info("Agent intent recognized: %s → %s", message[:80], intent)

        # Step 2: 调用工具获取数据
        tool_used, tool_result = self._call_tool(intent, message)

        # Step 3: 尝试 DeepSeek 生成回答
        answer = None
        llm_attempted = False
        fallback_reason = None
        if tool_used != "none" and self.llm.enabled:
            llm_attempted = True
            answer = self._try_llm_answer(intent, message, tool_result)
            if answer is None:
                fallback_reason = "DeepSeek未返回或调用失败"

        # Step 4: DeepSeek 不可用时回退本地模板
        if answer is None:
            answer = self._local_answer(intent, tool_result)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        if llm_attempted and answer and not fallback_reason:
            source = "deepseek"
        elif llm_attempted:
            source = "deepseek_fallback"
        else:
            source = "local"
        log_ai_assistant_response(intent, elapsed_ms, source, fallback_reason=fallback_reason)

        return {
            "success": True,
            "answer": answer,
            "intent": intent,
            "tool_used": tool_used if tool_used != "none" else None,
            "data": tool_result,
        }
