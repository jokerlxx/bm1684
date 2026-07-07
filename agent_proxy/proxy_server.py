"""
上位机 DeepSeek 代理服务 (Agent Proxy) — 分级智能响应机制

职责：
1. 接收前端 AI 助手请求
2. 转发给 BM1684 的 /api/agent/chat → 获取真实系统数据
3. 判断问题类型：查询类 → 直接返回 BM1684 本地回答；分析类 → DeepSeek 增强
4. DeepSeek 失败/超时时自动回退 BM1684 本地回答
5. BM1684 不直接访问外网，API Key 不返回前端

启动方式：
  cd agent_proxy
  python proxy_server.py

环境变量：
  BM1684_BASE_URL=http://192.168.150.5:5010
  LLM_ENABLE=true
  LLM_API_KEY=sk-xxx
  LLM_BASE_URL=https://api.deepseek.com
  LLM_MODEL=deepseek-chat
  LLM_TIMEOUT=8            # 默认 8 秒
  PROXY_PORT=7000
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

# ── .env 文件加载（最低优先级，环境变量覆盖）───────────────
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

# ── 日志（同时输出到控制台和文件）────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "proxy.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] [AgentProxy] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_root_logger.addHandler(_console)

_file = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_file.setFormatter(_fmt)
_root_logger.addHandler(_file)

logger = logging.getLogger("AgentProxy")

# ── 配置（优先级：环境变量 > .env 文件 > 默认值）─────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

BM1684_BASE_URL = _env("BM1684_BASE_URL", "http://192.168.150.5:5010").rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "7000"))

LLM_ENABLE = (_env("LLM_ENABLE", "false").lower() in ("true", "1", "yes", "on"))
LLM_API_KEY = _env("LLM_API_KEY", "")
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = _env("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "8"))

if LLM_ENABLE and not LLM_API_KEY:
    logger.warning("LLM_ENABLE=true but LLM_API_KEY is empty → LLM disabled")
    LLM_ENABLE = False

logger.info(
    "Agent Proxy config: bm1684=%s llm_enable=%s model=%s timeout=%.1fs",
    BM1684_BASE_URL, LLM_ENABLE, LLM_MODEL, LLM_TIMEOUT,
)

# ── DeepSeek 系统提示词 ───────────────────────────────────
SYSTEM_PROMPT = (
    "你是粮库AI安全监管平台的安全助手。你只能基于系统工具返回的数据回答问题，"
    "不得编造报警事件、通道状态、任务配置、检测结果或文件信息。"
    "回答应面向粮库安全管理人员，要求简洁、准确、可执行。"
    "发现安全风险时，需要给出风险分析和处置建议。"
    "如果数据不足以回答问题，请如实说明。"
)

# ── Flask ─────────────────────────────────────────────────
app = Flask(__name__)


@app.after_request
def _add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


# ═══════════════════════════════════════════════════════════
#  分级智能响应 — 核心判断函数
# ═══════════════════════════════════════════════════════════

# 查询类关键词 → 不调 DeepSeek，直接用 BM1684 本地回答
_LOCAL_KEYWORDS = [
    # 系统状态
    "状态", "运行", "正常吗", "是否正常", "系统情况",
    # 报警查询
    "报警", "告警", "几次", "多少次", "多少条",
    "今天有哪些", "最近一次", "有没有",
    # 通道/视频流
    "通道", "视频流", "视频通道", "摄像头", "rtsp",
    # 任务
    "任务", "检测器", "配置", "配置了哪些",
]

# 分析/生成类关键词 → 调 DeepSeek
_LLM_KEYWORDS = [
    "分析", "建议", "如何处理", "怎么处理", "怎么整改",
    "整改", "优化", "方案", "报告", "日报", "总结",
    "撰写", "生成", "风险研判", "处置", "措施",
    "比赛", "创新点", "介绍", "说明", "阐述",
]

# 组合：同时含"报警+分析/建议/处理/报告" → 调 DeepSeek
_LLM_COMBO_PREFIXES = ["报警", "告警", "事件", "安全", "风险"]

_LLM_COMBO_SUFFIXES = ["分析", "建议", "处理", "报告", "方案", "处置", "整改", "优化"]


def should_use_llm(message: str, bm_result: Optional[Dict[str, Any]] = None) -> bool:
    """判断当前问题是否需要调用 DeepSeek 增强。

    规则（优先级从高到低）：
    1. LLM_ENABLE=False → 直接 False
    2. BM1684 返回的意图已是 generate_report → True
    3. 组合匹配：报警+分析/建议/报告 → True
    4. 含分析/生成类关键词 → True
    5. 含查询类关键词 → False
    6. 默认 → False（保守策略，优先本地）
    """
    if not LLM_ENABLE:
        return False

    msg = message.lower().strip()
    if not msg:
        return False

    # 规则 2：BM1684 意图层面判定
    intent = (bm_result or {}).get("intent", "")
    if intent == "generate_report":
        return True

    # 规则 3：组合匹配 — "报警" + "分析/建议/处理/报告" → LLM
    has_combo_prefix = any(kw in msg for kw in _LLM_COMBO_PREFIXES)
    has_combo_suffix = any(kw in msg for kw in _LLM_COMBO_SUFFIXES)
    if has_combo_prefix and has_combo_suffix:
        logger.info("should_use_llm: combo match → True")
        return True

    # 规则 4：含分析/生成类关键词 → LLM
    if any(kw in msg for kw in _LLM_KEYWORDS):
        logger.info("should_use_llm: LLM keyword match → True")
        return True

    # 规则 5：含查询类关键词 → 本地
    if any(kw in msg for kw in _LOCAL_KEYWORDS):
        logger.info("should_use_llm: local keyword match → False")
        return False

    # 规则 6：默认不调 LLM
    return False


def _build_llm_answer_prompt(intent: str, question: str, bm_data: Dict[str, Any]) -> str:
    """构建发送给 DeepSeek 的分析型提示词。"""
    data_json = json.dumps(bm_data, ensure_ascii=False, indent=2)
    if len(data_json) > 4000:
        data_json = data_json[:4000] + "\n... (truncated)"

    intent_hints = {
        "generate_report": "请生成一份完整的粮库安全巡检报告，包含风险分析和处置建议。",
        "alert_statistics": "请基于报警统计数据给出风险分析和安全建议。",
        "query_alerts": "请分析报警事件趋势并给出防范建议。",
        "general_help": "请给出专业的安全管理建议。",
    }
    hint = intent_hints.get(intent, "请给出专业分析和可执行的建议。")

    return (
        f"用户问题：{question}\n"
        f"意图：{intent}\n"
        f"系统数据：\n{data_json}\n"
        f"\n{hint}\n"
        f"要求：基于以上真实数据回答，不得编造任何信息。"
        f"回答应简洁、准确、面向粮库安全管理人员。"
    )


# ═══════════════════════════════════════════════════════════
#  核心 HTTP 调用
# ═══════════════════════════════════════════════════════════

def _fetch_bm1684_answer(message: str) -> Dict[str, Any]:
    """请求 BM1684 /api/agent/chat。"""
    url = f"{BM1684_BASE_URL}/api/agent/chat"
    payload = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("BM1684 request failed: %s", exc)
        return {"success": False, "answer": "AI安全助手处理失败，请稍后重试。", "intent": "error", "tool_used": None, "data": {}, "_error": str(exc)}


def _call_deepseek(intent: str, question: str, bm_data: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """调用 DeepSeek API。返回 (answer, fallback_reason)。"""
    if not LLM_ENABLE or not LLM_API_KEY:
        return None, "LLM not enabled or API key missing"

    user_content = _build_llm_answer_prompt(intent, question, bm_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    payload = json.dumps({"model": LLM_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 800, "stream": False}).encode("utf-8")
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"}

    start = time.monotonic()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            elapsed = time.monotonic() - start
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            logger.info("DeepSeek OK: %.1fs tokens(prompt=%s completion=%s)", elapsed,
                        result.get("usage", {}).get("prompt_tokens", "?"),
                        result.get("usage", {}).get("completion_tokens", "?"))
            return str(content).strip(), None
    except Exception as exc:
        elapsed = time.monotonic() - start
        reason = f"DeepSeek 调用失败（{elapsed:.0f}s）：{exc}" if elapsed > LLM_TIMEOUT * 0.8 else f"DeepSeek 不可用：{exc}"
        logger.warning("DeepSeek failed after %.1fs: %s", elapsed, exc)
        return None, reason


# ═══════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════

@app.route("/proxy/agent/chat", methods=["POST"])
def proxy_agent_chat():
    """分级智能响应：查询类→本地快速回答，分析类→DeepSeek 增强。

    返回字段：
      success, answer, intent, source, llm, llm_reason, tool_used, data
    source 取值：
      "bm1684_local"      — 本地快速回答（查询类）
      "deepseek_enhanced" — 大模型增强（分析类，成功）
      "bm1684_fallback"   — 本地兜底（分析类，大模型失败）
    """
    try:
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"success": False, "answer": "请发送一条消息。", "intent": "error", "source": "proxy"})

        # Step 1: 请求 BM1684
        bm_start = time.monotonic()
        bm_result = _fetch_bm1684_answer(message)
        logger.info("BM1684: %.2fs intent=%s", time.monotonic() - bm_start, bm_result.get("intent", "?"))
        if not bm_result.get("success"):
            return jsonify({"success": False, "answer": bm_result.get("answer", "BM1684 服务异常"), "intent": "error", "source": "bm1684"})

        intent = bm_result.get("intent", "unknown")
        bm_answer = bm_result.get("answer", "")
        bm_data = bm_result.get("data", {})

        # Step 2: 判断是否需要 LLM
        use_llm = should_use_llm(message, bm_result)

        if not use_llm:
            logger.info("Decision: LOCAL → returning BM1684 answer directly")
            return jsonify({
                "success": True, "answer": bm_answer, "intent": intent,
                "source": "bm1684_local", "llm": False,
                "tool_used": bm_result.get("tool_used"), "data": bm_data,
            })

        # Step 3: 调用 DeepSeek
        logger.info("Decision: LLM → calling DeepSeek for '%s'", intent)
        llm_answer, fallback_reason = _call_deepseek(intent, message, bm_data)

        if llm_answer:
            return jsonify({
                "success": True, "answer": llm_answer, "intent": intent,
                "source": "deepseek_enhanced", "llm": True,
                "tool_used": bm_result.get("tool_used"), "data": bm_data,
            })

        # Step 4: LLM 失败 → 回退
        logger.info("Decision: FALLBACK → %s", fallback_reason)
        fallback_answer = bm_answer
        if fallback_reason:
            fallback_answer += "\n\n---\n💡 提示：大模型响应超时/不可用，本次已使用本地安全助手回答。"

        return jsonify({
            "success": True, "answer": fallback_answer, "intent": intent,
            "source": "bm1684_fallback", "llm": False,
            "llm_reason": fallback_reason,
            "tool_used": bm_result.get("tool_used"), "data": bm_data,
        })

    except Exception as exc:
        logger.exception("Proxy error: %s", exc)
        return jsonify({"success": False, "answer": "AI安全助手处理失败，请稍后重试。", "intent": "error", "source": "proxy"})


@app.route("/proxy/health", methods=["GET"])
def proxy_health():
    return jsonify({"status": "ok", "bm1684_base_url": BM1684_BASE_URL, "llm_enabled": LLM_ENABLE, "llm_timeout": LLM_TIMEOUT})


if __name__ == "__main__":
    logger.info("Starting Agent Proxy on 0.0.0.0:%d", PROXY_PORT)
    app.run(host="0.0.0.0", port=PROXY_PORT, debug=False, threaded=True)
