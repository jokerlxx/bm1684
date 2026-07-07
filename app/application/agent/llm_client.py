"""
DeepSeek API Client for AI Safety Agent.

配置优先级：环境变量 > config_bm1684x.json 的 llm 字段 > 默认值。
通过环境变量配置（可选）：
    LLM_ENABLE=true
    LLM_API_KEY=sk-xxx
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL=deepseek-chat
通过 config_bm1684x.json 的 llm 字段配置（推荐）：
    "llm": { "enabled": true, "api_key": "sk-xxx", ... }

当未配置或调用失败时，外部调用者应回退到本地模板。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


class LLMClient:
    """DeepSeek API 客户端（OpenAI-compatible Chat Completions 接口）。

    配置优先级：环境变量 > config JSON > 默认值
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 12.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        # 从 config JSON 的 llm 字段读取（最低优先级）
        llm_cfg = (config or {}).get("llm", {}) if config else {}
        cfg_enabled = bool(llm_cfg.get("enabled", False))
        cfg_api_key = str(llm_cfg.get("api_key", "") or "").strip()
        cfg_base_url = str(llm_cfg.get("base_url", "") or "").strip()
        cfg_model = str(llm_cfg.get("model", "") or "").strip()
        cfg_timeout = float(llm_cfg.get("timeout", 0) or 0)

        # 环境变量覆盖 config JSON（高优先级）
        self.enabled = _env_bool("LLM_ENABLE", default=cfg_enabled)
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "") or cfg_api_key
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "") or cfg_base_url or "https://api.deepseek.com"
        self.model = model or os.environ.get("LLM_MODEL", "") or cfg_model or "deepseek-chat"
        self.timeout = max(8.0, min(15.0, float(timeout if timeout != 12.0 else (cfg_timeout or 12.0))))

        if self.enabled and not self.api_key:
            logger.warning("LLM enabled but api_key is empty, LLM will be disabled")
            self.enabled = False

        if self.enabled:
            logger.info(
                "LLM client initialized: model=%s base_url=%s timeout=%.1fs",
                self.model,
                self.base_url,
                self.timeout,
            )
        else:
            logger.info("LLM client disabled, will use local template fallback")

    def chat(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> Optional[str]:
        """调用 DeepSeek Chat Completions API。

        Args:
            messages: 消息列表 [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大 token 数

        Returns:
            成功：返回助手的文本回答
            失败：返回 None
        """
        if not self.enabled or not self.api_key:
            return None

        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed = time.monotonic() - start
                body = resp.read().decode("utf-8")
                result = json.loads(body)
                content = result["choices"][0]["message"]["content"]
                logger.info(
                    "LLM call succeeded: model=%s elapsed=%.2fs tokens(prompt=%s, completion=%s)",
                    self.model,
                    elapsed,
                    result.get("usage", {}).get("prompt_tokens", "?"),
                    result.get("usage", {}).get("completion_tokens", "?"),
                )
                return str(content).strip()

        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return None


# 预置的系统提示词
SYSTEM_PROMPT = (
    "你是粮库AI安全监管平台的安全助手。你只能基于系统工具返回的数据回答问题，"
    "不得编造报警事件、通道状态、任务配置、检测结果或文件信息。"
    "回答应面向粮库安全管理人员，要求简洁、准确、可执行。"
    "发现安全风险时，需要给出风险分析和处置建议。"
    "如果数据不足以回答问题，请如实说明。"
)
