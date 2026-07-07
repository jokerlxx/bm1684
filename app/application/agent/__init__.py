"""
AI Safety Agent — 粮库安全监管智能体模块。

提供自然语言交互接口，支持：
- 系统状态查询
- 报警事件查询与统计
- 任务配置查询
- 视频流配置查询
- 安全巡检日报生成
- DeepSeek API 自然语言润色（可选）
- 本地规则模板兜底
"""

from app.application.agent.safety_agent import SafetyAgent

__all__ = ["SafetyAgent"]
