"""
视频流接入层（Ingestion Layer）：接收并缓冲多路视频流。
从 RTSP/本地文件读取帧，以事件形式写入共享队列，供检测引擎消费。
"""

from ingestion.stream_service import run_stream_service

__all__ = ["run_stream_service"]
