"""
统一日志配置工具。

当前工程由主进程拉起多个子进程：
- 根 logger 输出到终端和主日志文件，用于常规运行日志
- 指标 logger 单独写入 bm-web.log，用于保存中文指标日志
"""

from pathlib import Path
import logging
import os
import time


DEFAULT_MAIN_LOG_FILE = "bm-main.log"
DEFAULT_METRIC_LOG_FILE = "bm-web.log"
MAIN_LOG_ENV_VAR = "BM_MAIN_LOG_FILE"
METRIC_LOG_ENV_VAR = "BM_WEB_LOG_FILE"
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
METRIC_LOGGER_NAME = "bm_metrics"
METRIC_LOG_FORMAT = "[%(asctime)s] %(message)s"


def _resolve_log_path(log_file=None, env_var=None, default_name=None):
    target = log_file or os.environ.get(env_var or "") or default_name
    log_path = Path(target).expanduser()
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def get_main_log_path(log_file=None):
    return str(
        _resolve_log_path(
            log_file=log_file,
            env_var=MAIN_LOG_ENV_VAR,
            default_name=DEFAULT_MAIN_LOG_FILE,
        )
    )


def get_metrics_log_path(log_file=None):
    return str(
        _resolve_log_path(
            log_file=log_file,
            env_var=METRIC_LOG_ENV_VAR,
            default_name=DEFAULT_METRIC_LOG_FILE,
        )
    )


def _same_path(path_a, path_b):
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except Exception:
        return str(path_a) == str(path_b)


def _clear_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _has_console_handler(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


def _has_file_handler(logger, log_path):
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        filename = getattr(handler, "baseFilename", None)
        if filename and _same_path(filename, log_path):
            return True
    return False


def configure_root_logging(level=logging.INFO, log_file=None, force=False):
    """
    配置根日志。

    Args:
        level: 日志级别
        log_file: 主日志文件路径，默认使用 BM_MAIN_LOG_FILE 或 bm-main.log
        force: 是否强制替换已有 handlers
    """
    main_log_path = Path(get_main_log_path(log_file))
    os.environ[MAIN_LOG_ENV_VAR] = str(main_log_path)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if force:
        _clear_handlers(root_logger)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    if not _has_console_handler(root_logger):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if not _has_file_handler(root_logger, main_log_path):
        file_handler = logging.FileHandler(str(main_log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    configure_metrics_logging(force=force)
    return str(main_log_path)


def ensure_root_logging(level=logging.INFO, log_file=None):
    """确保当前进程已接入主日志文件与终端日志。"""
    return configure_root_logging(level=level, log_file=log_file, force=False)


def configure_metrics_logging(log_file=None, force=False):
    """配置仅写文件的中文指标日志。"""
    log_path = Path(get_metrics_log_path(log_file))
    os.environ[METRIC_LOG_ENV_VAR] = str(log_path)
    logger = logging.getLogger(METRIC_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if force:
        _clear_handlers(logger)

    if not _has_file_handler(logger, log_path):
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(METRIC_LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(file_handler)

    return str(log_path)


def get_metrics_logger():
    return logging.getLogger(METRIC_LOGGER_NAME)


def format_channel_name(stream_id):
    return f"通道{int(stream_id) + 1}"


_INFERENCE_METRICS = {}
_INFERENCE_WARMUP = 1
_INFERENCE_LOG_EVERY = 30
_LOADED_MODELS = set()

_MODEL_PROFILES = {
    "fight_detection": ("打架检测", "YOLO26-Fight"),
    "fall_detection": ("摔倒检测", "YOLO26-Fall"),
    "ventilator_equipment": ("呼吸机检测", "YOLO26-Gasmask"),
    "helmet_detection": ("安全帽检测", "YOLO26-Helmet"),
    "window_door_inside": ("仓内门窗检测", "YOLO26-WindowDoor"),
    "window_door_outside": ("仓外门窗检测", "YOLO26-WindowDoor"),
}


def _guess_input_size(model_key, model_path=None):
    path = str(model_path or "").lower()
    if "320" in path or model_key == "fall_detection":
        return "320×320"
    if "640" in path:
        return "640×640"
    return "640×640"


def _resolve_model_perf_profile(model_key, detector_name=None, model_path=None):
    if model_key == "helmet_detection" and detector_name == "crowd":
        return ("人员聚集检测", "Person + 聚类分析", "640×640")
    if model_key == "helmet_detection" and detector_name == "ventilator":
        return ("呼吸机检测", "YOLO26-Helmet(人员定位)", "640×640")
    task, model = _MODEL_PROFILES.get(
        model_key,
        (str(model_key), str(model_key)),
    )
    return task, model, _guess_input_size(model_key, model_path)


def _metrics_state_key(stream_id, model_key, detector_name=None):
    return (int(stream_id), str(model_key), str(detector_name or ""))


def log_model_loaded(model_key, model_path=None):
    model_key = str(model_key)
    if model_key in _LOADED_MODELS:
        return
    _LOADED_MODELS.add(model_key)
    task, model, input_size = _resolve_model_perf_profile(model_key, model_path=model_path)
    get_metrics_logger().info(
        "【模型加载】检测任务=%s，模型=%s，输入尺寸=%s，model_key=%s",
        task,
        model,
        input_size,
        model_key,
    )


def log_model_inference_metrics(
    stream_id,
    model_key,
    timings=None,
    detector_name=None,
    model_path=None,
    batch_size=1,
):
    """写入 bm-web.log 的模型推理性能指标（对应论文表5-1）。"""
    timings = dict(timings or {})
    preprocess_ms = float(timings.get("preprocess_ms") or 0.0)
    inference_ms = float(timings.get("inference_ms") or 0.0)
    postprocess_ms = float(timings.get("postprocess_ms") or 0.0)
    total_ms = float(timings.get("total_ms") or (preprocess_ms + inference_ms + postprocess_ms))
    if total_ms <= 0.0:
        return

    task, model, input_size = _resolve_model_perf_profile(
        str(model_key),
        detector_name=detector_name,
        model_path=model_path,
    )
    key = _metrics_state_key(stream_id, model_key, detector_name)
    now = time.monotonic()
    state = _INFERENCE_METRICS.setdefault(
        key,
        {
            "count": 0,
            "sum_pre_ms": 0.0,
            "sum_infer_ms": 0.0,
            "sum_post_ms": 0.0,
            "sum_total_ms": 0.0,
            "last_pre_ms": 0.0,
            "last_infer_ms": 0.0,
            "last_post_ms": 0.0,
            "last_total_ms": 0.0,
            "first_ts": now,
            "started": False,
        },
    )
    state["count"] += 1
    state["sum_pre_ms"] += preprocess_ms
    state["sum_infer_ms"] += inference_ms
    state["sum_post_ms"] += postprocess_ms
    state["sum_total_ms"] += total_ms
    state["last_pre_ms"] = preprocess_ms
    state["last_infer_ms"] = inference_ms
    state["last_post_ms"] = postprocess_ms
    state["last_total_ms"] = total_ms

    if not state["started"]:
        state["started"] = True
        get_metrics_logger().info(
            "【模型启动】%s %s %s %s 开始推理，batch=%d",
            task,
            model,
            input_size,
            format_channel_name(stream_id),
            int(batch_size),
        )

    should_log = state["count"] <= _INFERENCE_WARMUP
    if not should_log and state["count"] % _INFERENCE_LOG_EVERY == 0:
        should_log = True
    if not should_log:
        return

    count = state["count"]
    avg_pre_ms = state["sum_pre_ms"] / count
    avg_infer_ms = state["sum_infer_ms"] / count
    avg_post_ms = state["sum_post_ms"] / count
    avg_total_ms = state["sum_total_ms"] / count
    elapsed_s = max(0.001, now - state["first_ts"])
    throughput_fps = count / elapsed_s
    peak_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    get_metrics_logger().info(
        "【模型性能】检测任务=%s，模型=%s，输入尺寸=%s，预处理/ms=%.2f，单帧推理/ms=%.2f，"
        "后处理/ms=%.2f，端到端/ms=%.2f，平均FPS=%.2f，吞吐FPS=%.2f，%s，样本=%d",
        task,
        model,
        input_size,
        avg_pre_ms,
        avg_infer_ms,
        avg_post_ms,
        avg_total_ms,
        peak_fps,
        throughput_fps,
        format_channel_name(stream_id),
        count,
    )
    if state["count"] <= _INFERENCE_WARMUP:
        get_metrics_logger().info(
            "【模型性能】%s 本次预处理/ms=%.2f，单帧推理/ms=%.2f，后处理/ms=%.2f，端到端/ms=%.2f",
            format_channel_name(stream_id),
            state["last_pre_ms"],
            state["last_infer_ms"],
            state["last_post_ms"],
            state["last_total_ms"],
        )


def log_inference_time(stream_id, detector_name, total_ms):
    """兼容旧接口：仅 total_ms 可用时仍写入 bm-web.log。"""
    log_model_inference_metrics(
        stream_id,
        detector_name,
        timings={"total_ms": float(total_ms)},
    )


def log_system_response(category, content, response_ms, result=None):
    """写入 bm-web.log 的系统响应指标（对应论文表5-2）。"""
    extras = []
    if result:
        extras.append(f"结果={result}")
    suffix = f"，{'，'.join(extras)}" if extras else ""
    get_metrics_logger().info(
        "【系统响应】%s，%s，响应时间: %.0f ms%s",
        category,
        content,
        float(response_ms),
        suffix,
    )


def log_alarm_snapshot(alarm_type, response_ms, image_path=None, stream_id=None):
    channel = format_channel_name(stream_id) if stream_id is not None else "未知通道"
    detail = f"告警类型={alarm_type}，{channel}"
    if image_path:
        detail += f"，文件={image_path}"
    log_system_response("告警截图生成", detail, response_ms, result="成功")


def log_alarm_video(alarm_type, response_ms, video_path=None, frame_count=None, stream_id=None):
    channel = format_channel_name(stream_id) if stream_id is not None else "未知通道"
    detail = f"告警类型={alarm_type}，{channel}"
    if frame_count is not None:
        detail += f"，帧数={int(frame_count)}"
    if video_path:
        detail += f"，文件={video_path}"
    log_system_response("告警录屏生成", detail, response_ms, result="成功")


def log_ai_assistant_response(intent, response_ms, source, fallback_reason=None):
    if source == "deepseek":
        category = "DeepSeek增强回答"
        result = "成功"
    elif source == "deepseek_fallback":
        category = "DeepSeek增强回答"
        result = "超时回退本地回答"
    else:
        category = "本地AI助手查询"
        result = "成功"
    detail = f"意图={intent}"
    if fallback_reason:
        detail += f"，原因={fallback_reason}"
    log_system_response(category, detail, response_ms, result=result)


def log_pull_fps(stream_id, actual_fps, target_fps=None, queue_size=None, source_type="RTSP"):
    extras = [f"拉流帧率: {actual_fps:.2f}"]
    if target_fps is not None:
        extras.append(f"目标帧率: {int(target_fps)}")
    if queue_size is not None:
        extras.append(f"队列: {int(queue_size)}")
    get_metrics_logger().info("【拉流】%s %s，%s", format_channel_name(stream_id), source_type, "，".join(extras))


def log_push_fps(push_name, actual_fps, target_fps=None, queue_size=None):
    extras = [f"推流帧率: {actual_fps:.2f}"]
    if target_fps is not None:
        extras.append(f"目标帧率: {int(target_fps)}")
    if queue_size is not None:
        extras.append(f"队列: {int(queue_size)}")
    get_metrics_logger().info("【推流】%s，%s", push_name, "，".join(extras))


def log_preview_fps(stage_name, actual_fps, target_fps=None, stream_count=None):
    extras = [f"预览帧率: {actual_fps:.2f}"]
    if target_fps is not None:
        extras.append(f"目标帧率: {int(target_fps)}")
    get_metrics_logger().info("【预览】%s，%s", stage_name, "，".join(extras))
    if stage_name == "四宫格合成" and stream_count and int(stream_count) >= 4:
        get_metrics_logger().info(
            "【系统响应】多路视频预览，%d路视频流实时显示，预览帧率: %.2f FPS，目标: %s",
            int(stream_count),
            float(actual_fps),
            int(target_fps) if target_fps is not None else "-",
        )


def log_hls_status(actual_fps, playlist_age_ms=None, segment_interval_ms=None):
    extras = [f"HLS输入帧率: {actual_fps:.2f}"]
    if playlist_age_ms is not None:
        extras.append(f"播放列表延迟: {int(playlist_age_ms)}ms")
    if segment_interval_ms is not None:
        extras.append(f"分片间隔: {int(segment_interval_ms)}ms")
    get_metrics_logger().info("【HLS】预览链路，%s", "，".join(extras))

