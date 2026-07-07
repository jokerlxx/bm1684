from __future__ import annotations

import json
import logging
from pathlib import Path

import pytz

from app.model_runtime.model_type_registry import validate_model_types_config


logger = logging.getLogger("StreamService")

BEIJING_TZ = pytz.timezone("Asia/Shanghai")
VIDEO_SOURCE_TYPE_RTSP = "rtsp"
VIDEO_SOURCE_TYPE_FILE = "file"
YOLO26_MODEL_ROOT = Path("/data/root_models_20260422")

_YOLO26_ALIAS_PATHS = {
    "fall": Path("fall_yolo26n_320/fall_yolo26n_320_int8_mix_1b.bmodel"),
    "fight": Path("fight_yolo26n_320/fight_yolo26n_320_int8_mix_1b.bmodel"),
    "helmet": Path("helmet_yolo26n_320/helmet_yolo26n_320_int8_mix_1b.bmodel"),
    "huxiji": Path("huxiji_yolo26n_320/huxiji_yolo26n_320_int8_mix_1b.bmodel"),
    "windows": Path("/home/linaro/windows_yolo26n_320/windows_yolo26n_320_int8_mix_1b.bmodel"),
}

# Window/door now has a dedicated YOLO26 export; the remaining shared slots still
# fall back to the closest available task model.
_YOLO26_MODEL_ALIASES = {
    "fall_detection": "fall",
    "ventilator_equipment": "huxiji",
    "ventilator_helmet": "helmet",
    "fight_detection": "fight",
    "crowd_person": "helmet",
    "helmet_detection": "helmet",
    "window_door_inside": "windows",
    "window_door_outside": "windows",
}


def _quantized_yolo26_models():
    return {
        model_key: str(YOLO26_MODEL_ROOT / _YOLO26_ALIAS_PATHS[alias])
        for model_key, alias in _YOLO26_MODEL_ALIASES.items()
    }


def _quantized_yolo26_model_types():
    return {model_key: "yolo26_int8" for model_key in _YOLO26_MODEL_ALIASES}


def _default_config():
    config = {
        "fps": 25,
        "models": _quantized_yolo26_models(),
        "model_types": _quantized_yolo26_model_types(),
        "output": {
            "video_output_dir": "./alarm_videos",
            "alarm_retention_days": 7,
            "display_port": 5000,
            "font_path": "simhei.ttf",
            "preview_transport": "mjpeg",
            "preview_fps": 20,
            "preview_fps_strict": True,
            "preview_encoder": "auto",
            "preview_hls_dir": "runtime/preview_hls",
            "preview_hls_segment_seconds": 1,
            "preview_hls_playlist_size": 3,
            "preview_result_ttl_s": 1.0,
            "preview_result_max_frame_lag": 20,
            "preview_alert_box_hold_s": 1.2,
            "preview_alert_box_tracking": True,
            "preview_alert_box_prediction_max_s": 0.0,
            "preview_alert_box_detection_lag_compensation": False,
            "preview_alert_box_track_max_shift_ratio": 0.35,
            "preview_alert_box_predict_max_shift_ratio": 1.5,
            "preview_max_alert_boxes_per_stream": 0,
            "preview_max_alert_labels_per_stream": 0,
            "alarm_buffer_fps": 8,
            "preview_merged_simple_alert_boxes": True,
            "preview_mjpeg_quality": 90,
            "preview_alert_debug_log_interval_s": 1.0,
            "preview_raw_model_boxes_enabled": False,
            "preview_raw_model_boxes_min_conf": 0.0,
        },
        "queue_sizes": {
            "frame_queue": 60,
            "result_queue": 10,
            "display_queue": 5,
        },
        "bm1684x": {
            "device_id": 0,
            "enable_sophon": True,
            "decode_backend": "auto",
            "scale_backend": "cv2",
            "detect_emit_fps": 2,
            "preview_source_fps": 20,
            "detect_frame_max_width": 0,
            "detect_frame_max_height": 0,
            "detect_slot_max_width": 1920,
            "detect_slot_max_height": 1080,
            "preview_frame_width": 0,
            "preview_frame_height": 0,
        },
    }
    validate_model_types_config(config)
    return config


def apply_output_defaults(config):
    output = config.setdefault("output", {})
    output.setdefault("video_output_dir", "./alarm_videos")
    output.setdefault("alarm_retention_days", 7)
    output.setdefault("display_port", 5000)
    output.setdefault("font_path", "simhei.ttf")
    output.setdefault("preview_transport", "mjpeg")
    output.setdefault("preview_fps", 20)
    output.setdefault("preview_fps_strict", True)
    output.setdefault("preview_encoder", "auto")
    output.setdefault("preview_hls_dir", "runtime/preview_hls")
    output.setdefault("preview_hls_segment_seconds", 1)
    output.setdefault("preview_hls_playlist_size", 3)
    output.setdefault("preview_result_ttl_s", 1.0)
    output.setdefault("preview_result_max_frame_lag", 20)
    output.setdefault("preview_alert_box_hold_s", 1.2)
    output.setdefault("preview_alert_box_tracking", True)
    output.setdefault("preview_alert_box_prediction_max_s", 0.0)
    output.setdefault("preview_alert_box_detection_lag_compensation", False)
    output.setdefault("preview_alert_box_track_max_shift_ratio", 0.35)
    output.setdefault("preview_alert_box_predict_max_shift_ratio", 1.5)
    output.setdefault("preview_max_alert_boxes_per_stream", 0)
    output.setdefault("preview_max_alert_labels_per_stream", 0)
    output.setdefault("alarm_buffer_fps", 8)
    output.setdefault("preview_merged_simple_alert_boxes", True)
    output.setdefault("preview_mjpeg_quality", 90)
    output.setdefault("preview_alert_debug_log_interval_s", 1.0)
    output.setdefault("preview_raw_model_boxes_enabled", False)
    output.setdefault("preview_raw_model_boxes_min_conf", 0.0)
    bm = config.setdefault("bm1684x", {})
    bm.setdefault("device_id", 0)
    bm.setdefault("enable_sophon", True)
    bm.setdefault("decode_backend", "auto")
    bm.setdefault("scale_backend", "cv2")
    bm.setdefault("detect_emit_fps", 2)
    bm.setdefault("preview_source_fps", output.get("preview_fps", 20))
    bm.setdefault("detect_frame_max_width", 0)
    bm.setdefault("detect_frame_max_height", 0)
    bm.setdefault("detect_slot_max_width", 1920)
    bm.setdefault("detect_slot_max_height", 1080)
    bm.setdefault("preview_frame_width", 0)
    bm.setdefault("preview_frame_height", 0)
    return config


def load_config(config_file="config_bm1684x.json"):
    try:
        with open(config_file, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        apply_output_defaults(config)
        validate_model_types_config(config)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_file)
        logger.info("Using default configuration")
        return _default_config()
    except Exception:
        logger.exception("Failed to load config")
        raise


def save_config(config, config_file="config_bm1684x.json"):
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def normalize_video_source_type(value):
    value = str(value or "").strip().lower()
    if value in ("file", "video", "video_file", "local", "local_file", "path"):
        return VIDEO_SOURCE_TYPE_FILE
    return VIDEO_SOURCE_TYPE_RTSP


def normalize_video_stream_entry(item, index=0):
    if not isinstance(item, dict):
        return None

    name = str(item.get("name") or f"通道 {index + 1}").strip() or f"通道 {index + 1}"
    raw_source = item.get("source")
    if raw_source in (None, ""):
        raw_source = item.get("ip")
    if raw_source in (None, ""):
        raw_source = item.get("path")
    source = str(raw_source or "").strip()

    raw_source_type = item.get("source_type")
    if raw_source_type in (None, "") and item.get("input_mode") is not None:
        try:
            raw_source_type = VIDEO_SOURCE_TYPE_FILE if int(item.get("input_mode")) == 1 else VIDEO_SOURCE_TYPE_RTSP
        except Exception:
            raw_source_type = None
    if raw_source_type in (None, "") and item.get("path") and not item.get("ip"):
        raw_source_type = VIDEO_SOURCE_TYPE_FILE
    source_type = normalize_video_source_type(raw_source_type)

    return {
        "name": name,
        "source_type": source_type,
        "source": source,
        "ip": source,
    }


def collect_video_stream_entries(config, max_streams=9):
    streams = (config or {}).get("video_streams")
    if not isinstance(streams, list) or len(streams) == 0:
        return [{"name": "通道 1", "source_type": VIDEO_SOURCE_TYPE_RTSP, "source": "", "ip": ""}]

    cleaned_streams = []
    for index, item in enumerate(streams[:max_streams]):
        normalized = normalize_video_stream_entry(item, index=index)
        if normalized is not None:
            cleaned_streams.append(normalized)

    if not cleaned_streams:
        return [{"name": "通道 1", "source_type": VIDEO_SOURCE_TYPE_RTSP, "source": "", "ip": ""}]
    return cleaned_streams


def summarize_video_source_modes(streams):
    types = {normalize_video_source_type((stream or {}).get("source_type")) for stream in (streams or [])}
    if not types or types == {VIDEO_SOURCE_TYPE_RTSP}:
        return "RTSP Stream"
    if types == {VIDEO_SOURCE_TYPE_FILE}:
        return "Video File (Loop)"
    return "Mixed (RTSP + Video File)"


def build_decode_source_specs(streams, max_streams=9):
    specs = []
    for stream in collect_video_stream_entries({"video_streams": streams}, max_streams=max_streams):
        specs.append(
            {
                "name": stream["name"],
                "source_type": stream["source_type"],
                "source": stream["source"],
            }
        )
    return specs


def resolve_config_path(config_path: str | Path = "config_bm1684x.json") -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path
