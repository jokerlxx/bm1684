from __future__ import annotations

import json
import logging
import multiprocessing as mp
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from core.frame_hub import FrameHubConfig, LatestFrameHub

from app.application.detector_registry import VALID_DETECTORS
from app.bootstrap.config import (
    BEIJING_TZ,
    VIDEO_SOURCE_TYPE_FILE,
    VIDEO_SOURCE_TYPE_RTSP,
    apply_output_defaults,
    build_decode_source_specs,
    collect_video_stream_entries,
    load_config,
    normalize_video_stream_entry,
    resolve_config_path,
    save_config,
    summarize_video_source_modes,
)
from app.infrastructure.logging import configure_root_logging, get_metrics_log_path
from app.pipeline.decode import run_decode_hub
from app.pipeline.display import run_display_service
from app.pipeline.entrypoints import (
    run_crowd_detector,
    run_fall_detector,
    run_fight_detector,
    run_helmet_detector,
    run_ventilator_detector,
    run_window_door_inside_detector,
    run_window_door_outside_detector,
)
from app.pipeline.model_scheduler import run_batch_model_scheduler


LOG_FILE_PATH = configure_root_logging(force=True)
METRIC_LOG_FILE_PATH = get_metrics_log_path()
logger = logging.getLogger("StreamService")
logger.info("Main logging to console and file: %s", LOG_FILE_PATH)
logger.info("Metric logging to file: %s", METRIC_LOG_FILE_PATH)

DEFAULT_STREAM_INDEX = 1
DEFAULT_STREAM_ID = 0
FIXED_PREVIEW_STREAM_COUNT = 4
MAX_TASKS = 9


def _fixed_stream_sources(config):
    streams = collect_video_stream_entries(config, max_streams=FIXED_PREVIEW_STREAM_COUNT)
    normalized = build_decode_source_specs(streams)
    while len(normalized) < FIXED_PREVIEW_STREAM_COUNT:
        channel = len(normalized) + 1
        normalized.append({"name": f"通道 {channel}", "source_type": VIDEO_SOURCE_TYPE_RTSP, "source": ""})
    return normalized[:FIXED_PREVIEW_STREAM_COUNT]


class SystemOrchestrator:
    """Application-level orchestrator responsible for process lifecycle."""

    def __init__(self, config):
        self.config = config
        self.running = False
        self.stream_count_estimate = self._estimate_stream_count(config)
        self.active_stream_count = self.stream_count_estimate
        self.input_fps = max(1, int(config.get("fps", 25)))
        self.output_cfg = apply_output_defaults(self.config).get("output", {})
        self.preview_transport = str(self.output_cfg.get("preview_transport", "local")).strip().lower() or "local"
        self.preview_fps = self._effective_output_fps(self.output_cfg.get("preview_fps", 20), self.stream_count_estimate)
        self.output_fps = self.preview_fps
        bm_cfg = apply_output_defaults(self.config).get("bm1684x", {})

        self.processes = {}
        self.decode_control_queue = mp.Queue()
        self.preview_frame_queues = [mp.Queue(maxsize=1) for _ in range(9)]
        self.frame_hub = LatestFrameHub(
            FrameHubConfig(
                stream_count=9,
                max_width=max(320, int(bm_cfg.get("detect_slot_max_width", 1920))),
                max_height=max(180, int(bm_cfg.get("detect_slot_max_height", 1080))),
                channels=3,
            )
        )
        self.result_queues = {
            "fall": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "ventilator": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "fight": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "crowd": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "helmet": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "window_door_inside": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
            "window_door_outside": mp.Queue(maxsize=config["queue_sizes"]["result_queue"]),
        }
        self.output_queue = mp.Queue(maxsize=config["queue_sizes"]["display_queue"])
        self.alert_queue = mp.Queue(maxsize=200)
        self._manager = mp.Manager()
        self.preview_status = self._manager.dict()
        self._set_preview_status(
            transport=self.preview_transport,
            healthy=False,
            target_fps=self.preview_fps,
            decoder_backend=None,
            encoder_backend=None,
            scale_backend=None,
            compose_fps=0.0,
            encode_in_fps=0.0,
            last_frame_ts=None,
            last_segment_ts=None,
            playlist_age_ms=None,
            unhealthy_reason=None,
        )
        self.control_queues = {
            "fall": mp.Queue(),
            "ventilator": mp.Queue(),
            "fight": mp.Queue(),
            "crowd": mp.Queue(),
            "helmet": mp.Queue(),
            "window_door_inside": mp.Queue(),
            "window_door_outside": mp.Queue(),
            "display": mp.Queue(),
        }
        self.batch_scheduler_enabled = bool((config.get("batch_scheduler") or {}).get("enabled", False))
        self.batch_control_queue = mp.Queue()
        self.detector_running = {name: False for name in VALID_DETECTORS}
        self.task_running = {}
        self.detector_streams = {}
        self._reload_stop_lock = threading.Lock()

    @staticmethod
    def _estimate_stream_count(config):
        del config
        return FIXED_PREVIEW_STREAM_COUNT

    @staticmethod
    def _effective_output_fps(configured_fps, stream_count):
        del stream_count
        return max(1, int(configured_fps))


    def _configured_stream_count(self):
        return FIXED_PREVIEW_STREAM_COUNT

    def _active_stream_count(self):
        return FIXED_PREVIEW_STREAM_COUNT

    def _validate_stream_index(self, stream_index):
        return 1 <= int(stream_index) <= FIXED_PREVIEW_STREAM_COUNT, FIXED_PREVIEW_STREAM_COUNT

    def _drain_queue(self, queue_obj, max_items=10000):
        drained = 0
        try:
            while drained < max_items:
                queue_obj.get_nowait()
                drained += 1
        except Exception:
            pass
        return drained

    def _preview_config(self):
        output = apply_output_defaults(self.config).get("output", {})
        bm_cfg = apply_output_defaults(self.config).get("bm1684x", {})
        return {
            "transport": str(output.get("preview_transport", "local")).strip().lower() or "local",
            "fps": max(1, int(output.get("preview_fps", 20))),
            "strict": bool(output.get("preview_fps_strict", True)),
            "encoder": str(output.get("preview_encoder", "auto")).strip() or "auto",
            "hls_dir": str(output.get("preview_hls_dir", "runtime/preview_hls")).strip() or "runtime/preview_hls",
            "hls_segment_seconds": max(1, int(output.get("preview_hls_segment_seconds", 1))),
            "hls_playlist_size": max(2, int(output.get("preview_hls_playlist_size", 3))),
            "result_ttl_s": float(output.get("preview_result_ttl_s", 1.0)),
            "result_max_frame_lag": int(output.get("preview_result_max_frame_lag", max(3, int(max(1, int(output.get("preview_fps", 20))) * 1.0)))),
            "alert_box_hold_s": float(output.get("preview_alert_box_hold_s", 1.2)),
            "alert_box_tracking": bool(output.get("preview_alert_box_tracking", True)),
            "alert_box_prediction_max_s": float(output.get("preview_alert_box_prediction_max_s", 0.0)),
            "alert_box_detection_lag_compensation": bool(output.get("preview_alert_box_detection_lag_compensation", False)),
            "alert_box_track_max_shift_ratio": float(output.get("preview_alert_box_track_max_shift_ratio", 0.35)),
            "alert_box_predict_max_shift_ratio": float(output.get("preview_alert_box_predict_max_shift_ratio", 1.5)),
            "max_alert_boxes_per_stream": int(output.get("preview_max_alert_boxes_per_stream", 0)),
            "max_alert_labels_per_stream": int(
                output.get(
                    "preview_max_alert_labels_per_stream",
                    output.get("preview_max_alert_boxes_per_stream", 0),
                )
            ),
            "alarm_buffer_fps": int(output.get("alarm_buffer_fps", 8)),
            "merged_simple_alert_boxes": bool(output.get("preview_merged_simple_alert_boxes", True)),
            "mjpeg_quality": int(output.get("preview_mjpeg_quality", 55)),
            "alert_debug_log_interval_s": float(output.get("preview_alert_debug_log_interval_s", 1.0)),
            "raw_model_boxes_enabled": bool(output.get("preview_raw_model_boxes_enabled", False)),
            "raw_model_boxes_min_conf": float(output.get("preview_raw_model_boxes_min_conf", 0.0)),
            "decode_backend": str(bm_cfg.get("decode_backend", "auto")).strip().lower() or "auto",
            "scale_backend": str(bm_cfg.get("scale_backend", "cv2")).strip().lower() or "cv2",
        }

    def _preview_output_dir(self):
        preview_cfg = self._preview_config()
        path = Path(preview_cfg["hls_dir"]).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path)

    def _set_preview_status(self, **kwargs):
        if not hasattr(self, "preview_status") or self.preview_status is None:
            return
        for key, value in kwargs.items():
            self.preview_status[key] = value

    def get_preview_status(self):
        try:
            return dict(self.preview_status)
        except Exception:
            return {
                "transport": self.preview_transport,
                "healthy": False,
                "target_fps": self.preview_fps,
                "decoder_backend": None,
                "encoder_backend": None,
                "scale_backend": None,
                "compose_fps": 0.0,
                "encode_in_fps": 0.0,
                "last_frame_ts": None,
                "last_segment_ts": None,
                "playlist_age_ms": None,
                "unhealthy_reason": None,
            }

    def get_preview_hls_dir(self):
        return self._preview_output_dir()

    def start(self):
        logger.info("Starting layered detection system")
        try:
            video_sources = _fixed_stream_sources(self.config)
            stream_count = FIXED_PREVIEW_STREAM_COUNT
            self.active_stream_count = stream_count
            input_mode_desc = summarize_video_source_modes(video_sources)

            self._drain_queue(self.decode_control_queue)
            self._drain_queue(self.control_queues["display"])
            for queue_obj in self.preview_frame_queues:
                self._drain_queue(queue_obj)

            preview_cfg = self._preview_config()
            self.processes["decode"] = mp.Process(
                target=run_decode_hub,
                args=(
                    video_sources[:stream_count],
                    self.preview_frame_queues[:stream_count],
                    self.frame_hub,
                    self.decode_control_queue,
                    self.input_fps,
                    self.preview_fps,
                    0,
                    preview_cfg,
                    self.preview_status,
                    self.config.get("bm1684x", {}),
                ),
                daemon=True,
            )
            self.processes["decode"].start()
            time.sleep(2)

            out_cfg = self.config.get("output", {})
            font_path = out_cfg.get("font_path", "simhei.ttf")
            alert_box_mode = str(out_cfg.get("alert_box_mode", "follow")).strip().lower()
            if alert_box_mode not in ("blink", "follow"):
                alert_box_mode = "follow"
            self.processes["display"] = mp.Process(
                target=run_display_service,
                args=(
                    self.preview_frame_queues[:stream_count],
                    self.result_queues,
                    self.output_queue,
                    self.control_queues["display"],
                    out_cfg["video_output_dir"],
                    self.output_fps,
                    stream_count,
                    self.alert_queue,
                    font_path,
                    self.preview_status,
                    preview_cfg,
                    alert_box_mode,
                    max(1, int(out_cfg.get("alarm_retention_days", 7))),
                ),
                daemon=True,
            )
            self.processes["display"].start()
            self.running = True
            logger.info("Core services started. Mode=%s streams=%s", input_mode_desc, stream_count)
        except Exception:
            self.stop()
            raise

    def start_detector(self, detector_name, enabled_streams=None):
        if detector_name not in VALID_DETECTORS:
            return False
        enabled_streams = {int(x) for x in (enabled_streams or {DEFAULT_STREAM_ID}) if 0 <= int(x) < FIXED_PREVIEW_STREAM_COUNT}
        if not enabled_streams:
            enabled_streams = {DEFAULT_STREAM_ID}
        if self.batch_scheduler_enabled:
            return self._start_detector_batch(detector_name, enabled_streams=enabled_streams)
        if detector_name in self.processes and self.processes[detector_name].is_alive():
            self.detector_streams[detector_name] = set(enabled_streams)
            return True
        self._drain_queue(self.control_queues[detector_name])
        try:
            if detector_name == "fall":
                process = mp.Process(
                    target=run_fall_detector,
                    args=(self.config["models"]["fall_detection"], self.frame_hub, self.result_queues["fall"], self.control_queues["fall"], self.config),
                    daemon=True,
                )
            elif detector_name == "ventilator":
                process = mp.Process(
                    target=run_ventilator_detector,
                    args=(
                        self.config["models"]["ventilator_equipment"],
                        self.config["models"]["ventilator_helmet"],
                        self.frame_hub,
                        self.result_queues["ventilator"],
                        self.control_queues["ventilator"],
                        self.config,
                    ),
                    daemon=True,
                )
            elif detector_name == "fight":
                process = mp.Process(
                    target=run_fight_detector,
                    args=(self.config["models"]["fight_detection"], self.frame_hub, self.result_queues["fight"], self.control_queues["fight"], self.config),
                    daemon=True,
                )
            elif detector_name == "crowd":
                process = mp.Process(
                    target=run_crowd_detector,
                    args=(self.config["models"]["crowd_person"], self.frame_hub, self.result_queues["crowd"], self.control_queues["crowd"], self.config),
                    daemon=True,
                )
            elif detector_name == "helmet":
                process = mp.Process(
                    target=run_helmet_detector,
                    args=(self.config["models"]["helmet_detection"], self.frame_hub, self.result_queues["helmet"], self.control_queues["helmet"], self.config),
                    daemon=True,
                )
            elif detector_name == "window_door_inside":
                process = mp.Process(
                    target=run_window_door_inside_detector,
                    args=(
                        self.config["models"].get("window_door_inside") or self.config["models"].get("window_door_detection"),
                        self.frame_hub,
                        self.result_queues["window_door_inside"],
                        self.control_queues["window_door_inside"],
                        self.config,
                    ),
                    daemon=True,
                )
            elif detector_name == "window_door_outside":
                process = mp.Process(
                    target=run_window_door_outside_detector,
                    args=(
                        self.config["models"].get("window_door_outside") or self.config["models"].get("window_door_detection"),
                        self.frame_hub,
                        self.result_queues["window_door_outside"],
                        self.control_queues["window_door_outside"],
                        self.config,
                    ),
                    daemon=True,
                )
            else:
                return False
            process.start()
            self.processes[detector_name] = process
            self.detector_running[detector_name] = True
            time.sleep(1.0)
            if enabled_streams:
                self.control_queues[detector_name].put({"cmd": "set_streams", "stream_ids": list(enabled_streams)})
            self.control_queues[detector_name].put("enable")
            time.sleep(0.2)
            return True
        except Exception:
            logger.exception("Failed to start detector %s", detector_name)
            return False

    def _ensure_batch_scheduler(self):
        process = self.processes.get("model_scheduler")
        if process and process.is_alive():
            return True
        self._drain_queue(self.batch_control_queue)
        process = mp.Process(
            target=run_batch_model_scheduler,
            args=(self.frame_hub, self.result_queues, self.batch_control_queue, self.config, self.preview_status),
            daemon=True,
        )
        process.start()
        self.processes["model_scheduler"] = process
        time.sleep(0.5)
        return True

    def _start_detector_batch(self, detector_name, enabled_streams=None):
        if detector_name not in VALID_DETECTORS:
            return False
        try:
            self._ensure_batch_scheduler()
            streams = {int(x) for x in (enabled_streams or {DEFAULT_STREAM_ID}) if 0 <= int(x) < FIXED_PREVIEW_STREAM_COUNT}
            if not streams:
                streams = {DEFAULT_STREAM_ID}
            self.detector_streams[detector_name] = streams
            self.batch_control_queue.put(
                {
                    "cmd": "enable",
                    "detector": detector_name,
                    "stream_ids": list(self.detector_streams.get(detector_name, [])),
                }
            )
            self.detector_running[detector_name] = True
            return True
        except Exception:
            logger.exception("Failed to start detector %s in batch scheduler", detector_name)
            return False

    def stop_detector(self, detector_name):
        if self.batch_scheduler_enabled:
            return self._stop_detector_batch(detector_name)
        if detector_name not in self.processes or not self.processes[detector_name].is_alive():
            return False
        try:
            disabled_result = {
                "detector_type": detector_name,
                "enabled": False,
                "timestamp": datetime.now(),
                "detections": [],
                "display_alerts": [],
                "clear_all": True,
            }
            while not self.result_queues[detector_name].empty():
                self.result_queues[detector_name].get_nowait()
            for _ in range(3):
                self.result_queues[detector_name].put(disabled_result, block=False)
        except Exception:
            pass
        try:
            self.control_queues[detector_name].put("stop")
        except Exception:
            pass
        process = self.processes[detector_name]
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join()
        self.detector_running[detector_name] = False
        self._drain_queue(self.control_queues[detector_name])
        return True

    def _stop_detector_batch(self, detector_name):
        if detector_name not in VALID_DETECTORS:
            return False
        try:
            disabled_result = {
                "detector_type": detector_name,
                "enabled": False,
                "timestamp": datetime.now(),
                "detections": [],
                "display_alerts": [],
                "clear_all": True,
            }
            while not self.result_queues[detector_name].empty():
                self.result_queues[detector_name].get_nowait()
            self.result_queues[detector_name].put(disabled_result, block=False)
        except Exception:
            pass
        try:
            self.batch_control_queue.put({"cmd": "disable", "detector": detector_name})
        except Exception:
            pass
        self.detector_running[detector_name] = False
        self.detector_streams.pop(detector_name, None)
        return True

    def _stop_other_detectors(self, keep_detector):
        for detector_name in list(VALID_DETECTORS):
            if detector_name == keep_detector:
                continue
            if self.detector_running.get(detector_name) or detector_name in self.detector_streams:
                self.stop_detector(detector_name)
        for task_id in list(self.task_running):
            self.task_running[task_id] = False

    def get_task_running_states(self):
        return dict(self.task_running)

    def reload_algorithm_config(self, config=None):
        if config is not None:
            self.config = config
        if self.batch_scheduler_enabled:
            try:
                self.batch_control_queue.put({"cmd": "reload_config", "config": self.config})
            except Exception:
                return False
            return True
        ok = True
        for detector_name in list(VALID_DETECTORS):
            if not self.detector_running.get(detector_name):
                continue
            try:
                self.control_queues[detector_name].put({"cmd": "reload_config", "config": self.config})
            except Exception:
                ok = False
        return ok

    def start_task(self, task_id):
        """启动任务：为该任务的所有检测器添加对应通道，各任务互不干扰。

        多个任务可以同时运行在不同通道上，同一检测器可以被多个任务共享
        （聚合所有任务的通道集合）。
        """
        tasks = self.config.get("tasks") or []
        task = next((item for item in tasks if isinstance(item, dict) and str(item.get("id")) == str(task_id)), None)
        if not task:
            return {"status": "error", "message": "任务不存在"}
        stream_index = int(task.get("stream_index", DEFAULT_STREAM_INDEX))
        valid_stream, active_count = self._validate_stream_index(stream_index)
        if not valid_stream:
            return {"status": "error", "message": f"任务通道 {stream_index} 未配置视频流，当前仅有 {active_count} 路"}
        stream_id = stream_index - 1
        detectors = [item for item in (task.get("detectors") or []) if item in VALID_DETECTORS]
        if not detectors:
            return {"status": "error", "message": "任务至少需要配置一个检测器"}

        started = []
        for detector_name in detectors:
            # 聚合通道：将该任务的通道加入检测器的通道集合
            current_streams = set(self.detector_streams.get(detector_name, set()))
            current_streams.add(stream_id)
            self.detector_streams[detector_name] = current_streams

            if self.start_detector(detector_name, enabled_streams=current_streams):
                started.append(detector_name)
                continue
            # 启动失败，回滚
            for started_detector in started:
                self.detector_streams[started_detector].discard(stream_id)
                if not self.detector_streams[started_detector]:
                    self.stop_detector(started_detector)
            self.detector_streams[detector_name].discard(stream_id)
            if not self.detector_streams.get(detector_name):
                self.detector_streams.pop(detector_name, None)
            self.task_running[str(task_id)] = False
            return {"status": "error", "message": f"{detector_name} 检测器启动失败"}

        self.task_running[str(task_id)] = True
        return {"status": "success", "message": "任务已启动"}

    def stop_task(self, task_id):
        """停止任务：只移除该任务对应的通道，不影响其他任务使用的检测器。"""
        tasks = self.config.get("tasks") or []
        task = next((item for item in tasks if isinstance(item, dict) and str(item.get("id")) == str(task_id)), None)
        if not task:
            return {"status": "error", "message": "任务不存在"}
        stream_index = int(task.get("stream_index", 1))
        stream_id = stream_index - 1 if 1 <= stream_index <= 9 else 0
        detectors = [item for item in (task.get("detectors") or []) if item in VALID_DETECTORS]
        for detector_name in detectors:
            if detector_name not in self.detector_streams:
                continue
            self.detector_streams[detector_name].discard(stream_id)
            remaining = self.detector_streams[detector_name]
            if not remaining:
                # 没有其他任务使用此检测器，停止它
                self.detector_streams.pop(detector_name, None)
                self.stop_detector(detector_name)
            elif self.batch_scheduler_enabled:
                # 更新通道集合（移除该任务的通道）
                self.batch_control_queue.put(
                    {"cmd": "set_streams", "detector": detector_name, "stream_ids": list(remaining)}
                )
                self.control_queues["display"].put_nowait(
                    {"cmd": "clear_sticky", "detector": detector_name, "stream_id": stream_id}
                )
            elif detector_name in self.processes and self.processes[detector_name].is_alive():
                self.control_queues[detector_name].put(
                    {"cmd": "set_streams", "stream_ids": list(remaining)}
                )
                self.control_queues["display"].put_nowait(
                    {"cmd": "clear_sticky", "detector": detector_name, "stream_id": stream_id}
                )
        self.task_running[str(task_id)] = False
        return {"status": "success", "message": "任务已停止"}

    def reload_streams(self, config=None):
        with self._reload_stop_lock:
            if not self.running:
                if config is not None:
                    self.config = config
                return
            if config is not None:
                self.config = config
            video_sources = _fixed_stream_sources(self.config)
            stream_count = FIXED_PREVIEW_STREAM_COUNT
            self.active_stream_count = stream_count

            for key in list(self.processes.keys()):
                if key not in ("decode", "display"):
                    continue
                process = self.processes.get(key)
                if process and process.is_alive():
                    try:
                        if key == "decode":
                            self.decode_control_queue.put("stop")
                        else:
                            self.control_queues["display"].put("stop")
                    except Exception:
                        pass
                    process.terminate()
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join()
                self.processes.pop(key, None)
            time.sleep(1)

            self._drain_queue(self.decode_control_queue)
            self._drain_queue(self.control_queues["display"])
            for queue_obj in self.preview_frame_queues:
                self._drain_queue(queue_obj)
            self._drain_queue(self.output_queue)
            for stream_id in range(9):
                self.frame_hub.clear_stream(stream_id)

            preview_cfg = self._preview_config()
            self.processes["decode"] = mp.Process(
                target=run_decode_hub,
                args=(
                    video_sources[:stream_count],
                    self.preview_frame_queues[:stream_count],
                    self.frame_hub,
                    self.decode_control_queue,
                    self.config["fps"],
                    preview_cfg["fps"],
                    0,
                    preview_cfg,
                    self.preview_status,
                    self.config.get("bm1684x", {}),
                ),
                daemon=True,
            )
            self.processes["decode"].start()
            time.sleep(1)
            out_cfg = self.config.get("output", {})
            self.processes["display"] = mp.Process(
                target=run_display_service,
                args=(
                    self.preview_frame_queues[:stream_count],
                    self.result_queues,
                    self.output_queue,
                    self.control_queues["display"],
                    self.config["output"]["video_output_dir"],
                    preview_cfg["fps"],
                    stream_count,
                    self.alert_queue,
                    out_cfg.get("font_path", "simhei.ttf"),
                    self.preview_status,
                    preview_cfg,
                    str(out_cfg.get("alert_box_mode", "follow")).strip().lower(),
                    max(1, int(out_cfg.get("alarm_retention_days", 7))),
                ),
                daemon=True,
            )
            self.processes["display"].start()

    def stop(self):
        with self._reload_stop_lock:
            self.running = False
            self._set_preview_status(
                transport=self.preview_transport,
                healthy=False,
                target_fps=self.preview_fps,
                decoder_backend=None,
                encoder_backend=None,
                scale_backend=None,
                compose_fps=0.0,
                encode_in_fps=0.0,
                last_frame_ts=None,
                last_segment_ts=None,
                playlist_age_ms=None,
                unhealthy_reason=None,
            )
            try:
                self.decode_control_queue.put("stop")
            except Exception:
                pass
            try:
                self.batch_control_queue.put("stop")
            except Exception:
                pass
            for queue_obj in list(self.control_queues.values()):
                try:
                    queue_obj.put("stop")
                except Exception:
                    pass
            time.sleep(1)
            for name, process in list(self.processes.items()):
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join()
            self.processes.clear()
            for stream_id in range(9):
                self.frame_hub.clear_stream(stream_id)
            for key in list(self.detector_running.keys()):
                self.detector_running[key] = False

    def close(self):
        try:
            self.frame_hub.close()
        except Exception:
            pass
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def get_status(self):
        status = {"system_running": self.running, "detectors": {}, "preview": self.get_preview_status()}
        for name in VALID_DETECTORS:
            if self.batch_scheduler_enabled:
                scheduler = self.processes.get("model_scheduler")
                is_running = bool(scheduler and scheduler.is_alive() and self.detector_running[name])
                pid = scheduler.pid if is_running else None
            else:
                is_running = name in self.processes and self.processes[name].is_alive() and self.detector_running[name]
                pid = self.processes[name].pid if is_running else None
            status["detectors"][name] = {"running": is_running, "pid": pid}
        return status


class SystemController:
    def __init__(self, config_path: str | Path = "config_bm1684x.json"):
        self.config_path = resolve_config_path(config_path)
        self.config = load_config(str(self.config_path))
        self.scheduler: SystemOrchestrator | None = None

    def refresh_config(self):
        self.config = load_config(str(self.config_path))
        return self.config

    def start_system(self):
        try:
            if self.scheduler is not None and self.scheduler.running:
                return {"status": "error", "message": "系统已在运行中"}
            self.scheduler = SystemOrchestrator(self.config)
            self.scheduler.start()
            return {"status": "success", "message": "系统启动成功"}
        except Exception as exc:
            logger.error("Failed to start system: %s", exc)
            return {"status": "error", "message": str(exc)}

    def stop_system(self):
        try:
            if self.scheduler is None:
                return {"status": "error", "message": "系统未运行"}
            self.scheduler.stop()
            self.scheduler.close()
            self.scheduler = None
            return {"status": "success", "message": "系统停止成功"}
        except Exception as exc:
            logger.error("Failed to stop system: %s", exc)
            return {"status": "error", "message": str(exc)}

    def toggle_detector(self, data):
        try:
            if self.scheduler is None or not self.scheduler.running:
                return {"status": "error", "message": "系统未运行，请先启动系统"}
            detector = data.get("detector")
            enabled = data.get("enabled", True)
            if detector not in VALID_DETECTORS:
                return {"status": "error", "message": f"未知的检测器: {detector}"}
            success = self.scheduler.start_detector(detector, enabled_streams={DEFAULT_STREAM_ID}) if enabled else self.scheduler.stop_detector(detector)
            action = "启动" if enabled else "停止"
            if success:
                return {"status": "success", "message": f"{detector} detector {action}成功"}
            return {"status": "error", "message": f"{detector} detector {action}失败"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def get_status(self):
        if self.scheduler is None:
            preview = apply_output_defaults(self.config).get("output", {})
            return {
                "system_running": False,
                "detectors": {},
                "streams": [],
                "preview": {
                    "transport": str(preview.get("preview_transport", "local")).strip().lower() or "local",
                    "healthy": False,
                    "target_fps": max(1, int(preview.get("preview_fps", 20))),
                    "decoder_backend": None,
                    "encoder_backend": None,
                    "scale_backend": None,
                    "compose_fps": 0.0,
                    "encode_in_fps": 0.0,
                    "last_frame_ts": None,
                    "last_segment_ts": None,
                    "playlist_age_ms": None,
                    "unhealthy_reason": None,
                },
            }
        base = self.scheduler.get_status()
        streams = []
        try:
            stream_meta = self.scheduler.frame_hub.snapshot_meta()
            active_count = getattr(self.scheduler, "active_stream_count", len(stream_meta))
            for index, meta in enumerate(stream_meta[:active_count]):
                last_ts = meta.get("timestamp")
                streams.append(
                    {
                        "name": f"stream_{index}",
                        "running": bool(meta.get("connected")),
                        "pid": self.scheduler.processes.get("decode").pid if self.scheduler.processes.get("decode") else None,
                        "last_frame_ts": datetime.fromtimestamp(float(last_ts), tz=BEIJING_TZ).isoformat() if last_ts else None,
                    }
                )
        except Exception:
            pass
        base["streams"] = streams
        return base

    def get_alerts_history(self):
        try:
            from shutil import disk_usage

            from app.infrastructure.storage import cleanup_old_alarm_files, get_alarm_output_dir, list_alarm_files

            output_dir = get_alarm_output_dir(self.config.get("output"))
            retention_days = max(1, int(self.config.get("output", {}).get("alarm_retention_days", 7)))
            cleanup_old_alarm_files(output_dir, max_age_days=retention_days)
            items = list_alarm_files(output_dir, limit=500)
            total_size = sum((item.get("size") or 0) for item in items)
            disk_total = disk_used = disk_free = None
            try:
                usage = disk_usage(output_dir)
                disk_total = usage.total
                disk_used = usage.used
                disk_free = usage.free
            except Exception:
                pass
            return {
                "status": "success",
                "items": items,
                "output_dir": output_dir,
                "total_size": total_size,
                "disk_total": disk_total,
                "disk_used": disk_used,
                "disk_free": disk_free,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc), "items": [], "output_dir": ""}

    def get_alerts_cleanup(self, request=None):
        try:
            from app.infrastructure.storage import cleanup_old_alarm_files, get_alarm_output_dir

            output_dir = get_alarm_output_dir(self.config.get("output"))
            max_days = max(1, int(self.config.get("output", {}).get("alarm_retention_days", 7)))
            if request:
                data = request.get_json(silent=True)
                if isinstance(data, dict) and isinstance(data.get("max_days"), (int, float)) and 1 <= data["max_days"] <= 365:
                    max_days = int(data["max_days"])
            deleted = cleanup_old_alarm_files(output_dir, max_age_days=max_days)
            return {"status": "success", "message": f"已删除 {deleted} 个过期文件", "deleted": deleted}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "deleted": 0}

    def get_alerts_clear(self, request=None):
        del request
        try:
            from app.infrastructure.storage import clear_all_alarm_files, get_alarm_output_dir

            output_dir = get_alarm_output_dir(self.config.get("output"))
            deleted = clear_all_alarm_files(output_dir)
            return {"status": "success", "message": f"已删除 {deleted} 个文件", "deleted": deleted}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "deleted": 0}

    def delete_alerts_batch(self, data):
        try:
            from storage import _is_alarm_file, get_alarm_output_dir

            output_dir = Path(get_alarm_output_dir(self.config.get("output")))
            names = data.get("names") if isinstance(data, dict) else None
            if not isinstance(names, list) or not names:
                return {"status": "error", "message": "缺少要删除的文件列表"}
            deleted = 0
            for name in names:
                name = (str(name) or "").strip()
                if not name or ".." in name or "/" in name or "\\" in name:
                    continue
                file_path = output_dir / name
                if file_path.is_file() and _is_alarm_file(file_path):
                    file_path.unlink()
                    deleted += 1
            return {"status": "success", "message": f"已删除 {deleted} 个文件", "deleted": deleted}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "deleted": 0}

    def get_alerts_file(self, request):
        from storage import get_alarm_output_dir

        name = (request.args.get("name") or "").strip()
        if not name or ".." in name or "/" in name or "\\" in name:
            return {"status": "error", "message": "无效文件名"}
        output_dir = get_alarm_output_dir(self.config.get("output"))
        output_path = Path(output_dir)
        file_path = output_path / name
        try:
            file_path = file_path.resolve()
            output_abs = output_path.resolve()
            if not file_path.is_file() or not str(file_path).startswith(str(output_abs)):
                return {"status": "error", "message": "文件不存在"}
            return (str(file_path), name)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def get_video_streams(self):
        try:
            streams = collect_video_stream_entries(self.config, max_streams=FIXED_PREVIEW_STREAM_COUNT)
            while len(streams) < FIXED_PREVIEW_STREAM_COUNT:
                channel = len(streams) + 1
                streams.append({"name": f"通道 {channel}", "source_type": VIDEO_SOURCE_TYPE_RTSP, "source": "", "ip": ""})
            return {"status": "success", "streams": streams[:FIXED_PREVIEW_STREAM_COUNT], "max_streams": FIXED_PREVIEW_STREAM_COUNT}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "streams": [], "max_streams": FIXED_PREVIEW_STREAM_COUNT}

    def save_video_streams(self, data):
        try:
            raw_streams = data.get("streams") or []
            if not isinstance(raw_streams, list):
                return {"status": "error", "message": "无效的视频流数据"}
            cleaned_streams = []
            for index, item in enumerate(raw_streams):
                if len(cleaned_streams) >= FIXED_PREVIEW_STREAM_COUNT:
                    break
                normalized = normalize_video_stream_entry(item, index=index)
                if normalized is None or not normalized["source"]:
                    continue
                cleaned_streams.append(normalized)
            if not cleaned_streams:
                return {"status": "error", "message": "至少需要配置一路有效的视频源地址或本地文件路径"}

            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            config["video_streams"] = cleaned_streams
            save_config(config, str(self.config_path))
            self.refresh_config()
            if self.scheduler is not None and getattr(self.scheduler, "running", False):
                self.scheduler.reload_streams(self.config)
            return {"status": "success", "message": "视频流配置已保存"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


    def _configured_stream_count(self):
        return FIXED_PREVIEW_STREAM_COUNT

    def _active_stream_count(self):
        if self.scheduler is not None and getattr(self.scheduler, "running", False):
            return FIXED_PREVIEW_STREAM_COUNT
        return self._configured_stream_count()

    def _validate_stream_index(self, stream_index):
        return 1 <= int(stream_index) <= FIXED_PREVIEW_STREAM_COUNT, FIXED_PREVIEW_STREAM_COUNT

    @staticmethod
    def _detectors_from_payload(data):
        detectors = [item for item in (data.get("detectors") or []) if item in VALID_DETECTORS]
        if not detectors:
            return None
        return list(dict.fromkeys(detectors))

    def get_tasks(self):
        try:
            tasks = self.config.get("tasks")
            if not isinstance(tasks, list):
                tasks = []
            tasks = [task for task in tasks if isinstance(task, dict)][:MAX_TASKS]
            for task in tasks:
                if isinstance(task, dict) and ("id" not in task or task["id"] is None):
                    task["id"] = "t_" + str(int(time.time() * 1000))
                stream_index = int(task.get("stream_index", DEFAULT_STREAM_INDEX) or DEFAULT_STREAM_INDEX)
                task["stream_index"] = min(max(1, stream_index), FIXED_PREVIEW_STREAM_COUNT)
                detectors = [item for item in (task.get("detectors") or []) if item in VALID_DETECTORS]
                task["detectors"] = list(dict.fromkeys(detectors))
            return {"status": "success", "tasks": tasks}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "tasks": []}

    def add_task(self, data):
        try:
            name = (data.get("name") or "").strip()
            if not name:
                return {"status": "error", "message": "任务名称不能为空"}
            stream_index = int(data.get("stream_index", DEFAULT_STREAM_INDEX))
            valid_stream, active_count = self._validate_stream_index(stream_index)
            if not valid_stream:
                return {"status": "error", "message": f"通道 {stream_index} 未配置视频流，当前仅有 {active_count} 路"}
            detectors = self._detectors_from_payload(data)
            if detectors is None:
                return {"status": "error", "message": "任务至少需要选择一个检测器"}
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            tasks = config.get("tasks")
            if not isinstance(tasks, list):
                tasks = []
            task_id = "t_" + str(int(time.time() * 1000))
            tasks.append({"id": task_id, "name": name, "stream_index": stream_index, "detectors": detectors})
            # 限制最大任务数
            if len(tasks) > MAX_TASKS:
                tasks = tasks[-MAX_TASKS:]
            config["tasks"] = tasks
            save_config(config, str(self.config_path))
            self.refresh_config()
            if self.scheduler is not None and getattr(self.scheduler, "running", False):
                self.scheduler.config = self.config
                reload_config = getattr(self.scheduler, "reload_algorithm_config", None)
                if callable(reload_config):
                    reload_config(self.config)
            return {"status": "success", "message": "任务已添加", "id": task_id}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def update_task(self, data):
        try:
            task_id = data.get("id")
            if not task_id:
                return {"status": "error", "message": "缺少任务 id"}
            scheduler_running = self.scheduler is not None and getattr(self.scheduler, "running", False)
            was_running = bool(
                scheduler_running
                and getattr(self.scheduler, "task_running", {}).get(str(task_id))
            )
            name = (data.get("name") or "").strip()
            if not name:
                return {"status": "error", "message": "任务名称不能为空"}
            stream_index = int(data.get("stream_index", DEFAULT_STREAM_INDEX))
            valid_stream, active_count = self._validate_stream_index(stream_index)
            if not valid_stream:
                return {"status": "error", "message": f"通道 {stream_index} 未配置视频流，当前仅有 {active_count} 路"}
            detectors = self._detectors_from_payload(data)
            if detectors is None:
                return {"status": "error", "message": "任务至少需要选择一个检测器"}
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            tasks = config.get("tasks")
            if not isinstance(tasks, list):
                return {"status": "error", "message": "无任务列表"}
            found = False
            updated_task = None
            for task in tasks:
                if isinstance(task, dict) and str(task.get("id")) == str(task_id):
                    task["name"] = name
                    task["stream_index"] = stream_index
                    task["detectors"] = detectors
                    updated_task = task
                    found = True
                    break
            if not found:
                return {"status": "error", "message": "任务不存在"}
            # 保留所有任务，仅更新匹配项（tasks 列表已在上面被原地修改）
            save_config(config, str(self.config_path))
            self.refresh_config()
            if scheduler_running:
                self.scheduler.config = self.config
                reload_config = getattr(self.scheduler, "reload_algorithm_config", None)
                if callable(reload_config):
                    reload_config(self.config)
                if was_running:
                    start_task = getattr(self.scheduler, "start_task", None)
                    if not callable(start_task):
                        return {"status": "error", "message": "任务已保存，但运行态热切换失败：系统调度器不可用"}
                    switch_result = start_task(task_id)
                    if not isinstance(switch_result, dict) or switch_result.get("status") != "success":
                        message = switch_result.get("message") if isinstance(switch_result, dict) else "未知错误"
                        return {"status": "error", "message": f"任务已保存，但运行态热切换失败：{message}"}
                    return {"status": "success", "message": "任务已更新，运行中的算法已切换"}
            return {"status": "success", "message": "任务已更新"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def delete_task(self, data):
        try:
            task_id = data.get("id")
            if not task_id:
                return {"status": "error", "message": "缺少任务 id"}
            if self.scheduler is not None and getattr(self.scheduler, "task_running", {}).get(str(task_id)):
                self.scheduler.stop_task(task_id)
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            tasks = config.get("tasks")
            if not isinstance(tasks, list):
                return {"status": "success", "message": "已删除"}
            config["tasks"] = [task for task in tasks if not (isinstance(task, dict) and str(task.get("id")) == str(task_id))]
            save_config(config, str(self.config_path))
            self.refresh_config()
            if self.scheduler is not None and getattr(self.scheduler, "running", False):
                self.scheduler.config = self.config
            return {"status": "success", "message": "已删除"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def handlers(self):
        return {
            "start": self.start_system,
            "stop": self.stop_system,
            "toggle_detector": self.toggle_detector,
            "status": self.get_status,
            "alerts_history": self.get_alerts_history,
            "alerts_file": self.get_alerts_file,
            "alerts_cleanup": self.get_alerts_cleanup,
            "alerts_clear": self.get_alerts_clear,
            "alerts_delete_batch": self.delete_alerts_batch,
            "video_streams_get": self.get_video_streams,
            "video_streams_save": self.save_video_streams,
            "tasks_get": self.get_tasks,
            "tasks_add": self.add_task,
            "tasks_update": self.update_task,
            "tasks_delete": self.delete_task,
        }

    def get_scheduler(self):
        return self.scheduler

    def get_alert_queue(self):
        return getattr(self.scheduler, "alert_queue", None) if self.scheduler else None

    def shutdown(self):
        if self.scheduler:
            self.scheduler.stop()
            self.scheduler.close()
            self.scheduler = None


def install_signal_handlers(controller: SystemController):
    def signal_handler(signum, frame):
        logger.info("Received signal %s, shutting down", signum)
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
