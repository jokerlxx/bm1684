"""
统一多路解码中心。

职责：
1. 统一管理多路 RTSP / 文件输入；
2. 为检测链路写入每路“最新原始帧”到 LatestFrameHub；
3. 为预览链路写入每路预览尺寸的最新帧到独立 mp.Queue(maxsize=1)。
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.frame_hub import LatestFrameHub
from core.logging_utils import ensure_root_logging, log_pull_fps


try:
    import sophon.sail as sail

    SOPHON_AVAILABLE = True
except ImportError:
    sail = None
    SOPHON_AVAILABLE = False


logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))
PREVIEW_CELL_W = 320
PREVIEW_CELL_H = 180
PREVIEW_SINGLE_W = 1280
PREVIEW_SINGLE_H = 720
SOURCE_TYPE_RTSP = "rtsp"
SOURCE_TYPE_FILE = "file"


@dataclass
class _ChannelStats:
    last_log_ts: float = 0.0
    fps_t0: float = 0.0
    fps_n: int = 0
    last_frame_ts: float = 0.0
    frame_number: int = 0


def normalize_source_type(value: Optional[str]) -> str:
    value = str(value or "").strip().lower()
    if value in ("file", "video", "video_file", "local", "local_file", "path"):
        return SOURCE_TYPE_FILE
    return SOURCE_TYPE_RTSP


def normalize_source_spec(source_entry, default_input_mode: int = 0) -> dict:
    raw_source_type = None
    raw_source = source_entry

    if isinstance(source_entry, dict):
        raw_source_type = source_entry.get("source_type")
        raw_source = source_entry.get("source")
        if raw_source in (None, ""):
            raw_source = source_entry.get("ip")
        if raw_source in (None, ""):
            raw_source = source_entry.get("path")
        if raw_source_type in (None, "") and source_entry.get("input_mode") is not None:
            try:
                raw_source_type = SOURCE_TYPE_FILE if int(source_entry.get("input_mode")) == 1 else SOURCE_TYPE_RTSP
            except Exception:
                raw_source_type = None
        if raw_source_type in (None, "") and source_entry.get("path") and not source_entry.get("ip"):
            raw_source_type = SOURCE_TYPE_FILE

    source = str(raw_source or "").strip()
    source_type = normalize_source_type(
        raw_source_type if raw_source_type not in (None, "") else (
            SOURCE_TYPE_FILE if int(default_input_mode) == 1 else SOURCE_TYPE_RTSP
        )
    )
    return {
        "source": source,
        "source_type": source_type,
        "input_mode": 1 if source_type == SOURCE_TYPE_FILE else 0,
    }


class PreviewScaler:
    def __init__(
        self,
        device_id: int = 0,
        requested_backend: str = "bmcv",
        target_width: int = PREVIEW_CELL_W,
        target_height: int = PREVIEW_CELL_H,
    ):
        self.device_id = int(device_id)
        self.requested_backend = str(requested_backend or "bmcv").strip().lower()
        self.target_width = max(1, int(target_width or PREVIEW_CELL_W))
        self.target_height = max(1, int(target_height or PREVIEW_CELL_H))
        self.actual_backend = "cv2"
        self.handle = None
        self.bmcv = None

        if self.requested_backend == "bmcv" and SOPHON_AVAILABLE:
            try:
                self.handle = sail.Handle(self.device_id)
                self.bmcv = sail.Bmcv(self.handle)
                self.actual_backend = "bmcv"
                logger.info("PreviewScaler using BMCV hardware resize")
            except Exception as exc:
                logger.warning("PreviewScaler BMCV init failed, fallback to cv2: %s", exc)
                self.handle = None
                self.bmcv = None

    def _as_preview_mat(self, bmimg) -> np.ndarray:
        resized = self.bmcv.resize(bmimg, self.target_width, self.target_height)
        resized_mat = resized.asmat()
        if resized_mat is None:
            raise RuntimeError("bmcv.resize returned empty frame")
        return resized_mat

    def resize_bmimage(self, bmimg) -> np.ndarray:
        if self.actual_backend == "bmcv" and self.bmcv is not None:
            try:
                return self._as_preview_mat(bmimg)
            except Exception as exc:
                logger.warning("PreviewScaler bmcv resize BMImage failed, fallback to asmat+cv2: %s", exc)
        frame = bmimg.asmat()
        if frame is None:
            raise RuntimeError("BMImage.asmat returned empty frame")
        return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        if self.actual_backend != "bmcv" or self.bmcv is None:
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        try:
            height, width = frame.shape[:2]
            bmimg = sail.BMImage(
                self.handle,
                height,
                width,
                sail.Format.FORMAT_BGR_PACKED,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
            self.bmcv.mat_to_bm_image(frame, bmimg)
            return self._as_preview_mat(bmimg)
        except Exception as exc:
            logger.warning("PreviewScaler bmcv resize ndarray failed, fallback to cv2: %s", exc)
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)


class _DecoderPoolChannel:
    def __init__(self, source, stream_id: int, input_mode: int, fps: int, device_id: int):
        spec = normalize_source_spec(source, default_input_mode=input_mode)
        self.source_spec = spec
        self.source = spec["source"]
        self.stream_id = int(stream_id)
        self.input_mode = int(spec["input_mode"])
        self.fps = int(fps)
        self.device_id = int(device_id)
        self.is_video_file = self.input_mode == 1
        self.video_loop = self.is_video_file
        self.decoder = None
        self.cap = None
        self.handle = sail.Handle(self.device_id) if SOPHON_AVAILABLE else None
        self.connected = False
        self.last_reconnect_ts = 0.0
        self.last_frame_number = 0

    def connect(self):
        self.release()
        self.connected = False
        if not self.source:
            return False

        if SOPHON_AVAILABLE:
            try:
                self.decoder = sail.Decoder(self.source, True, self.device_id)
                if self.decoder.is_opened():
                    self.connected = True
                    return True
                self.decoder.release()
                self.decoder = None
            except Exception as exc:
                logger.warning("DecoderPool stream %s sail.Decoder open failed: %s", self.stream_id, exc)
                self.decoder = None

        if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = True
        return True

    def reconnect(self):
        if self.is_video_file:
            return self.connect()
        now = time.monotonic()
        if now - self.last_reconnect_ts < 1.0:
            return False
        self.last_reconnect_ts = now
        logger.warning("DecodeHub reconnect stream %s", self.stream_id)
        return self.connect()

    def _read_from_decoder_once(self):
        try:
            bmimg = sail.BMImage()
            ret = self.decoder.read(self.handle, bmimg)
            if ret == 0:
                frame = bmimg.asmat()
                if frame is not None:
                    self.connected = True
                    self.last_frame_number += 1
                    return frame
        except Exception as exc:
            logger.warning("DecodeHub stream %s sail read failed: %s", self.stream_id, exc)
        self.connected = False
        return None

    def _read_from_cap_once(self):
        if self.cap is None:
            self.connected = False
            return None

        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            self.connected = False
            return None

        self.connected = True
        self.last_frame_number += 1
        return frame

    def _restart_video_file(self):
        if not self.is_video_file or not self.video_loop:
            return False

        logger.info("DecodeHub stream %s video file ended, restarting from beginning", self.stream_id)
        if self.decoder is not None:
            return self.connect()
        if self.cap is not None:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True
            except Exception:
                return False
        return False

    def read(self):
        if self.decoder is not None:
            frame = self._read_from_decoder_once()
            if frame is not None:
                return frame
            if self._restart_video_file():
                return self._read_from_decoder_once()
            return None

        frame = self._read_from_cap_once()
        if frame is not None:
            return frame
        if self._restart_video_file():
            return self._read_from_cap_once()
        return None

    def release(self):
        if self.decoder is not None:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class DecodeHub:
    def __init__(
        self,
        sources: List[str],
        preview_frame_queues,
        frame_hub: LatestFrameHub,
        control_queue,
        fps: int = 20,
        preview_fps: int = 20,
        input_mode: int = 0,
        preview_cfg: Optional[dict] = None,
        preview_status=None,
        bm_cfg: Optional[dict] = None,
    ):
        raw_sources = list(sources or [])
        self.source_specs = [normalize_source_spec(source, input_mode) for source in raw_sources]
        self.sources = [spec["source"] for spec in self.source_specs]
        self.preview_frame_queues = list(preview_frame_queues or [])
        self.frame_hub = frame_hub
        self.control_queue = control_queue
        self.fps = max(1, int(fps))
        self.preview_fps = max(1, int(preview_fps))
        self.input_mode = int(input_mode)
        self.preview_cfg = dict(preview_cfg or {})
        self.preview_status = preview_status
        self.bm_cfg = dict(bm_cfg or {})
        self.device_id = int(self.bm_cfg.get("device_id", 0))
        self.decode_backend = str(self.bm_cfg.get("decode_backend", "auto")).strip().lower() or "auto"
        self.scale_backend = str(self.bm_cfg.get("scale_backend", "bmcv")).strip().lower() or "bmcv"
        self.detect_emit_fps = max(1, int(self.bm_cfg.get("detect_emit_fps", self.fps)))
        self.preview_source_fps = max(1, int(self.bm_cfg.get("preview_source_fps", self.preview_fps)))
        self.detect_frame_max_width = int(self.bm_cfg.get("detect_frame_max_width", 0) or 0)
        self.detect_frame_max_height = int(self.bm_cfg.get("detect_frame_max_height", 0) or 0)
        self.stream_count = min(len(self.source_specs), len(self.preview_frame_queues))
        self.running = False
        self.preview_target_width, self.preview_target_height = self._preview_target_size()
        self.scaler = PreviewScaler(
            self.device_id,
            requested_backend=self.scale_backend,
            target_width=self.preview_target_width,
            target_height=self.preview_target_height,
        )
        self._decoder_pool: List[_DecoderPoolChannel] = []
        self._multi_decoder = None
        self._multi_channels: Dict[int, int] = {}
        self._detect_next_emit: Dict[int, float] = {sid: 0.0 for sid in range(self.stream_count)}
        self._preview_next_emit: Dict[int, float] = {sid: 0.0 for sid in range(self.stream_count)}
        self._detect_resize_logged = set()
        self._latest_locks = [threading.Lock() for _ in range(self.stream_count)]
        self._latest_frames: Dict[int, dict] = {}
        self._worker_threads: List[threading.Thread] = []
        self._stats: Dict[int, _ChannelStats] = {
            sid: _ChannelStats(last_log_ts=time.time(), fps_t0=time.time())
            for sid in range(self.stream_count)
        }

    def _preview_target_size(self):
        if self.stream_count == 1:
            width = int(self.bm_cfg.get("preview_frame_width", 0) or 0)
            height = int(self.bm_cfg.get("preview_frame_height", 0) or 0)
            if width <= 0 or height <= 0:
                return 0, 0
            return max(1, width), max(1, height)

        width = int(self.bm_cfg.get("preview_cell_width", PREVIEW_CELL_W) or PREVIEW_CELL_W)
        height = int(self.bm_cfg.get("preview_cell_height", PREVIEW_CELL_H) or PREVIEW_CELL_H)
        return max(1, width), max(1, height)

    def _fit_detect_frame(self, frame: np.ndarray, stream_id: Optional[int] = None) -> np.ndarray:
        slot_max_w = int(getattr(self.frame_hub.config, "max_width", frame.shape[1]))
        slot_max_h = int(getattr(self.frame_hub.config, "max_height", frame.shape[0]))
        max_w = slot_max_w
        max_h = slot_max_h
        if self.detect_frame_max_width > 0:
            max_w = min(max_w, max(1, self.detect_frame_max_width))
        if self.detect_frame_max_height > 0:
            max_h = min(max_h, max(1, self.detect_frame_max_height))
        height, width = frame.shape[:2]
        if width <= max_w and height <= max_h:
            return frame
        scale = min(max_w / max(1, width), max_h / max(1, height))
        target_w = max(1, int(width * scale))
        target_h = max(1, int(height * scale))
        log_key = "unknown" if stream_id is None else int(stream_id)
        if log_key not in self._detect_resize_logged:
            self._detect_resize_logged.add(log_key)
            logger.info(
                "DecodeHub resizing detect frame: stream=%s %sx%s -> %sx%s limit=%sx%s slot=%sx%s",
                "-" if stream_id is None else int(stream_id) + 1,
                width,
                height,
                target_w,
                target_h,
                max_w,
                max_h,
                slot_max_w,
                slot_max_h,
            )
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _set_preview_status(self, **kwargs):
        if self.preview_status is None:
            return
        for key, value in kwargs.items():
            self.preview_status[key] = value

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("DecodeHub received signal %s", signum)
        self.stop()
        sys.exit(0)

    def _put_preview_frame(self, stream_id: int, frame_data: dict):
        queue_obj = self.preview_frame_queues[stream_id]
        try:
            while True:
                try:
                    queue_obj.get_nowait()
                except Exception:
                    break
            queue_obj.put(frame_data, block=False)
        except Exception:
            pass

    def _record_frame(self, stream_id: int, frame: np.ndarray):
        timestamp = datetime.now(BEIJING_TZ)
        source_size = (frame.shape[1], frame.shape[0])
        stats = self._stats[stream_id]
        stats.frame_number += 1
        stats.fps_n += 1
        stats.last_frame_ts = time.time()

        with self._latest_locks[stream_id]:
            self._latest_frames[stream_id] = {
                "frame": frame,
                "frame_number": stats.frame_number,
                "timestamp": timestamp,
                "source_size": source_size,
            }

        now = time.time()
        if now - stats.last_log_ts >= 3.0:
            elapsed = now - stats.fps_t0
            actual_fps = stats.fps_n / elapsed if elapsed > 0 else 0.0
            log_pull_fps(
                stream_id,
                actual_fps,
                target_fps=self.fps,
                source_type="DecodeHub",
            )
            stats.last_log_ts = now
            stats.fps_t0 = now
            stats.fps_n = 0

    def _record_bmimage(self, stream_id: int, bmimg):
        timestamp = datetime.now(BEIJING_TZ)
        source_size = (int(bmimg.width()), int(bmimg.height()))
        stats = self._stats[stream_id]
        stats.frame_number += 1
        stats.fps_n += 1
        stats.last_frame_ts = time.time()

        with self._latest_locks[stream_id]:
            self._latest_frames[stream_id] = {
                "bmimg": bmimg,
                "frame_number": stats.frame_number,
                "timestamp": timestamp,
                "source_size": source_size,
            }

        now = time.time()
        if now - stats.last_log_ts >= 3.0:
            elapsed = now - stats.fps_t0
            actual_fps = stats.fps_n / elapsed if elapsed > 0 else 0.0
            log_pull_fps(
                stream_id,
                actual_fps,
                target_fps=self.fps,
                source_type="DecodeHub",
            )
            stats.last_log_ts = now
            stats.fps_t0 = now
            stats.fps_n = 0

    def _get_latest_frame(self, stream_id: int):
        with self._latest_locks[stream_id]:
            return self._latest_frames.get(stream_id)

    def _process_latest_frames(self, stream_id: int):
        last_detect_frame_number = 0
        last_preview_frame_number = 0
        detect_interval = 1.0 / float(self.detect_emit_fps)
        preview_interval = 1.0 / float(self.preview_source_fps)
        scaler = PreviewScaler(
            self.device_id,
            requested_backend=self.scale_backend,
            target_width=self.preview_target_width,
            target_height=self.preview_target_height,
        )

        while self.running:
            now_mono = time.monotonic()
            next_due = min(self._detect_next_emit[stream_id], self._preview_next_emit[stream_id])
            if now_mono < next_due:
                time.sleep(min(0.01, max(0.001, next_due - now_mono)))
                continue

            payload = self._get_latest_frame(stream_id)
            if payload is None:
                time.sleep(0.005)
                continue

            frame = payload.get("frame")
            bmimg = payload.get("bmimg")
            frame_number = int(payload["frame_number"])
            timestamp = payload["timestamp"]
            source_size = payload["source_size"]

            if now_mono >= self._detect_next_emit[stream_id]:
                if frame_number > last_detect_frame_number:
                    detect_source = frame if frame is not None else bmimg.asmat()
                    if detect_source is not None:
                        detect_frame = self._fit_detect_frame(detect_source, stream_id=stream_id)
                        self.frame_hub.write_frame(
                            stream_id,
                            detect_frame,
                            frame_number=frame_number,
                            timestamp=timestamp,
                            connected=True,
                            source_size=source_size,
                        )
                    last_detect_frame_number = frame_number
                self._detect_next_emit[stream_id] = now_mono + detect_interval

            if now_mono >= self._preview_next_emit[stream_id]:
                if frame_number > last_preview_frame_number:
                    if self.stream_count == 1 and (
                        self.preview_target_width <= 0 or self.preview_target_height <= 0
                    ):
                        preview_source = frame if frame is not None else bmimg.asmat()
                        if preview_source is None:
                            self._preview_next_emit[stream_id] = now_mono + preview_interval
                            continue
                        preview_frame = np.ascontiguousarray(preview_source).copy()
                    else:
                        preview_frame = scaler.resize(frame) if frame is not None else scaler.resize_bmimage(bmimg)
                    preview_data = {
                        "frame": preview_frame,
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "shape": preview_frame.shape,
                        "stream_id": stream_id,
                        "original_width": int(source_size[0]),
                        "original_height": int(source_size[1]),
                    }
                    self._put_preview_frame(stream_id, preview_data)
                    last_preview_frame_number = frame_number
                self._preview_next_emit[stream_id] = now_mono + preview_interval

    def _start_processing_workers(self):
        self._worker_threads = []
        for stream_id in range(self.stream_count):
            thread = threading.Thread(
                target=self._process_latest_frames,
                args=(stream_id,),
                daemon=True,
                name=f"decodehub-process-{stream_id + 1}",
            )
            thread.start()
            self._worker_threads.append(thread)

    def _init_multi_decoder(self) -> bool:
        if not SOPHON_AVAILABLE or not hasattr(sail, "MultiDecoder"):
            return False
        if any(spec.get("input_mode") == 1 for spec in self.source_specs[: self.stream_count]):
            logger.info("DecodeHub detected local video files, using decoder_pool instead of sail.MultiDecoder")
            return False
        try:
            self._multi_decoder = sail.MultiDecoder(16, self.device_id)
            self._multi_decoder.set_local_flag(True)
            self._multi_channels = {}
            for stream_id, spec in enumerate(self.source_specs[: self.stream_count]):
                self._multi_channels[stream_id] = self._multi_decoder.add_channel(spec["source"], 0)
            self._set_preview_status(decoder_backend="multidecoder", scale_backend=self.scaler.actual_backend)
            logger.info("DecodeHub using sail.MultiDecoder for %s streams", self.stream_count)
            return True
        except Exception as exc:
            logger.warning("DecodeHub MultiDecoder init failed, fallback to decoder_pool: %s", exc)
            self._multi_decoder = None
            self._multi_channels = {}
            return False

    def _init_decoder_pool(self):
        self._decoder_pool = []
        for stream_id, source_spec in enumerate(self.source_specs[: self.stream_count]):
            channel = _DecoderPoolChannel(
                source_spec,
                stream_id=stream_id,
                input_mode=self.input_mode,
                fps=self.fps,
                device_id=self.device_id,
            )
            channel.connect()
            self._decoder_pool.append(channel)
        self._set_preview_status(decoder_backend="decoder_pool", scale_backend=self.scaler.actual_backend)
        logger.info("DecodeHub using decoder_pool for %s streams", self.stream_count)

    def _read_multi_decoder(self):
        got_any = False
        now = time.time()
        for stream_id, channel_index in self._multi_channels.items():
            bmimg = sail.BMImage()
            try:
                ret = self._multi_decoder.read(int(channel_index), bmimg)
            except Exception as exc:
                logger.warning("DecodeHub MultiDecoder read failed stream=%s: %s", stream_id, exc)
                ret = -1
            if ret != 0:
                continue
            got_any = True
            self._record_bmimage(stream_id, bmimg)
        for stream_id, stats in self._stats.items():
            if stats.last_frame_ts and now - stats.last_frame_ts > 2.0:
                self.frame_hub.mark_disconnected(stream_id)
        return got_any

    def _read_decoder_pool(self):
        got_any = False
        for channel in self._decoder_pool:
            frame = channel.read()
            if frame is None:
                self.frame_hub.mark_disconnected(channel.stream_id)
                channel.reconnect()
                continue
            got_any = True
            self._record_frame(channel.stream_id, frame)
        return got_any

    def _check_control_commands(self):
        try:
            while True:
                cmd = self.control_queue.get_nowait()
                if cmd == "stop":
                    self.stop()
                    return
        except queue.Empty:
            return
        except Exception:
            return

    def start(self):
        logger.info("DecodeHub starting...")
        self.setup_signal_handlers()
        self.running = True

        backend = self.decode_backend
        if backend in ("auto", "multidecoder"):
            if not self._init_multi_decoder():
                self._init_decoder_pool()
        else:
            self._init_decoder_pool()

        self._start_processing_workers()
        self.run()

    def run(self):
        while self.running:
            self._check_control_commands()
            if not self.running:
                break

            got_any = False
            if self._multi_decoder is not None:
                got_any = self._read_multi_decoder()
            else:
                got_any = self._read_decoder_pool()

            if not got_any:
                time.sleep(0.005)

        logger.info("DecodeHub stopped")

    def stop(self):
        logger.info("DecodeHub stopping...")
        self.running = False
        for stream_id in range(self.stream_count):
            with self._latest_locks[stream_id]:
                self._latest_frames.pop(stream_id, None)
        for thread in self._worker_threads:
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass
        self._worker_threads.clear()
        if self._multi_decoder is not None:
            try:
                self._multi_decoder = None
            except Exception:
                pass
        for channel in self._decoder_pool:
            try:
                channel.release()
            except Exception:
                pass
        self._decoder_pool.clear()
        for stream_id in range(self.stream_count):
            try:
                self.frame_hub.clear_stream(stream_id)
            except Exception:
                pass


def run_decode_hub(
    sources,
    preview_frame_queues,
    frame_hub: LatestFrameHub,
    control_queue,
    fps=20,
    preview_fps=20,
    input_mode=0,
    preview_cfg=None,
    preview_status=None,
    bm_cfg=None,
):
    ensure_root_logging()
    service = DecodeHub(
        sources=sources,
        preview_frame_queues=preview_frame_queues,
        frame_hub=frame_hub,
        control_queue=control_queue,
        fps=fps,
        preview_fps=preview_fps,
        input_mode=input_mode,
        preview_cfg=preview_cfg,
        preview_status=preview_status,
        bm_cfg=bm_cfg,
    )
    service.start()
