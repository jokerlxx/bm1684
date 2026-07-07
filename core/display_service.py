"""
第三层：展示服务 (Display Service)
职责：
1. 接收所有检测服务的结果
2. 汇总并渲染到画面上
3. 处理警报：保存前后6秒的视频和报警关键帧图片
4. 输出视频流供Web界面展示
5. 告警事件通过 alert_queue 推送给 Web 端（SSE），报警视频/图片通过 Web 端下载
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime, timedelta, timezone

# 北京时间（UTC+8），告警文件名与时间比较统一使用
BEIJING_TZ = timezone(timedelta(hours=8))
import logging
import signal
import sys
import shutil
import subprocess
from collections import deque
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Threading 异步保存支持
import threading
from queue import Queue, Full, Empty

from core.logging_utils import ensure_root_logging, log_alarm_snapshot, log_alarm_video, log_hls_status, log_preview_fps

try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [DisplayService] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VideoBufferManager:
    """管理视频帧缓冲，用于保存警报前后的视频片段 - Threading异步版本"""
    
    def __init__(self, fps=30, pre_alarm_seconds=6, post_alarm_seconds=6, 
                 output_dir='./alarm_videos', on_video_saved=None, retention_days=7):
        self.fps = fps
        self.pre_alarm_seconds = pre_alarm_seconds
        self.post_alarm_seconds = post_alarm_seconds
        self.retention_days = max(1, int(retention_days))
        
        max_buffer_frames = int(pre_alarm_seconds * 30)
        self.pre_buffer = deque(maxlen=max_buffer_frames)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.on_video_saved = on_video_saved
        
        # 录制状态
        self.recording_alarms = {}
        
        # 🆕 异步保存队列和线程
        self.save_queue = Queue(maxsize=10)
        self.save_thread = threading.Thread(
            target=self._async_save_worker,
            daemon=True,
            name="VideoSaveThread"
        )
        self.save_thread_running = True
        self.save_thread.start()
        
        # 🆕 统计信息
        self.total_saved = 0
        self.total_dropped = 0

        self._cleanup_old_outputs(reason="startup")
        
        logger.info(f"VideoBufferManager initialized (Threading Async):")
        logger.info(f"   ├─ {pre_alarm_seconds}s pre + {post_alarm_seconds}s post")
        logger.info(f"   ├─ Max pre-buffer: {max_buffer_frames} frames")
        logger.info(f"   ├─ Output: {self.output_dir}")
        logger.info(f"   ├─ Retention: {self.retention_days} days")
        logger.info(f"   └─ ✅ Async save thread started")
        if SOPHON_AVAILABLE:
            logger.info("Sophon SAIL detected: alarm videos can use hardware encoder like yolov8_bmcv")

    def _cleanup_old_outputs(self, reason="periodic"):
        try:
            from storage import cleanup_old_alarm_files

            deleted = cleanup_old_alarm_files(str(self.output_dir), max_age_days=self.retention_days)
            if deleted > 0:
                logger.info(
                    "🧹 [ALARM CLEANUP] reason=%s retention_days=%d deleted=%d",
                    reason,
                    self.retention_days,
                    deleted,
                )
        except Exception as e:
            logger.warning("Failed to cleanup old alarm files: %s", e)
    
    def add_frame(self, frame, frame_number, timestamp):
        """添加帧到预缓冲区"""
        self.pre_buffer.append({
            'frame': frame.copy(),
            'frame_number': frame_number,
            'timestamp': timestamp
        })
    
    def trigger_alarm(self, alarm_id, alarm_type, alarm_info, stream_id=None):
        """触发警报，开始录制"""
        if alarm_id in self.recording_alarms:
            return
        
        logger.warning(f"Alarm triggered: {alarm_type} - {alarm_id}")
        
        current_time = datetime.now(BEIJING_TZ)
        cutoff_time = current_time - timedelta(seconds=self.pre_alarm_seconds)
        
        pre_frames = []
        for frame_data in self.pre_buffer:
            if frame_data['timestamp'] >= cutoff_time:
                pre_frames.append(frame_data)
        
        if len(pre_frames) >= 2:
            actual_pre_duration = (current_time - pre_frames[0]['timestamp']).total_seconds()
        else:
            actual_pre_duration = 0
        
        end_time = current_time + timedelta(seconds=self.post_alarm_seconds)
        trigger_frame = pre_frames[-1]['frame'].copy() if pre_frames else None
        
        logger.info(f"   ├─ Pre-buffer frames: {len(pre_frames)}")
        logger.info(f"   ├─ Actual pre-duration: {actual_pre_duration:.1f}s (target: {self.pre_alarm_seconds}s)")
        logger.info(f"   ├─ Will record until: {end_time.strftime('%H:%M:%S')}")
        logger.info(f"   ├─ Trigger frame captured: {trigger_frame is not None}")
        logger.info(f"   └─ Expected total duration: ~{actual_pre_duration + self.post_alarm_seconds:.1f}s")
        
        self.recording_alarms[alarm_id] = {
            'alarm_type': alarm_type,
            # 记录触发告警时的关键信息（包括所属通道）
            'alarm_info': {
                **(alarm_info or {}),
                'stream_id': stream_id,
            },
            'frames': pre_frames,
            'end_time': end_time,
            'start_time': pre_frames[0]['timestamp'] if pre_frames else current_time,
            'trigger_time': current_time,
            'trigger_frame': trigger_frame
        }
    
    def update(self, frame, frame_number, timestamp):
        """更新所有正在录制的警报 - 异步版本"""
        to_remove = []
        
        for alarm_id, recording in self.recording_alarms.items():
            recording['frames'].append({
                'frame': frame.copy(),
                'frame_number': frame_number,
                'timestamp': timestamp
            })
            
            frames_collected = len(recording['frames'])
            end_time = recording['end_time']
            trigger_time = recording['trigger_time']
            time_remaining = (end_time - timestamp).total_seconds()
            elapsed_since_trigger = (timestamp - trigger_time).total_seconds()
            
            if frames_collected % 30 == 0:
                alarm_type = recording['alarm_type']
                progress = (elapsed_since_trigger / self.post_alarm_seconds) * 100
                progress = min(progress, 100)
                logger.info(f"🎬 [RECORDING] {alarm_type} - "
                        f"Collected: {frames_collected} frames, "
                        f"Time elapsed: {elapsed_since_trigger:.1f}s / {self.post_alarm_seconds}s, "
                        f"Progress: {progress:.1f}%, "
                        f"Remaining: {time_remaining:.1f}s")
            
            if timestamp >= end_time:
                to_remove.append(alarm_id)
                total_duration = (timestamp - recording['start_time']).total_seconds()
                post_duration = (timestamp - trigger_time).total_seconds()
                pre_duration = (trigger_time - recording['start_time']).total_seconds()
                
                logger.info(f"✅ [RECORDING] {recording['alarm_type']} completed")
                logger.info(f"   ├─ Total frames: {frames_collected}")
                logger.info(f"   ├─ Pre-alarm duration: {pre_duration:.2f}s")
                logger.info(f"   ├─ Post-alarm duration: {post_duration:.2f}s")
                logger.info(f"   └─ Total duration: {total_duration:.2f}s")
        
        # 🆕 异步保存 - 立即返回，不阻塞
        for alarm_id in to_remove:
            self._queue_save_task(alarm_id)
            del self.recording_alarms[alarm_id]
    
    def _queue_save_task(self, alarm_id):
        """将保存任务加入异步队列"""
        recording = self.recording_alarms[alarm_id]
        
        save_data = {
            'alarm_id': alarm_id,
            'alarm_type': recording['alarm_type'],
            'alarm_info': recording['alarm_info'],
            'start_time': recording['start_time'],
            'trigger_frame': recording['trigger_frame'].copy() if recording.get('trigger_frame') is not None else None,
            'frames': recording['frames']
        }
        
        try:
            self.save_queue.put_nowait(save_data)
            logger.info(f"📤 [ASYNC] Task queued: {alarm_id} (Queue: {self.save_queue.qsize()}/{self.save_queue.maxsize})")
        except Full:
            self.total_dropped += 1
            logger.error(f"❌ [ASYNC] Queue FULL! Dropped: {alarm_id} (Total: {self.total_dropped})")
    
    def _async_save_worker(self):
        """后台保存线程"""
        logger.info("🧵 [THREAD] Video save worker started")
        
        while self.save_thread_running:
            try:
                save_data = self.save_queue.get(timeout=1.0)
                
                alarm_id = save_data['alarm_id']
                logger.info(f"💾 [THREAD] Saving: {alarm_id}")
                
                save_start = time.time()
                self._save_video_impl(save_data)
                save_duration = time.time() - save_start
                
                self.total_saved += 1
                logger.info(f"✅ [THREAD] Saved in {save_duration:.2f}s: {alarm_id} "
                           f"(Total: {self.total_saved}, Dropped: {self.total_dropped})")
                
                self.save_queue.task_done()
                
            except Empty:
                # 超时无任务或关闭时队列空，属正常，不记错
                pass
            except Exception as e:
                if "timed out" not in str(e).lower():
                    logger.error(f"❌ [THREAD] Error: {e}", exc_info=True)
        
        logger.info("🧵 [THREAD] Video save worker stopped")
    
    def _save_video_impl(self, save_data):
        """实际保存逻辑（在后台线程执行）"""
        self._cleanup_old_outputs(reason="before_save")

        alarm_id = save_data['alarm_id']
        alarm_type = save_data['alarm_type']
        alarm_info = save_data['alarm_info']
        frames = save_data['frames']
        trigger_frame = save_data['trigger_frame']
        start_time = save_data['start_time']
        
        if len(frames) == 0:
            logger.warning(f"No frames to save for alarm {alarm_id}")
            return
        
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        # 从 alarm_info 中提取通道信息（0 基 -> 展示为 1,2,...）
        stream_id = None
        try:
            sid = alarm_info.get('stream_id')
            if isinstance(sid, int):
                stream_id = sid
        except Exception:
            stream_id = None
        
        # 1. 保存关键帧图片
        image_path = None
        image_save_ms = None
        if trigger_frame is not None:
            if stream_id is not None:
                image_path = self.output_dir / f"{alarm_type}_ch{stream_id + 1}_{timestamp}_frame.jpg"
            else:
                image_path = self.output_dir / f"{alarm_type}_{timestamp}_frame.jpg"
            try:
                image_start = time.perf_counter()
                cv2.imwrite(str(image_path), trigger_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image_save_ms = (time.perf_counter() - image_start) * 1000.0
                image_size = image_path.stat().st_size
                logger.info(f"📸 [IMAGE SAVE] Frame saved: {image_path.name} ({image_size/1024:.1f}KB)")
                log_alarm_snapshot(
                    alarm_type,
                    image_save_ms,
                    image_path=image_path.name,
                    stream_id=stream_id,
                )
            except Exception as e:
                logger.error(f"Failed to save alarm frame: {e}")
                image_path = None
        
        # 2. 保存视频（同样编码通道信息，便于历史列表按通道筛选）
        if stream_id is not None:
            video_path = self.output_dir / f"{alarm_type}_ch{stream_id + 1}_{timestamp}.mp4"
        else:
            video_path = self.output_dir / f"{alarm_type}_{timestamp}.mp4"
        
        h, w = frames[0]['frame'].shape[:2]
        
        if len(frames) >= 2:
            first_ts = frames[0]['timestamp']
            last_ts = frames[-1]['timestamp']
            actual_duration = (last_ts - first_ts).total_seconds()
            actual_fps = len(frames) / actual_duration if actual_duration > 0 else self.fps
            actual_fps = max(1.0, min(actual_fps, 30.0))
        else:
            actual_fps = self.fps
        
        logger.info("=" * 60)
        logger.info(f"🎬 [VIDEO SAVE] Starting to save alarm video")
        logger.info(f"   ├─ Alarm ID: {alarm_id}")
        logger.info(f"   ├─ Alarm Type: {alarm_type}")
        logger.info(f"   ├─ Video output: {video_path}")
        logger.info(f"   ├─ Image output: {image_path if image_path else 'N/A'}")
        logger.info(f"   ├─ Total frames collected: {len(frames)}")
        logger.info(f"   ├─ Target duration: {self.pre_alarm_seconds + self.post_alarm_seconds}s")
        logger.info(f"   ├─ Target FPS: {self.fps}")
        logger.info(f"   🎯 Actual FPS (dynamic): {actual_fps:.2f}")
        logger.info(f"   ├─ Frame size: {w}x{h}")
        logger.info(f"   └─ Playback duration: {len(frames)/actual_fps:.2f}s")
        
        # 与 yolov8_bmcv 保持一致：若可用则优先使用 Sophon 硬件编码器，否则退回 OpenCV 软件编码
        use_sophon_encoder = SOPHON_AVAILABLE
        write_start_time = time.time()
        frame_count = len(frames)
        if use_sophon_encoder:
            try:
                device_id = 0
                handle = sail.Handle(device_id)
                bmcv = sail.Bmcv(handle)
                enc_params = (
                    f"width={w}:height={h}:bitrate=2000:gop=32:"
                    f"gop_preset=2:framerate={int(actual_fps)}"
                )
                encoder = sail.Encoder(str(video_path), device_id, 'h264_bm', 'NV12', enc_params, 10)
                if not encoder.is_opened():
                    logger.error("❌ Failed to open Sophon Encoder, fallback to OpenCV VideoWriter")
                    use_sophon_encoder = False
                else:
                    bmimg = sail.BMImage(handle, h, w, sail.Format.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                    for idx, frame_data in enumerate(frames):
                        frame = frame_data['frame']
                        try:
                            bmcv.mat_to_bm_image(frame, bmimg)
                            encoder.video_write(bmimg)
                        except Exception as e:
                            logger.error(f"❌ Sophon encode error on frame {idx}: {e}")
                            use_sophon_encoder = False
                            break
                        if (idx + 1) % 60 == 0:
                            logger.info(
                                f"   [Sophon] Writing progress: {idx+1}/{frame_count} "
                                f"frames ({(idx+1)/frame_count*100:.1f}%)"
                            )
                    try:
                        encoder.release()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"❌ Sophon encoder initialization failed: {e}")
                use_sophon_encoder = False

        if not use_sophon_encoder:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, actual_fps, (w, h))
            if not out.isOpened():
                logger.error(f"❌ Failed to create video writer!")
                return
            for idx, frame_data in enumerate(frames):
                frame = frame_data['frame']
                out.write(frame)
                if (idx + 1) % 60 == 0:
                    logger.info(
                        f"   Writing progress: {idx+1}/{frame_count} "
                        f"frames ({(idx+1)/frame_count*100:.1f}%)"
                    )
            out.release()
        
        write_duration = time.time() - write_start_time
        file_size = video_path.stat().st_size
        
        logger.info(f"✅ [VIDEO SAVE] Completed!")
        logger.info(f"   ├─ Video file: {video_path}")
        logger.info(f"   ├─ Video size: {file_size/1024/1024:.2f} MB")
        logger.info(f"   ├─ Image file: {image_path if image_path else 'N/A'}")
        logger.info(f"   ├─ Write duration: {write_duration:.2f}s")
        logger.info(f"   └─ Write speed: {frame_count/write_duration:.1f} fps")
        logger.info("=" * 60)
        log_alarm_video(
            alarm_type,
            write_duration * 1000.0,
            video_path=video_path.name,
            frame_count=frame_count,
            stream_id=stream_id,
        )
        self._cleanup_old_outputs(reason="after_save")
        
        # 调用回调
        if self.on_video_saved:
            try:
                self.on_video_saved(
                    str(video_path), 
                    str(image_path) if image_path else None,
                    alarm_type, 
                    alarm_info
                )
            except Exception as e:
                logger.error(f"Error in video saved callback: {e}")
    
    def get_active_recordings(self):
        """获取当前正在录制的警报数量"""
        return len(self.recording_alarms)
    
    def shutdown(self, timeout=30):
        """优雅关闭 - 等待所有保存任务完成"""
        logger.info("Shutting down VideoBufferManager...")
        logger.info(f"   Pending saves: {self.save_queue.qsize()}")
        
        try:
            self.save_queue.join()
            logger.info("   ✅ All pending saves completed")
        except Exception as e:
            logger.warning(f"   Error waiting for queue: {e}")
        
        self.save_thread_running = False
        self.save_thread.join(timeout=timeout)
        
        if self.save_thread.is_alive():
            logger.warning("   ⚠️ Save thread did not stop gracefully")
        else:
            logger.info("   ✅ Save thread stopped")
        
        logger.info(f"   Stats: Saved={self.total_saved}, Dropped={self.total_dropped}")

# ==================== 辅助函数 ====================
def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


# ==================== 展示服务 ====================
# 多路固定 1280x720 四宫格合成；单路保留整屏预览，避免先压到宫格再放大。
COMPOSITE_OUTPUT_W = 1280
COMPOSITE_OUTPUT_H = 720
GRID_COLUMNS = 2
GRID_ROWS = 2
GRID_CELL_COUNT = GRID_COLUMNS * GRID_ROWS
CELL_W = COMPOSITE_OUTPUT_W // GRID_COLUMNS
CELL_H = COMPOSITE_OUTPUT_H // GRID_ROWS


class HLSWriter:
    def __init__(self, preview_cfg=None, preview_status=None):
        self.preview_cfg = dict(preview_cfg or {})
        self.preview_status = preview_status
        self.process = None
        self.output_dir = Path(self.preview_cfg.get("hls_dir", "runtime/preview_hls")).expanduser()
        if not self.output_dir.is_absolute():
            self.output_dir = Path.cwd() / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.playlist_path = self.output_dir / "live.m3u8"
        self.segment_pattern = str(self.output_dir / "seg_%03d.ts")
        self.requested_encoder = str(self.preview_cfg.get("encoder", "auto")).strip() or "auto"
        self.encoder = self.requested_encoder
        self.strict = bool(self.preview_cfg.get("strict", True))
        self.runtime_fallback_enabled = bool(self.preview_cfg.get("runtime_encoder_fallback", True))
        self.preferred_encoder_candidates = [
            "h264_bm",
            "libx264",
            "libopenh264",
            "h264_omx",
            "h264_v4l2m2m",
        ]
        self.fallback_encoder_candidates = [
            "libx264",
            "libopenh264",
            "h264_omx",
            "h264_v4l2m2m",
        ]
        self.fps = max(1, int(self.preview_cfg.get("fps", 20)))
        self.width = 0
        self.height = 0
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._last_segment_mtime = None
        self._last_playlist_mtime = None
        self._last_logged_ts = 0.0
        self._unhealthy_reason = None
        self._stderr_handle = None
        self._stderr_log_path = self.output_dir / "ffmpeg.stderr.log"
        self._started_at = 0.0
        self._playlist_seen = False
        self._ffmpeg_bin = None
        self._encoder_list = None
        self._using_sail_backend = False
        self._sail_handle = None
        self._sail_bmcv = None
        self._sail_bmimg = None
        self._sail_encoder = None
        self._sail_segment_index = 0
        self._sail_segment_frames = 0
        self._sail_segments = []
        self._sail_last_segment_close_ts = None
        self._segment_seconds = max(1, int(self.preview_cfg.get("hls_segment_seconds", 1)))
        self._playlist_size = max(2, int(self.preview_cfg.get("hls_playlist_size", 3)))
        self._bitrate_kbps = max(256, int(self.preview_cfg.get("bitrate_kbps", 2000)))
        self._device_id = int(self.preview_cfg.get("device_id", 0))

    def _set_preview_status(self, **kwargs):
        if self.preview_status is None:
            return
        for key, value in kwargs.items():
            self.preview_status[key] = value

    def _read_ffmpeg_error(self):
        try:
            if not self._stderr_log_path.exists():
                return None
            text = self._stderr_log_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                return None
            return " | ".join(lines[-4:])
        except Exception:
            return None

    def _load_encoder_list(self):
        if self._ffmpeg_bin is not None and self._encoder_list is not None:
            return self._ffmpeg_bin, self._encoder_list, None

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None, None, "ffmpeg not found"

        try:
            res = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = res.stdout or ""
        except Exception as exc:
            return ffmpeg, None, f"failed to inspect ffmpeg encoders: {exc}"

        self._ffmpeg_bin = ffmpeg
        self._encoder_list = output
        return self._ffmpeg_bin, self._encoder_list, None

    def _build_cmd(self, ffmpeg, width: int, height: int, encoder: str):
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(int(self.fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-g",
            str(int(self.fps)),
            "-bf",
            "0",
            "-hls_time",
            str(max(1, int(self.preview_cfg.get("hls_segment_seconds", 1)))),
            "-hls_list_size",
            str(max(2, int(self.preview_cfg.get("hls_playlist_size", 3)))),
            "-hls_flags",
            "delete_segments+append_list+independent_segments",
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            self.segment_pattern,
            "-f",
            "hls",
            str(self.playlist_path),
        ]
        if encoder == "libx264":
            cmd[cmd.index("-g") : cmd.index("-g")] = ["-preset", "ultrafast", "-tune", "zerolatency"]
            cmd[cmd.index("-hls_time") : cmd.index("-hls_time")] = ["-pix_fmt", "yuv420p"]
        elif encoder == "h264_bm":
            cmd[cmd.index("-hls_time") : cmd.index("-hls_time")] = ["-pix_fmt", "nv12"]
        return cmd

    def _cleanup_output_dir(self):
        for old in self.output_dir.glob("*"):
            try:
                old.unlink()
            except Exception:
                pass

    def _sail_segment_path(self, index: int) -> Path:
        return self.output_dir / f"seg_{int(index):03d}.ts"

    def _write_live_playlist(self):
        if not self._sail_segments:
            return
        media_sequence = int(self._sail_segments[0]["index"])
        target_duration = max(
            self._segment_seconds,
            max(int(np.ceil(seg["duration"])) for seg in self._sail_segments),
        )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}",
            "#EXT-X-INDEPENDENT-SEGMENTS",
        ]
        for seg in self._sail_segments:
            lines.append(f"#EXTINF:{seg['duration']:.3f},")
            lines.append(seg["path"].name)
        self.playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _open_sail_segment_encoder(self):
        segment_path = self._sail_segment_path(self._sail_segment_index)
        enc_params = (
            f"width={self.width}:height={self.height}:bitrate={self._bitrate_kbps}:"
            f"gop={int(self.fps)}:gop_preset=2:framerate={int(self.fps)}"
        )
        encoder = sail.Encoder(
            str(segment_path),
            self._device_id,
            "h264_bm",
            "NV12",
            enc_params,
            10,
        )
        if not encoder.is_opened():
            raise RuntimeError(f"failed to open sail encoder for segment {segment_path.name}")
        self._sail_encoder = encoder
        self._sail_segment_frames = 0

    def _close_sail_segment(self, *, final=False):
        if self._sail_encoder is None:
            return
        try:
            self._sail_encoder.release()
        except Exception:
            pass
        self._sail_encoder = None

        if self._sail_segment_frames <= 0:
            return

        segment_path = self._sail_segment_path(self._sail_segment_index)
        duration = float(self._sail_segment_frames) / max(1.0, float(self.fps))
        self._sail_segments.append(
            {
                "index": self._sail_segment_index,
                "path": segment_path,
                "duration": duration,
            }
        )
        while len(self._sail_segments) > self._playlist_size:
            old = self._sail_segments.pop(0)
            try:
                old["path"].unlink()
            except Exception:
                pass
        self._write_live_playlist()
        self._playlist_seen = True
        self._sail_last_segment_close_ts = time.time()
        self._set_preview_status(
            healthy=True,
            encoder_backend="h264_bm",
            last_segment_ts=datetime.fromtimestamp(self._sail_last_segment_close_ts, tz=BEIJING_TZ).isoformat(),
            playlist_age_ms=0,
            unhealthy_reason=None if not final else self._unhealthy_reason,
        )
        self._sail_segment_index += 1
        self._sail_segment_frames = 0

    def _start_sail_backend(self):
        if not SOPHON_AVAILABLE:
            self._unhealthy_reason = "sophon.sail unavailable for h264_bm backend"
            return False
        self._cleanup_output_dir()
        try:
            self._sail_handle = sail.Handle(self._device_id)
            self._sail_bmcv = sail.Bmcv(self._sail_handle)
            self._sail_bmimg = sail.BMImage(
                self._sail_handle,
                self.height,
                self.width,
                sail.Format.FORMAT_BGR_PLANAR,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
            self._sail_segments = []
            self._sail_segment_index = 0
            self._open_sail_segment_encoder()
            self._using_sail_backend = True
        except Exception as exc:
            self._unhealthy_reason = f"failed to start sail h264_bm backend: {exc}"
            logger.error(self._unhealthy_reason)
            self._using_sail_backend = False
            self._sail_handle = None
            self._sail_bmcv = None
            self._sail_bmimg = None
            self._sail_encoder = None
            return False

        self._started_at = time.time()
        self._playlist_seen = False
        self._last_segment_mtime = None
        self._last_playlist_mtime = None
        self._set_preview_status(
            encoder_backend="h264_bm",
            healthy=False,
            unhealthy_reason="等待 h264_bm 生成 HLS 分片",
        )
        return True

    def _write_sail_frame(self, frame: np.ndarray):
        try:
            self._sail_bmcv.mat_to_bm_image(frame, self._sail_bmimg)
            encode_ret = self._sail_encoder.video_write(self._sail_bmimg)
            retry = 0
            while encode_ret != 0 and retry < 10:
                time.sleep(0.01)
                encode_ret = self._sail_encoder.video_write(self._sail_bmimg)
                retry += 1
            if encode_ret != 0:
                raise RuntimeError(f"sail encoder video_write failed: {encode_ret}")
            self._sail_segment_frames += 1
            if self._sail_segment_frames >= max(1, int(self.fps) * self._segment_seconds):
                self._close_sail_segment()
                self._open_sail_segment_encoder()
            return True
        except Exception as exc:
            reason = f"sail h264_bm write failed: {exc}"
            logger.error(reason)
            self._close_sail_segment(final=True)
            self._using_sail_backend = False
            self._close_process_handles()
            self._unhealthy_reason = reason
            return self._try_runtime_fallback(reason)

    def _available_encoder_candidates(self, encoder_list: str, candidates, exclude=None):
        exclude = exclude or set()
        available = []
        seen = set()
        for candidate in candidates:
            if candidate in seen or candidate in exclude:
                continue
            seen.add(candidate)
            if candidate in encoder_list:
                available.append(candidate)
        return available

    def _find_available_fallback_encoder(self, encoder_list: str):
        candidates = self._available_encoder_candidates(
            encoder_list,
            self.fallback_encoder_candidates,
            exclude={self.encoder},
        )
        return candidates[0] if candidates else None

    def _find_preferred_encoder(self, encoder_list: str):
        candidates = self._available_encoder_candidates(
            encoder_list,
            self.preferred_encoder_candidates,
        )
        return candidates[0] if candidates else None

    def _close_process_handles(self):
        if self._using_sail_backend:
            self._using_sail_backend = False
            try:
                self._close_sail_segment(final=True)
            except Exception:
                pass
            if self._sail_encoder is not None:
                try:
                    self._sail_encoder.release()
                except Exception:
                    pass
                self._sail_encoder = None
            self._sail_bmimg = None
            self._sail_bmcv = None
            self._sail_handle = None
        if self.process is not None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None

    def _spawn_with_encoder(self, ffmpeg, encoder: str, width: int, height: int):
        self.encoder = encoder
        self._cleanup_output_dir()

        try:
            self._stderr_handle = open(self._stderr_log_path, "w", encoding="utf-8")
        except Exception:
            self._stderr_handle = None

        cmd = self._build_cmd(ffmpeg, width, height, encoder)
        if encoder == "h264_bm":
            logger.info("HLSWriter preview uses ffmpeg muxing with h264_bm to produce browser-compatible HLS segments")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_handle or subprocess.DEVNULL,
            )
        except Exception as exc:
            self._unhealthy_reason = f"failed to start ffmpeg with {encoder}: {exc}"
            self._set_preview_status(
                healthy=False,
                encoder_backend=None,
                unhealthy_reason=self._unhealthy_reason,
            )
            logger.error("HLSWriter failed to spawn ffmpeg (%s): %s", encoder, exc)
            self.process = None
            if self._stderr_handle is not None:
                try:
                    self._stderr_handle.close()
                except Exception:
                    pass
                self._stderr_handle = None
            return False

        self._started_at = time.time()
        self._playlist_seen = False
        self._last_segment_mtime = None
        self._last_playlist_mtime = None
        self._set_preview_status(
            encoder_backend=self.encoder,
            healthy=False,
            unhealthy_reason=f"等待 {self.encoder} 生成 HLS 播放列表",
        )
        return True

    def _select_start_encoder(self):
        ffmpeg, encoder_list, error = self._load_encoder_list()
        if error:
            return None, None, error

        if self.requested_encoder == "auto":
            preferred = self._find_preferred_encoder(encoder_list)
            if preferred:
                return ffmpeg, preferred, None
            return ffmpeg, None, "no supported preview encoder found"

        if self.requested_encoder in encoder_list:
            return ffmpeg, self.requested_encoder, None
        if self.strict:
            return ffmpeg, None, f"encoder {self.requested_encoder} unavailable"
        fallback = self._find_available_fallback_encoder(encoder_list)
        if fallback:
            return ffmpeg, fallback, None
        return ffmpeg, None, f"encoder {self.requested_encoder} unavailable and no fallback found"

    def _try_runtime_fallback(self, reason: str) -> bool:
        if not self.runtime_fallback_enabled:
            logger.error("HLSWriter runtime fallback disabled: %s", reason)
            return False
        ffmpeg, encoder_list, error = self._load_encoder_list()
        if error or not encoder_list:
            logger.error("HLSWriter runtime fallback unavailable: %s", error or reason)
            return False
        candidates = self._available_encoder_candidates(
            encoder_list,
            self.fallback_encoder_candidates,
            exclude={self.encoder},
        )
        if not candidates:
            logger.error(
                "HLSWriter runtime fallback unavailable, no supported fallback encoder found. candidates=%s reason=%s",
                ",".join(self.fallback_encoder_candidates),
                reason,
            )
            return False
        failed_attempts = []
        current_encoder = self.encoder
        for fallback in candidates:
            logger.warning("HLSWriter runtime fallback trying: %s -> %s (%s)", current_encoder, fallback, reason)
            self._close_process_handles()
            self._unhealthy_reason = f"{current_encoder} 编码失败，尝试切换到 {fallback}: {reason}"
            if self._spawn_with_encoder(ffmpeg, fallback, self.width, self.height):
                return True
            failed_attempts.append(fallback)
            current_encoder = fallback
        logger.error(
            "HLSWriter runtime fallback failed after trying candidates=%s reason=%s last_error=%s",
            ",".join(failed_attempts),
            reason,
            self._unhealthy_reason,
        )
        return False

    def start(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

        ffmpeg, encoder, error = self._select_start_encoder()
        if error or not encoder:
            self._unhealthy_reason = error
            self._set_preview_status(
                healthy=False,
                encoder_backend=None,
                unhealthy_reason=error,
            )
            logger.error("HLSWriter init failed: %s", error)
            return False

        return self._spawn_with_encoder(ffmpeg, encoder, self.width, self.height)

    def write(self, frame: np.ndarray):
        if self._using_sail_backend:
            return self._write_sail_frame(frame)

        if self.process is None or self.process.stdin is None:
            return False

        if self.process.poll() is not None:
            reason = self._read_ffmpeg_error() or f"ffmpeg exited with code {self.process.returncode}"
            if not self._try_runtime_fallback(reason):
                self._unhealthy_reason = reason
                self._set_preview_status(
                    healthy=False,
                    unhealthy_reason=reason,
                )
                return False

        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.process.stdin.flush()
        except Exception as exc:
            reason = self._read_ffmpeg_error() or str(exc)
            if self._try_runtime_fallback(reason):
                try:
                    self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    self.process.stdin.flush()
                except Exception as exc2:
                    self._unhealthy_reason = self._read_ffmpeg_error() or f"fallback encoder write failed: {exc2}"
                    self._set_preview_status(
                        healthy=False,
                        unhealthy_reason=self._unhealthy_reason,
                    )
                    return False
            else:
                self._unhealthy_reason = f"ffmpeg stdin write failed: {reason}. 若板端无 libx264/libopenh264/h264_v4l2m2m 等回退编码器，将无法自动切换。"
                self._set_preview_status(
                    healthy=False,
                    unhealthy_reason=self._unhealthy_reason,
                )
                return False

        self._fps_n += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            actual_fps = self._fps_n / max(0.001, now - self._fps_t0)
            self._fps_t0 = now
            self._fps_n = 0
            segment_interval_ms = None
            playlist_age_ms = None
            if self.playlist_path.exists():
                playlist_mtime = self.playlist_path.stat().st_mtime
                self._last_playlist_mtime = playlist_mtime
                playlist_age_ms = max(0, int((now - playlist_mtime) * 1000))
                self._playlist_seen = True
            newest_segment_mtime = None
            for segment in self.output_dir.glob("*.ts"):
                mtime = segment.stat().st_mtime
                if newest_segment_mtime is None or mtime > newest_segment_mtime:
                    newest_segment_mtime = mtime
            if newest_segment_mtime is not None:
                if self._last_segment_mtime is not None:
                    segment_interval_ms = max(0, int((newest_segment_mtime - self._last_segment_mtime) * 1000))
                self._last_segment_mtime = newest_segment_mtime
            poll_code = self.process.poll()
            healthy = poll_code is None and self.playlist_path.exists()
            reason = self._unhealthy_reason
            if poll_code is not None:
                reason = self._read_ffmpeg_error() or f"ffmpeg exited with code {poll_code}"
            elif not self.playlist_path.exists():
                wait_s = now - self._started_at
                if wait_s >= max(2.0, float(self.preview_cfg.get("hls_segment_seconds", 1)) * 2.0):
                    reason = self._read_ffmpeg_error() or "ffmpeg 已启动，但 HLS 播放列表仍未生成"
                else:
                    reason = "等待 HLS 播放列表生成"
            else:
                reason = None
            self._unhealthy_reason = reason
            self._set_preview_status(
                healthy=healthy,
                encode_in_fps=actual_fps,
                encoder_backend=self.encoder,
                last_segment_ts=datetime.fromtimestamp(self._last_segment_mtime, tz=BEIJING_TZ).isoformat() if self._last_segment_mtime else None,
                playlist_age_ms=playlist_age_ms,
                unhealthy_reason=reason,
            )
            log_hls_status(actual_fps, playlist_age_ms=playlist_age_ms, segment_interval_ms=segment_interval_ms)
        return True

    def stop(self):
        self._close_process_handles()


class DisplayService:
    """展示服务 - 独立进程"""

    _RENDER_DETECTORS = (
        'fall',
        'ventilator',
        'fight',
        'crowd',
        'helmet',
        'window_door_inside',
        'window_door_outside',
    )
    
    def __init__(self, frame_queue, result_queues, output_queue, control_queue,
                 video_output_dir='./alarm_videos', fps=30, stream_count=1, alert_queue=None,
                 font_path_config=None, preview_status=None, preview_cfg=None, alarm_retention_days=7,
                 raw_output_queue=None):
        if isinstance(frame_queue, (list, tuple)):
            self.frame_queues = list(frame_queue)
        else:
            self.frame_queues = [frame_queue]
        self.result_queues = result_queues
        self.output_queue = output_queue
        self.raw_output_queue = raw_output_queue
        self.control_queue = control_queue
        self.alert_queue = alert_queue  # 告警事件队列，供 Web 层实时推送
        self.preview_status = preview_status
        self.preview_cfg = dict(preview_cfg or {})
        self.preview_transport = str(self.preview_cfg.get("transport", "local")).strip().lower() or "local"
        self.local_preview_enabled = (self.preview_transport == "local")
        self._local_preview_window_name = "SICNU Local Preview"
        self._local_preview_ready = False
        self._local_preview_error = None
        self.fps = fps
        # 支持 1~4 路视频流；预览采用四宫格布局（多路时）
        self.stream_count = max(1, min(GRID_CELL_COUNT, int(stream_count)))
        self._multi_stream = (self.stream_count > 1)
        self.latest_frames = {}  # stream_id -> frame_data（多路时使用）
        
        self.video_buffer = VideoBufferManager(
            fps=fps, 
            output_dir=video_output_dir,
            on_video_saved=self._on_video_saved,
            retention_days=alarm_retention_days,
        )
        
        # 中文字体（用于报警标签，避免中文乱码）
        self.font_path = None
        self._load_chinese_font(font_path_config)
        self._english_font_path = self._resolve_english_font_path()
        self._font_cache = {}
        self._label_sprite_cache = {}

        # 检测结果缓存（一路时 latest_results[det]=result；多路时 latest_results[det][stream_id]=result）
        self.latest_results = {
            'fall': {} if self._multi_stream else None,
            'ventilator': {} if self._multi_stream else None,
            'fight': {} if self._multi_stream else None,
            'crowd': {} if self._multi_stream else None,
            'helmet': {} if self._multi_stream else None,
            'window_door_inside': {} if self._multi_stream else None,
            'window_door_outside': {} if self._multi_stream else None,
        }
        self.sticky_results = {
            'fall': {} if self._multi_stream else None,
            'ventilator': {} if self._multi_stream else None,
            'fight': {} if self._multi_stream else None,
            'crowd': {} if self._multi_stream else None,
            'helmet': {} if self._multi_stream else None,
            'window_door_inside': {} if self._multi_stream else None,
            'window_door_outside': {} if self._multi_stream else None,
        }
        
        # 🎯 改进的警报跟踪系统（用于视频录制触发）。多路时 triggered_alarms[det][stream_id]=set()
        self.triggered_alarms = {
            'fall': {} if self._multi_stream else set(),
            'ventilator': {} if self._multi_stream else set(),
            'fight': {} if self._multi_stream else set(),
            'crowd': {} if self._multi_stream else set(),
            'helmet': {} if self._multi_stream else set(),
            'window_door_inside': {} if self._multi_stream else set(),
            'window_door_outside': {} if self._multi_stream else set(),
        }
        
        self.running = False
        
        # 警报框状态管理。多路时 active_alert_boxes[det][stream_id][tracker_id]=...
        self.active_alert_boxes = {
            'fall': {} if self._multi_stream else {},
            'ventilator': {} if self._multi_stream else {},
            'fight': {} if self._multi_stream else {},
            'crowd': {} if self._multi_stream else {},
            'helmet': {} if self._multi_stream else {},
            'window_door_inside': {} if self._multi_stream else {},
            'window_door_outside': {} if self._multi_stream else {},
        }
        # 兼容旧代码/测试命名，实际指向同一个报警框缓存。
        self.active_alerts = self.active_alert_boxes
        self.frame_count = 0

        # 每路视频状态：FPS + 最近帧时间（用于在每个宫格窗口叠加显示）
        self._stream_stats = {
            sid: {
                "last_frame_number": None,
                "last_seen_ts": None,   # datetime（BEIJING_TZ）
                "fps_t0": time.time(),
                "fps_n": 0,
                "fps": 0.0,
            }
            for sid in range(self.stream_count)
        }

        # 报警框显示模式：跟随/跳帧（闪烁）。默认跟随；由 main.py 根据配置覆盖。
        # 可选值：
        # - "follow": 每帧都画报警框（紧跟目标）
        # - "blink":  周期性闪烁显示（当前实现为 1Hz，0.5s 显示/0.5s 隐藏）
        self.alert_box_mode = getattr(self, 'alert_box_mode', 'follow')
        # 闪烁频率（Hz），仅在 alert_box_mode == "blink" 时生效
        self.alert_box_blink_hz = 1.0

        # 叠加层缓存：避免每帧 PIL 画字造成帧率下降
        self._overlay_cache = {
            sid: {
                "last_conn": None,
                "last_fps_int": None,
                "rgba": None,   # numpy RGBA 小图
            }
            for sid in range(self.stream_count)
        }

        # 最近一次收到检测结果时间（用于“无检测输出”时跳过告警缓冲以提升预览 FPS）
        self._last_detector_result_ts = None  # float(time.time())
        
        # 方案 C：按帧号缓存检测结果，渲染时取与当前帧最接近的结果（帧-结果对齐）
        self._result_by_frame = {}  # detector_type -> stream_id -> deque[(frame_number, result)]
        self._result_by_frame_maxlen = 60
        self._active_result_streams = {
            detector_type: set() for detector_type in self._RENDER_DETECTORS
        }

        # 方案 A：报警框 bbox 平滑缓存 (detector_type, stream_id, track_id) -> {'bbox': [x1,y1,x2,y2]}
        self._bbox_smooth_cache = {}
        # 框跟随优先：检测已经是低频结果，展示层不再额外滞后 bbox。
        self._bbox_smooth_alpha = 0.0
        # 检测结果用于更新报警框缓存；报警框本身另有短时 hold，避免低频检测下闪烁。
        self._display_result_ttl_s = max(0.1, float(self.preview_cfg.get("result_ttl_s", 1.0)))
        self._display_max_frame_lag = int(self.preview_cfg.get("result_max_frame_lag", max(3, int(self.fps * self._display_result_ttl_s))))
        self._alert_box_hold_s = max(0.1, float(self.preview_cfg.get("alert_box_hold_s", 1.2)))
        self._alert_box_tracking_enabled = bool(self.preview_cfg.get("alert_box_tracking", True))
        self._alert_box_prediction_max_s = max(0.0, float(self.preview_cfg.get("alert_box_prediction_max_s", 0.0)))
        self._alert_box_detection_lag_compensation = bool(self.preview_cfg.get("alert_box_detection_lag_compensation", False))
        self._alert_box_track_max_shift_ratio = max(0.05, float(self.preview_cfg.get("alert_box_track_max_shift_ratio", 0.35)))
        self._alert_box_predict_max_shift_ratio = max(0.1, float(self.preview_cfg.get("alert_box_predict_max_shift_ratio", 1.5)))
        self._alert_box_velocity_blend = max(0.0, min(1.0, float(self.preview_cfg.get("alert_box_velocity_blend", 0.65))))
        self._max_alert_boxes_per_stream = max(0, int(self.preview_cfg.get("max_alert_boxes_per_stream", 0)))
        self._max_alert_labels_per_stream = max(0, int(self.preview_cfg.get("max_alert_labels_per_stream", 0)))
        self._merged_simple_alert_boxes = bool(self.preview_cfg.get("merged_simple_alert_boxes", True))
        self._mjpeg_quality = max(30, min(95, int(self.preview_cfg.get("mjpeg_quality", 55))))
        self._alarm_buffer_fps = max(1, min(int(self.fps), int(self.preview_cfg.get("alarm_buffer_fps", 8))))
        self._alarm_buffer_interval_s = 1.0 / float(self._alarm_buffer_fps)
        self._last_alarm_buffer_mono = 0.0
        self._jpeg_input_queue = Queue(maxsize=1)
        self._jpeg_encoder_thread = None
        self._jpeg_encoder_running = False
        self._raw_jpeg_input_queue = Queue(maxsize=1) if self.raw_output_queue is not None else None
        self._raw_jpeg_encoder_thread = None
        self._raw_jpeg_encoder_running = False
        self._alert_debug_log_interval_s = float(self.preview_cfg.get("alert_debug_log_interval_s", 1.0))
        self._raw_model_boxes_enabled = bool(self.preview_cfg.get("raw_model_boxes_enabled", False))
        self._raw_model_boxes_min_conf = max(0.0, min(1.0, float(self.preview_cfg.get("raw_model_boxes_min_conf", 0.0))))
        self._alert_debug_last_log = {}
        self._alert_draw_counts = {sid: 0 for sid in range(self.stream_count)}
        self._alert_label_counts = {sid: 0 for sid in range(self.stream_count)}
        self._current_render_stream_id = 0
        self._alert_track_prev_gray = {}
        self._result_versions = {detector_type: {} for detector_type in self._RENDER_DETECTORS}
        self._rendered_cell_cache = {}

        self.preview_mode = 'merged'
        self._blank_cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        self._blank_cell[:] = (42, 23, 15)
        self._composite_canvas = np.empty((COMPOSITE_OUTPUT_H, COMPOSITE_OUTPUT_W, 3), dtype=np.uint8)
        self._max_batch_frames = max(12, self.stream_count * 2)
        self._perf_window = self._new_perf_window()
        self.hls_writer = (
            HLSWriter(preview_cfg=self.preview_cfg, preview_status=self.preview_status)
            if self.preview_transport == 'hls'
            else None
        )
        self._hls_start_attempted = False
        self._hls_started = False

        # 中文字体由 _load_chinese_font(font_path_config) 在 __init__ 中设置

    def _set_preview_status(self, **kwargs):
        if self.preview_status is None:
            return
        for key, value in kwargs.items():
            self.preview_status[key] = value

    def _new_perf_window(self):
        return {
            'queue_drain_ms': 0.0,
            'resize_ms': 0.0,
            'collect_ms': 0.0,
            'result_lookup_ms': 0.0,
            'overlay_ms': 0.0,
            'label_ms': 0.0,
            'composite_ms': 0.0,
            'buffer_ms': 0.0,
            'output_ms': 0.0,
            'input_frames': 0,
            'alert_boxes': 0,
            'stream_hits': [0 for _ in range(self.stream_count)],
        }

    def _perf_add(self, key, value_ms):
        if key in self._perf_window:
            self._perf_window[key] += max(0.0, float(value_ms))

    def _perf_count_input(self, stream_id, count=1):
        self._perf_window['input_frames'] += int(count)
        if 0 <= stream_id < self.stream_count:
            self._perf_window['stream_hits'][stream_id] += int(count)

    def _perf_inc_alert_boxes(self, count=1):
        self._perf_window['alert_boxes'] += int(count)

    def _resolve_english_font_path(self):
        candidates = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/arialbd.ttf',
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    ImageFont.truetype(path, 14)
                    return path
                except Exception:
                    continue
        return None

    def _get_cached_font(self, font_path, size):
        cache_key = (font_path or '__default__', int(size))
        font = self._font_cache.get(cache_key)
        if font is not None:
            return font
        try:
            if font_path:
                font = ImageFont.truetype(font_path, int(size))
            else:
                raise OSError("missing font path")
        except Exception:
            font = ImageFont.load_default()
        self._font_cache[cache_key] = font
        return font

    def _preview_frame_size(self):
        if self._multi_stream:
            return CELL_W, CELL_H
        return None

    def _prepare_frame_for_display(self, frame_data):
        frame = frame_data.get('frame')
        if frame is None or getattr(frame, 'size', 0) <= 0:
            return frame_data
        orig_h, orig_w = frame.shape[:2]
        source_w = frame_data.get('original_width')
        source_h = frame_data.get('original_height')
        if not isinstance(source_w, (int, float)) or source_w <= 0:
            source_w = orig_w
        if not isinstance(source_h, (int, float)) or source_h <= 0:
            source_h = orig_h
        target_size = self._preview_frame_size()
        if target_size is None:
            if frame_data.get('original_width') == int(source_w) and frame_data.get('original_height') == int(source_h):
                return frame_data
            return {
                **frame_data,
                'original_width': int(source_w),
                'original_height': int(source_h),
            }
        target_w, target_h = target_size
        if orig_w != target_w or orig_h != target_h:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return {
                **frame_data,
                'frame': resized,
                'original_width': int(source_w),
                'original_height': int(source_h),
            }
        if frame_data.get('original_width') == int(source_w) and frame_data.get('original_height') == int(source_h):
            return frame_data
        return {
            **frame_data,
            'original_width': int(source_w),
            'original_height': int(source_h),
        }

    def _absorb_latest_frames(self, wait_timeout=0.0, drain_timeout=0.0):
        del drain_timeout
        absorbed = 0
        deadline = time.monotonic() + max(0.0, float(wait_timeout))

        while True:
            got_any = False
            drain_start = time.perf_counter()
            for stream_id, frame_queue in enumerate(self.frame_queues[:self.stream_count]):
                latest_frame = None
                try:
                    while True:
                        latest_frame = frame_queue.get_nowait()
                except Exception:
                    pass
                if latest_frame is None:
                    continue
                got_any = True
                prepared = self._prepare_frame_for_display(latest_frame)
                self.latest_frames[stream_id] = prepared
                absorbed += 1
                self._perf_count_input(stream_id)
                try:
                    self._update_stream_stats(stream_id, prepared.get('frame_number'), prepared.get('timestamp'))
                except Exception:
                    pass
            self._perf_add('queue_drain_ms', (time.perf_counter() - drain_start) * 1000.0)

            if got_any or wait_timeout <= 0.0 or time.monotonic() >= deadline:
                break
            time.sleep(0.005)

        self._clear_sticky_for_failed_streams()
        return absorbed

    def _scale_bbox_to_display(self, bbox, original_wh, display_wh, bbox_format='xyxy'):
        """将原始分辨率下的 bbox 缩放到当前显示帧尺寸。original_wh/display_wh 为 (width, height)。"""
        if not bbox or len(bbox) < 4 or original_wh is None or display_wh is None:
            return bbox
        ow, oh = original_wh
        dw, dh = display_wh
        if ow <= 0 or oh <= 0 or dw <= 0 or dh <= 0:
            return bbox
        sx, sy = dw / ow, dh / oh
        try:
            if bbox_format == 'xywh':
                x, y, w, h = [float(b) for b in bbox[:4]]
                return [x * sx, y * sy, w * sx, h * sy]
            x1, y1, x2, y2 = [float(b) for b in bbox[:4]]
            return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        except (ValueError, TypeError):
            return bbox

    def _result_coord_wh(self, result, fallback_wh):
        """返回检测框所在坐标系尺寸；检测帧可能已被缩放，不一定等于原始视频尺寸。"""
        if not result:
            return fallback_wh
        try:
            coord_w = int(result.get('coord_width') or 0)
            coord_h = int(result.get('coord_height') or 0)
        except (ValueError, TypeError):
            return fallback_wh
        if coord_w > 0 and coord_h > 0:
            return (coord_w, coord_h)
        return fallback_wh

    def _update_stream_stats(self, stream_id, frame_number, timestamp):
        """更新每路 FPS/时间戳统计（用于叠加显示）。"""
        st = self._stream_stats.get(stream_id)
        if st is None:
            self._stream_stats[stream_id] = {
                "last_frame_number": None,
                "last_seen_ts": None,
                "fps_t0": time.time(),
                "fps_n": 0,
                "fps": 0.0,
            }
            st = self._stream_stats[stream_id]

        if timestamp is not None:
            st["last_seen_ts"] = timestamp

        # 仅在帧号变化时计数（避免复用同一帧导致 FPS 虚高）
        if frame_number is not None and frame_number != st.get("last_frame_number"):
            st["last_frame_number"] = frame_number
            st["fps_n"] += 1

        now = time.time()
        dt = now - st["fps_t0"]
        if dt >= 1.0:
            st["fps"] = float(st["fps_n"]) / dt if dt > 0 else 0.0
            st["fps_n"] = 0
            st["fps_t0"] = now

    def _get_stream_connection_state(self, stream_id):
        """
        连接状态推断（近似策略）：
        - 近 1.5s 有新帧：连接正常
        - 1.5s~5s 未更新：正在重连
        - 超过 5s 未更新：连接失败
        """
        st = self._stream_stats.get(stream_id) or {}
        ts = st.get("last_seen_ts")
        if ts is None:
            return "正在重连"
        try:
            age = (datetime.now(BEIJING_TZ) - ts).total_seconds()
        except Exception:
            return "正在重连"
        if age <= 1.5:
            return "连接正常"
        if age <= 5.0:
            return "正在重连"
        return "连接失败"

    def _clear_bbox_smooth_for(self, detector_type, stream_id):
        prefix = (detector_type, int(stream_id))
        for key in list(self._bbox_smooth_cache.keys()):
            if key[:2] == prefix:
                self._bbox_smooth_cache.pop(key, None)

    def _alert_box_store(self, detector_type, stream_id, create=True):
        if detector_type not in self.active_alert_boxes:
            if not create:
                return None
            self.active_alert_boxes[detector_type] = {} if not self._multi_stream else {}
        boxes = self.active_alert_boxes[detector_type]
        if self._multi_stream:
            if not isinstance(boxes, dict):
                if not create:
                    return None
                boxes = {}
                self.active_alert_boxes[detector_type] = boxes
            sid = int(stream_id)
            if create:
                return boxes.setdefault(sid, {})
            return boxes.get(sid)
        if not isinstance(boxes, dict):
            if not create:
                return None
            boxes = {}
            self.active_alert_boxes[detector_type] = boxes
        return boxes

    def _result_seen_mono(self, result):
        if isinstance(result, dict):
            try:
                return float(result.get('_display_received_mono') or time.monotonic())
            except Exception:
                pass
        return time.monotonic()

    def update_active_alert_box(
        self,
        detector_type,
        stream_id,
        track_id,
        bbox,
        label=None,
        bbox_format='xyxy',
        color=(0, 0, 255),
        result=None,
        coord_wh=None,
        info=None,
        is_recording=False,
        **extra,
    ):
        """Update the display-only alert box cache from a detection result."""
        if track_id is None or bbox is None:
            return False
        try:
            bbox_values = [float(v) for v in list(bbox)[:4]]
        except Exception:
            return False
        if len(bbox_values) < 4:
            return False

        store = self._alert_box_store(detector_type, stream_id, create=True)
        if store is None:
            return False
        seen_mono = self._result_seen_mono(result)
        entry = store.get(track_id)
        if entry is None:
            entry = {
                'start_time': result.get('timestamp') if isinstance(result, dict) else None,
                'created_mono': seen_mono,
            }
            store[track_id] = entry
        elif entry.get('bbox_format') != bbox_format:
            self._clear_bbox_smooth(detector_type, stream_id, track_id)

        if isinstance(label, (list, tuple)):
            chinese_text = label[0] if len(label) > 0 else ''
            english_text = label[1] if len(label) > 1 else ''
        elif isinstance(label, dict):
            chinese_text = label.get('chinese_text') or label.get('chinese') or ''
            english_text = label.get('english_text') or label.get('english') or ''
        else:
            chinese_text = str(label or '')
            english_text = str(extra.pop('english_text', '') or '')

        entry.update({
            'bbox': bbox_values,
            'bbox_format': bbox_format,
            'chinese_text': chinese_text,
            'english_text': english_text,
            'color': color,
            'coord_wh': coord_wh,
            'info': info if info is not None else entry.get('info', {}),
            'is_recording': bool(is_recording),
            'last_seen_mono': seen_mono,
            'last_detection_mono': seen_mono,
            'last_detection_frame_number': result.get('frame_number') if isinstance(result, dict) else None,
        })
        entry.update(extra)
        return True

    def _refresh_active_alert_box(
        self,
        detector_type,
        stream_id,
        track_id,
        bbox,
        result=None,
        coord_wh=None,
        bbox_format=None,
    ):
        store = self._alert_box_store(detector_type, stream_id, create=False)
        if not store or track_id not in store:
            return False
        entry = store[track_id]
        return self.update_active_alert_box(
            detector_type,
            stream_id,
            track_id,
            bbox,
            label=(entry.get('chinese_text', ''), entry.get('english_text', '')),
            bbox_format=bbox_format or entry.get('bbox_format', 'xyxy'),
            color=entry.get('color', (0, 0, 255)),
            result=result,
            coord_wh=coord_wh if coord_wh is not None else entry.get('coord_wh'),
            info=entry.get('info'),
            is_recording=entry.get('is_recording', False),
            **{k: v for k, v in entry.items() if k in ('extend_y2_by_height_ratio', 'disable_flow_tracking')}
        )

    def expire_stale_alert_boxes(self, stream_id=None):
        now = time.monotonic()
        changed_streams = set()
        detector_items = list(self.active_alert_boxes.items())
        for detector_type, boxes in detector_items:
            if self._multi_stream:
                if not isinstance(boxes, dict):
                    continue
                stream_items = (
                    [(int(stream_id), boxes.get(int(stream_id), {}))]
                    if stream_id is not None else list(boxes.items())
                )
            else:
                stream_items = [(0, boxes)]

            for sid, store in stream_items:
                if not isinstance(store, dict):
                    continue
                for track_id, alert in list(store.items()):
                    last_seen = alert.get('last_seen_mono', alert.get('created_mono', now))
                    try:
                        expired = now - float(last_seen) > self._alert_box_hold_s
                    except Exception:
                        expired = False
                    if not expired:
                        continue
                    self._clear_bbox_smooth(detector_type, sid, track_id)
                    store.pop(track_id, None)
                    changed_streams.add(int(sid))
        for sid in changed_streams:
            self._rendered_cell_cache.pop(sid, None)

    def _has_active_alert_boxes(self, stream_id):
        for detector_type in self._RENDER_DETECTORS:
            store = self._alert_box_store(detector_type, stream_id, create=False)
            if store:
                return True
        return False

    def _iter_active_alert_box_entries(self, stream_id):
        for detector_type in self._RENDER_DETECTORS:
            store = self._alert_box_store(detector_type, stream_id, create=False)
            if not store:
                continue
            for track_id, alert in list(store.items()):
                yield detector_type, track_id, alert

    def _clamp_display_bbox_xyxy(self, bbox, frame_shape):
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None
        fh, fw = frame_shape[:2]
        x1 = max(0.0, min(float(fw - 1), x1))
        y1 = max(0.0, min(float(fh - 1), y1))
        x2 = max(0.0, min(float(fw), x2))
        y2 = max(0.0, min(float(fh), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _alert_entry_detection_display_bbox(self, alert, frame_shape):
        bbox = alert.get('bbox')
        bbox_format = alert.get('bbox_format', 'xyxy')
        if bbox is None or len(bbox) < 4:
            return None
        disp_wh = (frame_shape[1], frame_shape[0])
        coord_wh = alert.get('coord_wh') or disp_wh
        scaled = self._scale_bbox_to_display(bbox, coord_wh, disp_wh, bbox_format)
        try:
            if bbox_format == 'xywh':
                x, y, w, h = [float(v) for v in scaled[:4]]
                xyxy = [x, y, x + w, y + h]
            else:
                xyxy = [float(v) for v in scaled[:4]]
        except Exception:
            return None
        return self._clamp_display_bbox_xyxy(xyxy, frame_shape)

    def _blend_alert_velocity(self, old_velocity, new_velocity):
        if not old_velocity or len(old_velocity) < 4:
            return list(new_velocity)
        a = self._alert_box_velocity_blend
        try:
            return [
                a * float(new_velocity[i]) + (1.0 - a) * float(old_velocity[i])
                for i in range(4)
            ]
        except Exception:
            return list(new_velocity)

    def _limit_alert_shift(self, bbox, dx, dy, ratio):
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            max_shift = max(4.0, max(box_w, box_h) * float(ratio))
            dx = max(-max_shift, min(max_shift, float(dx)))
            dy = max(-max_shift, min(max_shift, float(dy)))
        except Exception:
            dx, dy = 0.0, 0.0
        return dx, dy

    def _shift_display_bbox(self, bbox, dx, dy, frame_shape):
        return self._clamp_display_bbox_xyxy(
            [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy],
            frame_shape,
        )

    def _predict_alert_display_bbox(self, alert, frame_shape, now, ratio=None, max_dt=None):
        bbox = alert.get('display_bbox_xyxy')
        velocity = alert.get('display_bbox_velocity')
        if bbox is None or not velocity or len(velocity) < 4:
            return False
        try:
            last_mono = float(alert.get('display_bbox_mono') or now)
            dt = max(0.0, float(now) - last_mono)
            if max_dt is not None:
                dt = min(dt, max(0.0, float(max_dt)))
        except Exception:
            return False
        if dt <= 0.0:
            return False
        ratio = self._alert_box_track_max_shift_ratio if ratio is None else ratio
        try:
            dx = ((float(velocity[0]) + float(velocity[2])) * 0.5) * dt
            dy = ((float(velocity[1]) + float(velocity[3])) * 0.5) * dt
        except Exception:
            return False
        dx, dy = self._limit_alert_shift(bbox, dx, dy, ratio)
        shifted = self._shift_display_bbox(bbox, dx, dy, frame_shape)
        if shifted is None:
            return False
        alert['display_bbox_xyxy'] = shifted
        alert['display_bbox_mono'] = now
        return True

    def _sync_alert_display_bbox(self, alert, frame_shape, frame_number, now):
        detection_mono = alert.get('last_detection_mono', alert.get('last_seen_mono'))
        if alert.get('display_bbox_xyxy') is not None and alert.get('display_bbox_source_mono') == detection_mono:
            return False

        detection_bbox = self._alert_entry_detection_display_bbox(alert, frame_shape)
        if detection_bbox is None:
            return False

        prev_detection_bbox = alert.get('last_detection_display_bbox')
        prev_detection_mono = alert.get('last_detection_display_mono')
        try:
            if prev_detection_bbox is not None and prev_detection_mono is not None and detection_mono is not None:
                dt = float(detection_mono) - float(prev_detection_mono)
                if dt > 0.02:
                    new_velocity = [
                        (detection_bbox[i] - float(prev_detection_bbox[i])) / dt
                        for i in range(4)
                    ]
                    alert['display_bbox_velocity'] = self._blend_alert_velocity(
                        alert.get('display_bbox_velocity'),
                        new_velocity,
                    )
        except Exception:
            pass

        alert['last_detection_display_bbox'] = list(detection_bbox)
        alert['last_detection_display_mono'] = detection_mono
        alert['display_bbox_xyxy'] = list(detection_bbox)
        alert['display_bbox_mono'] = now
        alert['display_bbox_source_mono'] = detection_mono

        try:
            result_frame = alert.get('last_detection_frame_number')
            if self._alert_box_detection_lag_compensation and frame_number is not None and result_frame is not None and self.fps:
                lag_s = (int(frame_number) - int(result_frame)) / float(self.fps)
                lag_s = min(max(0.0, lag_s), self._alert_box_prediction_max_s)
                if lag_s > 0.0:
                    # 用检测间速度补偿“结果到达时已经落后当前预览帧”的时间差。
                    alert['display_bbox_mono'] = now - lag_s
                    self._predict_alert_display_bbox(
                        alert,
                        frame_shape,
                        now,
                        ratio=self._alert_box_predict_max_shift_ratio,
                        max_dt=lag_s,
                    )
        except Exception:
            pass
        return True

    def _track_alert_box_with_flow(self, prev_gray, gray, alert, frame_shape, now):
        bbox = alert.get('display_bbox_xyxy')
        if bbox is None:
            return False
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        except Exception:
            return False
        fh, fw = frame_shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        rx1, ry1, rx2, ry2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
        if rx2 - rx1 < 8 or ry2 - ry1 < 8:
            return False

        roi = prev_gray[ry1:ry2, rx1:rx2]
        try:
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=18,
                qualityLevel=0.01,
                minDistance=3,
                blockSize=3,
            )
        except Exception:
            return False
        if points is None or len(points) < 3:
            return False
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points[:, 0, 0] += float(rx1)
        points[:, 0, 1] += float(ry1)

        try:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
        except Exception:
            return False
        if next_points is None or status is None:
            return False
        valid = status.reshape(-1) == 1
        if int(np.count_nonzero(valid)) < 3:
            return False

        try:
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                next_points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            if back_points is not None and back_status is not None:
                fb_err = np.linalg.norm(back_points.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
                valid = valid & (back_status.reshape(-1) == 1) & (fb_err < 2.0)
        except Exception:
            pass
        if int(np.count_nonzero(valid)) < 3:
            return False

        shifts = next_points.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        med_x = float(np.median(shifts[:, 0]))
        med_y = float(np.median(shifts[:, 1]))
        spread = np.linalg.norm(shifts - np.array([med_x, med_y], dtype=np.float32), axis=1)
        max_spread = max(2.0, max(box_w, box_h) * 0.10)
        inliers = spread <= max_spread
        if int(np.count_nonzero(inliers)) >= 3:
            shifts = shifts[inliers]
        dx = float(np.median(shifts[:, 0]))
        dy = float(np.median(shifts[:, 1]))
        dx, dy = self._limit_alert_shift(bbox, dx, dy, self._alert_box_track_max_shift_ratio)
        shifted = self._shift_display_bbox(bbox, dx, dy, frame_shape)
        if shifted is None:
            return False

        try:
            dt = max(1e-3, now - float(alert.get('display_bbox_mono') or now))
            new_velocity = [dx / dt, dy / dt, dx / dt, dy / dt]
            alert['display_bbox_velocity'] = self._blend_alert_velocity(
                alert.get('display_bbox_velocity'),
                new_velocity,
            )
        except Exception:
            pass
        alert['display_bbox_xyxy'] = shifted
        alert['display_bbox_mono'] = now
        return True

    def _track_active_alert_boxes(self, frame, stream_id, frame_number=None):
        sid = int(stream_id)
        if not self._has_active_alert_boxes(sid):
            self._alert_track_prev_gray.pop(sid, None)
            return
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return
        prev_gray = self._alert_track_prev_gray.get(sid)
        if prev_gray is not None and prev_gray.shape != gray.shape:
            prev_gray = None
        now = time.monotonic()

        for _, _, alert in self._iter_active_alert_box_entries(sid):
            reset_to_detection = self._sync_alert_display_bbox(alert, frame.shape, frame_number, now)
            if reset_to_detection or alert.get('disable_flow_tracking') or not self._alert_box_tracking_enabled or prev_gray is None:
                continue
            if self._track_alert_box_with_flow(prev_gray, gray, alert, frame.shape, now):
                continue
            if self._alert_box_prediction_max_s > 0.0:
                self._predict_alert_display_bbox(
                    alert,
                    frame.shape,
                    now,
                    ratio=self._alert_box_track_max_shift_ratio,
                    max_dt=self._alert_box_prediction_max_s,
                )

        self._alert_track_prev_gray[sid] = gray

    def _draw_active_alert_entry(self, frame, detector_type, stream_id, track_id, alert):
        bbox = alert.get('display_bbox_xyxy')
        if bbox is None:
            bbox = self._alert_entry_detection_display_bbox(alert, frame.shape)
        if bbox is None or len(bbox) < 4:
            return frame

        try:
            extend_ratio = float(alert.get('extend_y2_by_height_ratio') or 0.0)
        except Exception:
            extend_ratio = 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if extend_ratio > 0:
            y2 = y2 + max(1.0, y2 - y1) * extend_ratio

        bbox = self._clamp_display_bbox_xyxy([x1, y1, x2, y2], frame.shape)
        if bbox is None:
            return frame
        return self._draw_unified_alert_box(
            frame,
            bbox,
            alert.get('chinese_text', ''),
            alert.get('english_text', ''),
            color=alert.get('color', (0, 0, 255)),
        )

    def draw_active_alert_boxes(self, frame, stream_id, frame_number=None):
        """Draw cached alert boxes every preview frame, independent of detection FPS."""
        self.expire_stale_alert_boxes(stream_id)
        self._track_active_alert_boxes(frame, stream_id, frame_number=frame_number)
        for detector_type, track_id, alert in self._iter_active_alert_box_entries(stream_id):
            frame = self._draw_active_alert_entry(frame, detector_type, stream_id, track_id, alert)
        return frame

    def _bump_result_version(self, detector_type, stream_id):
        if detector_type not in self._result_versions:
            return
        sid = int(stream_id)
        versions = self._result_versions[detector_type]
        versions[sid] = versions.get(sid, 0) + 1
        self._rendered_cell_cache.pop(sid, None)

    def _cell_render_key(self, stream_id, frame_data):
        sid = int(stream_id)
        frame_number = frame_data.get('frame_number') if frame_data else None
        result_versions = tuple(
            self._result_versions.get(detector_type, {}).get(sid, 0)
            for detector_type in self._RENDER_DETECTORS
        )
        # Refresh status overlays at most once per second even if the video frame is unchanged.
        status_tick = int(time.monotonic())
        return (frame_number, result_versions, status_tick)

    def _render_cell_for_stream(self, stream_id, frame_data):
        sid = int(stream_id)
        if not frame_data or 'frame' not in frame_data or frame_data.get('frame') is None:
            frame_cell = self._blank_cell.copy()
            try:
                return self._overlay_stream_status(frame_cell, sid)
            except Exception:
                return frame_cell

        key = self._cell_render_key(sid, frame_data)
        cached = self._rendered_cell_cache.get(sid)
        if cached and cached.get('key') == key:
            return cached['frame']

        frame_cell = frame_data['frame']
        orig_wh = (frame_data.get('original_width'), frame_data.get('original_height'))
        if None in orig_wh:
            orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
        frame_cell = self._render_results(
            frame_cell,
            frame_data.get('frame_number'),
            frame_data.get('timestamp'),
            stream_id=sid,
            original_size=orig_wh,
        )
        try:
            frame_cell = self._overlay_stream_status(frame_cell, sid)
        except Exception:
            pass
        self._rendered_cell_cache[sid] = {'key': key, 'frame': frame_cell}
        return frame_cell

    def _clear_sticky_for_stream(self, stream_id):
        for detector_type in self._RENDER_DETECTORS:
            self._clear_sticky_for_detector_stream(detector_type, stream_id)

    def _clear_sticky_for_detector_stream(self, detector_type, stream_id):
        if detector_type not in self._RENDER_DETECTORS:
            return
        stream_id = int(stream_id)
        if self._multi_stream and isinstance(self.sticky_results.get(detector_type), dict):
            self.sticky_results[detector_type].pop(stream_id, None)
            if isinstance(self.latest_results.get(detector_type), dict):
                self.latest_results[detector_type].pop(stream_id, None)
            if isinstance(self.active_alerts.get(detector_type), dict):
                self.active_alerts[detector_type].pop(stream_id, None)
            if isinstance(self.triggered_alarms.get(detector_type), dict):
                self.triggered_alarms[detector_type].pop(stream_id, None)
        elif stream_id == 0:
            self.sticky_results[detector_type] = None
            self.latest_results[detector_type] = None
            self.active_alerts[detector_type] = {}
            self.triggered_alarms[detector_type] = set()
        self._active_result_streams.get(detector_type, set()).discard(stream_id)
        if detector_type in self._result_by_frame:
            self._result_by_frame[detector_type].pop(stream_id, None)
        self._result_versions.get(detector_type, {}).pop(stream_id, None)
        self._rendered_cell_cache.pop(stream_id, None)
        self._alert_track_prev_gray.pop(stream_id, None)
        self._clear_bbox_smooth_for(detector_type, stream_id)

    def _clear_sticky_for_failed_streams(self):
        for stream_id in range(self.stream_count):
            if self._get_stream_connection_state(stream_id) == "连接失败":
                self._clear_sticky_for_stream(stream_id)

    def _overlay_stream_status(self, frame, stream_id):
        """不再在画面上叠加通道/FPS/连接状态信息框，直接返回原画面。"""
        return frame
    
    def _load_chinese_font(self, font_path_config=None):
        """加载中文字体，优先使用配置路径，避免报警中文乱码"""
        # 项目根目录（core 的上一级）
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        _cwd = os.getcwd()
        
        candidates = []
        if font_path_config:
            path = font_path_config.strip()
            if path:
                if os.path.isabs(path):
                    candidates.append(path)
                candidates.append(os.path.join(_project_root, path))
                candidates.append(os.path.join(_cwd, path))
                candidates.append(path)
        candidates.extend([
            os.path.join(_project_root, 'simhei.ttf'),
            os.path.join(_cwd, 'simhei.ttf'),
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            '/usr/share/fonts/truetype/simhei/simhei.ttf',
            '/usr/share/fonts/chinese/TrueType/simhei.ttf',
            '/System/Library/Fonts/PingFang.ttc',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            './simhei.ttf',
        ])
        
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    ImageFont.truetype(path, 18)
                    self.font_path = path
                    logger.info(f"Chinese font loaded: {path}")
                    return
                except Exception:
                    continue

        logger.warning("Chinese font not found; alarm labels in Chinese may be garbled. Add simhei.ttf to project root or set output.font_path in config.")

    def _get_label_sprite(self, chinese_text, english_text, chinese_font_size, english_font_size, color, bold=False):
        stroke_w = 1 if bold else 0
        # 加粗模式下 sprite 尺寸需要额外留出描边空间
        pad = stroke_w * 2
        cache_key = (
            chinese_text or '',
            english_text or '',
            int(chinese_font_size),
            int(english_font_size),
            tuple(int(c) for c in color),
            bool(self.font_path and chinese_text),
            int(stroke_w),
        )
        cached = self._label_sprite_cache.get(cache_key)
        if cached is not None:
            return cached

        use_chinese = bool(self.font_path and chinese_text)
        font_cn = self._get_cached_font(self.font_path, chinese_font_size) if use_chinese else None
        font_en = self._get_cached_font(self._english_font_path, english_font_size)
        measure_image = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        measure_draw = ImageDraw.Draw(measure_image)

        if use_chinese and font_cn:
            bbox_cn = measure_draw.textbbox((0, 0), chinese_text, font=font_cn, stroke_width=stroke_w)
            cn_width = max(0, bbox_cn[2] - bbox_cn[0])
            cn_height = max(0, bbox_cn[3] - bbox_cn[1])
        else:
            cn_width = 0
            cn_height = 0
        bbox_en = measure_draw.textbbox((0, 0), english_text, font=font_en, stroke_width=stroke_w)
        en_width = max(0, bbox_en[2] - bbox_en[0])
        en_height = max(0, bbox_en[3] - bbox_en[1])
        total_width = max(1, cn_width, en_width) + 4 + pad
        # 中文+英文叠加 + 行间距 + 底部内边距，避免英文 descender 被裁切
        total_height = max(1, (cn_height + 4 + en_height + 3) if use_chinese and font_cn else (en_height + 3)) + pad

        sprite = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        text_color = (255, 255, 255, 255)
        # 描边颜色：深色半透明，形成文字外轮廓使文字更粗更醒目
        stroke_color = (0, 0, 0, 200) if bold else None
        text_y = pad // 2
        if use_chinese and font_cn:
            draw.text((pad // 2, text_y), chinese_text, font=font_cn, fill=text_color,
                      stroke_width=stroke_w, stroke_fill=stroke_color)
            text_y += cn_height + 4
        draw.text((pad // 2, text_y), english_text, font=font_en, fill=text_color,
                  stroke_width=stroke_w, stroke_fill=stroke_color)
        rgba = np.array(sprite, dtype=np.uint8)
        self._label_sprite_cache[cache_key] = rgba
        return rgba

    def _blend_rgba(self, frame, overlay_rgba, x, y):
        if overlay_rgba is None or overlay_rgba.size == 0:
            return frame
        frame_h, frame_w = frame.shape[:2]
        over_h, over_w = overlay_rgba.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(frame_w, int(x) + over_w)
        y1 = min(frame_h, int(y) + over_h)
        if x0 >= x1 or y0 >= y1:
            return frame

        overlay = overlay_rgba[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)]
        alpha = overlay[..., 3:4].astype(np.float32) / 255.0
        if not np.any(alpha):
            return frame

        roi = frame[y0:y1, x0:x1].astype(np.float32)
        fg = overlay[..., :3].astype(np.float32)
        blended = roi * (1.0 - alpha) + fg * alpha
        frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    def _begin_alert_render_for_stream(self, stream_id):
        try:
            sid = int(stream_id)
        except Exception:
            sid = 0
        self._current_render_stream_id = sid
        self._alert_draw_counts[sid] = 0
        self._alert_label_counts[sid] = 0

    def _reserve_alert_box_slot(self):
        sid = getattr(self, '_current_render_stream_id', 0)
        used = self._alert_draw_counts.get(sid, 0)
        if self._max_alert_boxes_per_stream <= 0:
            self._alert_draw_counts[sid] = used + 1
            return True
        if used >= self._max_alert_boxes_per_stream:
            return False
        self._alert_draw_counts[sid] = used + 1
        return True

    def _reserve_alert_label_slot(self):
        sid = getattr(self, '_current_render_stream_id', 0)
        used = self._alert_label_counts.get(sid, 0)
        if self._max_alert_labels_per_stream <= 0:
            self._alert_label_counts[sid] = used + 1
            return True
        if used >= self._max_alert_labels_per_stream:
            return False
        self._alert_label_counts[sid] = used + 1
        return True

    def _encode_mjpeg_frame(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._mjpeg_quality])
        if not ok:
            return None
        return buffer.tobytes()

    def _start_jpeg_encoder(self):
        if self.preview_transport != 'mjpeg' or self._jpeg_encoder_thread is not None:
            return
        self._jpeg_encoder_running = True
        self._jpeg_encoder_thread = threading.Thread(
            target=self._jpeg_encoder_worker,
            daemon=True,
            name="DisplayJpegEncoder",
        )
        self._jpeg_encoder_thread.start()

    def _stop_jpeg_encoder(self):
        self._jpeg_encoder_running = False
        if self._jpeg_encoder_thread is not None:
            self._jpeg_encoder_thread.join(timeout=2.0)
            self._jpeg_encoder_thread = None

    def _start_raw_jpeg_encoder(self):
        if self.raw_output_queue is None or self._raw_jpeg_encoder_thread is not None:
            return
        self._raw_jpeg_encoder_running = True
        self._raw_jpeg_encoder_thread = threading.Thread(
            target=self._raw_jpeg_encoder_worker,
            daemon=True,
            name="RawDetectionJpegEncoder",
        )
        self._raw_jpeg_encoder_thread.start()

    def _stop_raw_jpeg_encoder(self):
        self._raw_jpeg_encoder_running = False
        if self._raw_jpeg_encoder_thread is not None:
            self._raw_jpeg_encoder_thread.join(timeout=2.0)
            self._raw_jpeg_encoder_thread = None

    def _submit_jpeg_frame(self, frame):
        try:
            if self._jpeg_input_queue.full():
                self._jpeg_input_queue.get_nowait()
            self._jpeg_input_queue.put(frame.copy(), block=False)
        except Exception:
            pass

    def _submit_raw_jpeg_frame(self, frame):
        if self._raw_jpeg_input_queue is None:
            return
        try:
            if self._raw_jpeg_input_queue.full():
                self._raw_jpeg_input_queue.get_nowait()
            self._raw_jpeg_input_queue.put(frame.copy(), block=False)
        except Exception:
            pass

    def _jpeg_encoder_worker(self):
        while self._jpeg_encoder_running:
            try:
                frame = self._jpeg_input_queue.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                time.sleep(0.01)
                continue
            encoded = self._encode_mjpeg_frame(frame)
            if encoded is None:
                continue
            try:
                if self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put(encoded, block=False)
            except Exception:
                pass

    def _raw_jpeg_encoder_worker(self):
        while self._raw_jpeg_encoder_running:
            try:
                frame = self._raw_jpeg_input_queue.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                time.sleep(0.01)
                continue
            encoded = self._encode_mjpeg_frame(frame)
            if encoded is None:
                continue
            try:
                if self.raw_output_queue.full():
                    self.raw_output_queue.get_nowait()
                self.raw_output_queue.put(encoded, block=False)
            except Exception:
                pass

    def _draw_bilingual_label(self, frame, bbox, chinese_text, english_text, color=(0, 0, 255)):
        """绘制报警标签（仅英文）。"""
        chinese_text = ""  # 只保留英文标签
        x1, y1, _, _ = map(int, bbox)
        fh, fw = frame.shape[:2]
        base = int(max(9, min(20, round(min(fw, fh) * 0.045))))
        english_font_size = max(11, int(round(base * 1.0)))
        label_start = time.perf_counter()
        sprite = self._get_label_sprite(
            "",             # chinese_text: 已移除中文
            english_text,
            0,              # chinese_font_size: 未使用
            english_font_size,
            color,
            bold=True,
        )
        label_y = y1 - sprite.shape[0] - 4
        self._blend_rgba(frame, sprite, x1, label_y)
        self._perf_add('label_ms', (time.perf_counter() - label_start) * 1000.0)
        return frame

    def _draw_unified_alert_box(self, frame, bbox_xyxy, chinese_text, english_text, color=(0, 0, 255)):
        """
        统一的报警框样式（所有检测器一致）：
        - 坐标裁剪到画面内
        - 无框内背景色（按需求更清爽）
        - 细边框 + 角标
        - 间隔显示（闪烁/周期性显示，避免每帧都渲染）
        - 统一双语标签字号策略
        """
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox_xyxy[:4]]
        except Exception:
            return frame
        fh, fw = frame.shape[:2]
        x1 = int(max(0, min(fw - 1, x1)))
        y1 = int(max(0, min(fh - 1, y1)))
        x2 = int(max(0, min(fw - 1, x2)))
        y2 = int(max(0, min(fh - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            return frame
        if not self._reserve_alert_box_slot():
            return frame

        # 间隔显示：当模式为 blink 时，按设定频率闪烁显示报警框
        # 仅影响“报警框与文字”，不影响底层视频刷新
        if getattr(self, 'alert_box_mode', 'follow') == 'blink':
            try:
                blink_hz = float(getattr(self, 'alert_box_blink_hz', 1.0) or 1.0)
            except Exception:
                blink_hz = 1.0
            if blink_hz > 0:
                period = 1.0 / blink_hz
                if (time.time() % period) > (period * 0.5):
                    return frame

        overlay_start = time.perf_counter()
        merged_simple = self._multi_stream and self._merged_simple_alert_boxes
        if merged_simple:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            self._perf_add('overlay_ms', (time.perf_counter() - overlay_start) * 1000.0)
            self._perf_inc_alert_boxes()
            self._alert_label_counts[getattr(self, '_current_render_stream_id', 0)] = (
                self._alert_label_counts.get(getattr(self, '_current_render_stream_id', 0), 0) + 1
            )
            frame = self._draw_bilingual_label(frame, [x1, y1, x2, y2], chinese_text, english_text, color=color)
            return frame

        # 四宫格走轻量矩形以保障 20 FPS；单路调试时保留完整角标样式。
        thickness = 2 if min(fw, fh) <= 400 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        corner = max(8, int(min(x2 - x1, y2 - y1) * 0.18))
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)
        self._perf_add('overlay_ms', (time.perf_counter() - overlay_start) * 1000.0)
        self._perf_inc_alert_boxes()

        if self._reserve_alert_label_slot():
            frame = self._draw_bilingual_label(frame, [x1, y1, x2, y2], chinese_text, english_text, color=color)
        return frame

    def _raw_model_box_style(self, model_key, class_id):
        model_key = str(model_key or "")
        class_id = int(class_id)
        if model_key == "crowd_person":
            if class_id == 0:
                return (0, 255, 255), "person"
            if class_id == 1:
                return (0, 0, 255), "person_no_helmet"
            return (0, 255, 255), f"person:{class_id}"
        if model_key in ("helmet_detection", "ventilator_helmet"):
            if class_id == 0:
                return (0, 255, 255), "helmet"
            if class_id == 1:
                return (0, 0, 255), "no_helmet"
            return (0, 255, 255), f"helmet:{class_id}"
        if model_key == "ventilator_equipment":
            if class_id == 1:
                return (0, 255, 0), "mask"
            if class_id == 0:
                return (255, 0, 0), "tank"
            return (255, 255, 0), f"equipment:{class_id}"
        if model_key in ("window_door_inside", "window_door_outside", "window_door_detection"):
            labels = {
                0: "window_open",
                1: "window_close",
                2: "door_close",
                3: "door_open",
            }
            colors = {
                0: (0, 0, 255),
                1: (0, 180, 0),
                2: (0, 180, 0),
                3: (0, 0, 255),
            }
            return colors.get(class_id, (255, 255, 0)), labels.get(class_id, f"window_door:{class_id}")
        return (255, 255, 0), f"{model_key}:{class_id}"

    def _draw_raw_model_box_label(self, frame, x1, y1, label, color):
        try:
            label = str(label or "")
        except Exception:
            return frame
        if not label:
            return frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_h = text_size[1] + baseline + 6
        label_w = text_size[0] + 8
        fh, fw = frame.shape[:2]
        x1 = int(max(0, min(fw - 1, x1)))
        y1 = int(max(0, min(fh - 1, y1)))
        y0 = y1 - label_h if y1 - label_h >= 0 else y1
        y_text = y0 + text_size[1] + 3
        x2 = int(min(fw, x1 + label_w))
        y2 = int(min(fh, y0 + label_h))
        cv2.rectangle(frame, (x1, y0), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1 + 4, y_text), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return frame

    def _draw_raw_model_box(self, frame, bbox, label, color, coord_wh, disp_wh):
        scaled = self._scale_bbox_to_display(bbox, coord_wh, disp_wh, 'xyxy')
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in scaled[:4]]
        except Exception:
            return frame
        clamped = self._clamp_display_bbox_xyxy([x1, y1, x2, y2], frame.shape)
        if clamped is None:
            return frame
        x1, y1, x2, y2 = [int(round(v)) for v in clamped[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return self._draw_raw_model_box_label(frame, x1, y1, label, color)

    def _draw_raw_model_detections(self, frame, result, coord_wh, disp_wh, force=False):
        if not force and not self._raw_model_boxes_enabled:
            return frame
        raw = result.get('raw_model_detections') if isinstance(result, dict) else None
        if not isinstance(raw, dict):
            return frame
        for model_key, detections in raw.items():
            if not isinstance(detections, list):
                continue
            for det in detections:
                if not isinstance(det, dict):
                    continue
                try:
                    conf = float(det.get('confidence') or 0.0)
                except Exception:
                    conf = 0.0
                if conf < self._raw_model_boxes_min_conf:
                    continue
                bbox = det.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                try:
                    class_id = int(det.get('class_id', -1))
                except Exception:
                    class_id = -1
                color, class_label = self._raw_model_box_style(model_key, class_id)
                class_name = det.get('class_name') or class_label
                label = f"{model_key}:{class_name} {conf:.2f}"
                frame = self._draw_raw_model_box(frame, bbox, label, color, coord_wh, disp_wh)
        return frame

    def _render_raw_model_results(self, frame, frame_number, stream_id=0, original_size=None):
        detector_results = {
            detector_type: self._get_result_for_frame(detector_type, stream_id, frame_number)
            for detector_type in self._RENDER_DETECTORS
        }
        if not any(result is not None for result in detector_results.values()):
            return frame

        rendered = frame.copy()
        disp_wh = (frame.shape[1], frame.shape[0])
        orig_wh = original_size if original_size and None not in original_size else disp_wh
        for detector_type, result in detector_results.items():
            if result is None:
                continue
            coord_wh = self._result_coord_wh(result, orig_wh)
            raw = result.get('raw_model_detections') if isinstance(result, dict) else None
            if isinstance(raw, dict):
                rendered = self._draw_raw_model_detections(rendered, result, coord_wh, disp_wh, force=True)
            else:
                rendered = self._draw_generic_detection_boxes(rendered, result, coord_wh, disp_wh, detector_type)
        return rendered

    def _draw_generic_detection_boxes(self, frame, result, coord_wh, disp_wh, detector_type):
        detections = result.get('detections') if isinstance(result, dict) else None
        if not isinstance(detections, (list, tuple)):
            detections = result.get('current_detections') if isinstance(result, dict) else None
        if not isinstance(detections, (list, tuple)):
            return frame
        color, _ = self._raw_model_box_style(detector_type, 0)
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get('bbox') or det.get('box')
            if not bbox or len(bbox) < 4:
                continue
            try:
                conf = float(det.get('confidence') or det.get('score') or 0.0)
            except Exception:
                conf = 0.0
            if conf < self._raw_model_boxes_min_conf:
                continue
            class_name = det.get('class_name') or det.get('label') or detector_type
            label = f"{detector_type}:{class_name} {conf:.2f}" if conf > 0 else f"{detector_type}:{class_name}"
            frame = self._draw_raw_model_box(frame, bbox, label, color, coord_wh, disp_wh)
        return frame

    def _build_raw_composite_frame(self):
        blank_cell = self._blank_cell
        if self.stream_count == 1:
            fd = self.latest_frames.get(0)
            if not fd or 'frame' not in fd or fd.get('frame') is None:
                return cv2.resize(blank_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            frame_cell = fd['frame']
            orig_wh = (fd.get('original_width'), fd.get('original_height'))
            if None in orig_wh:
                orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
            rendered = self._render_raw_model_results(
                frame_cell, fd.get('frame_number'), stream_id=0, original_size=orig_wh
            )
            return rendered

        composite = self._composite_canvas
        composite[:] = blank_cell[0, 0]
        for idx in range(GRID_CELL_COUNT):
            if idx < self.stream_count:
                fd = self.latest_frames.get(idx)
                if fd and 'frame' in fd and fd.get('frame') is not None:
                    frame_cell = fd['frame']
                    orig_wh = (fd.get('original_width'), fd.get('original_height'))
                    if None in orig_wh:
                        orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
                    frame_cell = self._render_raw_model_results(
                        frame_cell, fd.get('frame_number'), stream_id=idx, original_size=orig_wh
                    )
                else:
                    frame_cell = blank_cell.copy()
            else:
                frame_cell = blank_cell.copy()
            row = idx // GRID_COLUMNS
            col = idx % GRID_COLUMNS
            y0 = row * CELL_H
            x0 = col * CELL_W
            composite[y0:y0 + CELL_H, x0:x0 + CELL_W] = frame_cell
        return composite.copy()
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_local_preview(self):
        if not self.local_preview_enabled:
            return False
        if self._local_preview_ready:
            return True

        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            reason = "local preview requested but DISPLAY/WAYLAND_DISPLAY is not set"
            if self._local_preview_error != reason:
                logger.warning(reason)
            self._local_preview_error = reason
            self._set_preview_status(
                healthy=False,
                encoder_backend="local",
                unhealthy_reason=reason,
            )
            return False

        try:
            cv2.namedWindow(self._local_preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self._local_preview_window_name,
                COMPOSITE_OUTPUT_W,
                COMPOSITE_OUTPUT_H,
            )
            self._local_preview_ready = True
            self._local_preview_error = None
            self._set_preview_status(
                healthy=True,
                encoder_backend="local",
                unhealthy_reason=None,
            )
            logger.info("Local preview window initialized: %s", self._local_preview_window_name)
            return True
        except Exception as exc:
            reason = f"local preview init failed: {exc}"
            logger.error(reason)
            self._local_preview_error = reason
            self._local_preview_ready = False
            self._set_preview_status(
                healthy=False,
                encoder_backend="local",
                unhealthy_reason=reason,
            )
            return False

    def _show_local_preview(self, frame: np.ndarray):
        if not self.local_preview_enabled:
            return
        if not self._init_local_preview():
            return

        try:
            cv2.imshow(self._local_preview_window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                logger.info("Local preview requested exit via keyboard")
                self.running = False
        except Exception as exc:
            reason = f"local preview render failed: {exc}"
            logger.error(reason)
            self._local_preview_error = reason
            self._local_preview_ready = False
            self._set_preview_status(
                healthy=False,
                encoder_backend="local",
                unhealthy_reason=reason,
            )

    def _close_local_preview(self):
        if not self.local_preview_enabled:
            return
        if not self._local_preview_ready:
            return
        try:
            cv2.destroyWindow(self._local_preview_window_name)
            cv2.waitKey(1)
        except Exception:
            pass
        self._local_preview_ready = False
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"Received signal {signum}, stopping gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """启动展示服务"""
        logger.info("Display service starting...")
        self.setup_signal_handlers()
        
        self.running = True
        self.run()
    
    def run(self):
        """主循环：汇总结果、渲染、处理警报（支持一路到四路）"""
        logger.info("Display service running... (stream_count=%d)" % self.stream_count)
        if self.hls_writer is not None and self._multi_stream:
            self._hls_start_attempted = True
            self._hls_started = bool(self.hls_writer.start(COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H))
        self._start_jpeg_encoder()
        self._start_raw_jpeg_encoder()
        
        fps_start_time = time.time()
        fps_frame_count = 0
        processing_times = []
        # 预览与编码输出目标 FPS（节流避免过载）
        target_output_fps = min(25.0, float(self.fps))
        min_output_interval = 1.0 / target_output_fps if target_output_fps > 0 else 0.0
        next_output_mono = time.monotonic()
        drain_timeout = 0.001 if self._multi_stream else 0.0
        
        while self.running:
            self._check_control_commands()
            
            frame_start_time = time.time()

            now_mono = time.monotonic()
            if not self.latest_frames:
                absorbed = self._absorb_latest_frames(wait_timeout=0.1, drain_timeout=drain_timeout)
                if absorbed <= 0:
                    continue
            else:
                remaining = next_output_mono - now_mono
                wait_timeout = min(0.01, remaining) if remaining > 0.0 else 0.0
                self._absorb_latest_frames(wait_timeout=wait_timeout, drain_timeout=drain_timeout)
                now_mono = time.monotonic()
                if min_output_interval > 0.0 and now_mono < next_output_mono:
                    continue
                # 输出前再快速吸收一轮，尽量让本次四宫格使用最新缓存帧
                self._absorb_latest_frames(wait_timeout=0.0, drain_timeout=0.0)
            
            if self._multi_stream:
                # 允许“部分通道在线”：只要任意一路有帧就开始输出合成画面，
                # 缺失通道在 _build_composite_frame 内用黑屏占位，避免因 RTSP 失败导致整屏无输出。
                if not self.latest_frames:
                    continue
                # 选择一个参考帧用于 frame_number/timestamp（渲染时每格仍按各自帧号取结果）
                frame_data = self.latest_frames.get(0) or next(iter(self.latest_frames.values()))
            else:
                frame_data = self.latest_frames.get(0)
                if not frame_data:
                    continue
            
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            raw_frame = None
            if self._multi_stream:
                collect_start = time.time()
                self._collect_detection_results()
                collect_time = (time.time() - collect_start) * 1000
                self._perf_add('collect_ms', collect_time)
                render_start = time.time()
                rendered_frame = self._build_composite_frame()
                if self.raw_output_queue is not None:
                    raw_frame = self._build_raw_composite_frame()
                render_time = (time.time() - render_start) * 1000
                self._perf_add('composite_ms', render_time)
                # 使用参考帧的时间戳（避免 dict 顺序导致跳变）
                frame_number = frame_data.get('frame_number')
                timestamp = frame_data.get('timestamp')
            else:
                collect_start = time.time()
                self._collect_detection_results()
                collect_time = (time.time() - collect_start) * 1000
                self._perf_add('collect_ms', collect_time)
                render_start = time.time()
                orig_wh = (frame_data.get('original_width'), frame_data.get('original_height'))
                if None in orig_wh:
                    orig_wh = (frame.shape[1], frame.shape[0])
                rendered_frame = self._render_results(
                    frame, frame_number, timestamp, stream_id=0, original_size=orig_wh
                )
                if self.raw_output_queue is not None:
                    raw_frame = self._render_raw_model_results(
                        frame, frame_number, stream_id=0, original_size=orig_wh
                    )
                # 一路模式保留输入视频流尺寸，避免 1920x1080 预览被固定缩放。
                try:
                    rendered_frame = self._overlay_stream_status(rendered_frame, 0)
                except Exception:
                    pass
                render_time = (time.time() - render_start) * 1000
                self._perf_add('composite_ms', render_time)
            
            self.frame_count += 1
            fps_frame_count += 1
            
            buffer_start = time.time()
            if self._should_buffer_for_alerts():
                buffer_due = self._alarm_buffer_due()
                if buffer_due:
                    self.video_buffer.add_frame(rendered_frame, frame_number, timestamp)
                self._process_detections(timestamp)
                if buffer_due:
                    self.video_buffer.update(rendered_frame, frame_number, timestamp)
            else:
                # 无检测输出时跳过告警缓冲，提升预览帧率。
                pass
            buffer_time = (time.time() - buffer_start) * 1000
            self._perf_add('buffer_ms', buffer_time)
            
            output_start = time.time()
            output_start_mono = time.monotonic()
            if raw_frame is not None:
                self._submit_raw_jpeg_frame(raw_frame)
            if self.preview_transport == 'mjpeg':
                self._submit_jpeg_frame(rendered_frame)
            elif self.preview_transport != 'local':
                try:
                    if self.output_queue.full():
                        self.output_queue.get_nowait()
                    self.output_queue.put(rendered_frame, block=False)
                except Exception:
                    pass
            else:
                self._show_local_preview(rendered_frame)
            if self.hls_writer is not None:
                if not self._hls_start_attempted:
                    height, width = rendered_frame.shape[:2]
                    self._hls_start_attempted = True
                    self._hls_started = bool(self.hls_writer.start(width, height))
                if self._hls_started:
                    self.hls_writer.write(rendered_frame)
            if min_output_interval > 0.0:
                next_output_mono = max(next_output_mono + min_output_interval, output_start_mono)
            else:
                next_output_mono = output_start_mono
            output_time = (time.time() - output_start) * 1000
            self._perf_add('output_ms', output_time)
            
            frame_process_time = (time.time() - frame_start_time) * 1000
            processing_times.append(frame_process_time)
            
            if fps_frame_count >= 30:
                frames_in_window = fps_frame_count
                elapsed_time = time.time() - fps_start_time
                actual_fps = fps_frame_count / elapsed_time
                avg_process_time = sum(processing_times) / len(processing_times)
                max_process_time = max(processing_times)
                perf_window = self._perf_window
                stream_hits = ", ".join(
                    f"ch{idx + 1}={count}"
                    for idx, count in enumerate(perf_window['stream_hits'])
                    if count > 0
                ) or "none"
                
                active_recordings = self.video_buffer.get_active_recordings()
                
                logger.info(f"🖥️ [DISPLAY SERVICE] Frames: {self.frame_count}")
                logger.info(f"   ├─ Actual FPS: {actual_fps:.2f}")
                logger.info(f"   ├─ Avg process time: {avg_process_time:.1f}ms "
                        f"(collect: {collect_time:.1f}ms, render: {render_time:.1f}ms, "
                        f"buffer: {buffer_time:.1f}ms, output: {output_time:.1f}ms)")
                logger.info(f"   ├─ Max process time: {max_process_time:.1f}ms")
                logger.info(f"   ├─ Active recordings: {active_recordings}")
                logger.info(
                    "   ├─ Avg stage breakdown: drain=%.1fms, resize=%.1fms, collect=%.1fms, "
                    "lookup=%.1fms, overlay=%.1fms, label=%.1fms, composite=%.1fms, "
                    "buffer=%.1fms, output=%.1fms",
                    perf_window['queue_drain_ms'] / max(1, frames_in_window),
                    perf_window['resize_ms'] / max(1, frames_in_window),
                    perf_window['collect_ms'] / max(1, frames_in_window),
                    perf_window['result_lookup_ms'] / max(1, frames_in_window),
                    perf_window['overlay_ms'] / max(1, frames_in_window),
                    perf_window['label_ms'] / max(1, frames_in_window),
                    perf_window['composite_ms'] / max(1, frames_in_window),
                    perf_window['buffer_ms'] / max(1, frames_in_window),
                    perf_window['output_ms'] / max(1, frames_in_window),
                )
                logger.info(
                    "   ├─ Window stats: input_frames=%d, alert_boxes=%d, stream_hits=%s",
                    perf_window['input_frames'],
                    perf_window['alert_boxes'],
                    stream_hits,
                )
                try:
                    frame_qsize = max((q.qsize() for q in self.frame_queues[:self.stream_count]), default=0)
                except Exception:
                    frame_qsize = 0
                logger.info(f"   └─ Queue sizes: frame={frame_qsize}, output={self.output_queue.qsize()}")
                log_preview_fps("四宫格合成", actual_fps, target_fps=self.fps, stream_count=self.stream_count)
                try:
                    self.preview_status["compose_fps"] = float(actual_fps)
                    self.preview_status["last_frame_ts"] = (
                        timestamp.isoformat() if isinstance(timestamp, datetime) else None
                    )
                    if self.preview_transport == 'local':
                        self.preview_status["healthy"] = bool(self._local_preview_ready and not self._local_preview_error)
                        self.preview_status["encoder_backend"] = "local"
                        self.preview_status["encode_in_fps"] = float(actual_fps)
                        self.preview_status["last_segment_ts"] = None
                        self.preview_status["playlist_age_ms"] = None
                        self.preview_status["unhealthy_reason"] = self._local_preview_error
                    elif self.preview_transport != 'hls':
                        self.preview_status["healthy"] = True
                        self.preview_status["encoder_backend"] = "mjpeg"
                        self.preview_status["encode_in_fps"] = float(actual_fps)
                        self.preview_status["last_segment_ts"] = None
                        self.preview_status["playlist_age_ms"] = None
                        self.preview_status["unhealthy_reason"] = None
                except Exception:
                    pass
                
                fps_start_time = time.time()
                fps_frame_count = 0
                processing_times = []
                self._perf_window = self._new_perf_window()
        
        self._stop_jpeg_encoder()
        self._stop_raw_jpeg_encoder()
        if self.hls_writer is not None:
            self.hls_writer.stop()
        self._close_local_preview()
        logger.info("Display service stopped")
    

    def _alert_identity(self, alert):
        for key in ('tracker_id', 'track_id', 'alert_id', 'cluster_id'):
            if key in alert:
                return f"{key}={alert.get(key)}"
        return "id=?"

    def _alert_duration_text(self, alert):
        fields = []
        for key in ('duration', 'observation_span', 'elapsed', 'fight_rate', 'mask_wearing_rate', 'fall_percentage'):
            if key not in alert:
                continue
            value = alert.get(key)
            if isinstance(value, (int, float)):
                fields.append(f"{key}={float(value):.2f}")
            else:
                fields.append(f"{key}={value}")
        return ",".join(fields) if fields else "-"

    def _debug_log_detection_result(self, detector_type, stream_id, result):
        display_alerts = result.get('display_alerts') or []
        detections = result.get('detections') or []
        recordable_alerts = result.get('recordable_alerts') or []
        recording_alerts = [item for item in display_alerts if item.get('is_recording', False)]
        interval = max(0.0, self._alert_debug_log_interval_s)
        now = time.monotonic()
        key = (detector_type, int(stream_id))
        last = self._alert_debug_last_log.get(key, 0.0)
        if not recording_alerts and interval > 0.0 and now - last < interval:
            return
        self._alert_debug_last_log[key] = now

        metrics = result.get('metrics') or result
        cooldown = metrics.get('cooldown_remaining_s')
        cooldown_text = f"{float(cooldown):.1f}s" if isinstance(cooldown, (int, float)) else "-"
        alert_parts = []
        for alert in display_alerts[:5]:
            alert_parts.append(
                "%s rec=%s dur={%s}" % (
                    self._alert_identity(alert),
                    bool(alert.get('is_recording', False)),
                    self._alert_duration_text(alert),
                )
            )
        if len(display_alerts) > 5:
            alert_parts.append(f"+{len(display_alerts) - 5} more")
        logger.info(
            "[ALERT DEBUG] detector=%s ch=%d frame=%s display_alerts=%d recording=%d "
            "detections=%d recordable=%d cooldown=%s alarm_number=%s total_alerts=%s alerts=[%s]",
            detector_type,
            int(stream_id) + 1,
            result.get('frame_number'),
            len(display_alerts),
            len(recording_alerts),
            len(detections),
            len(recordable_alerts),
            cooldown_text,
            metrics.get('alarm_number', '-'),
            metrics.get('total_alerts', '-'),
            "; ".join(alert_parts) if alert_parts else "-",
        )

    def _collect_detection_results(self):
        """收集所有检测服务的最新结果（支持多路 stream_id）"""
        for detector_type, result_queue in self.result_queues.items():
            try:
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    if not result:
                        continue
                    try:
                        result['_display_received_mono'] = time.monotonic()
                    except Exception:
                        pass
                    # 记录最近一次“有检测结果输出”的时间，用于纯预览性能优化
                    try:
                        if result.get('enabled', True):
                            self._last_detector_result_ts = time.time()
                    except Exception:
                        pass
                    stream_id = int(result.get('stream_id', 0))
                    if stream_id < 0 or stream_id >= self.stream_count:
                        logger.warning(
                            "Ignoring %s result for unconfigured stream_id=%s (stream_count=%s)",
                            detector_type,
                            stream_id,
                            self.stream_count,
                        )
                        continue
                    if not result.get('enabled', True):
                        logger.info(f"🔕 [{detector_type.upper()}] Detector disabled, clearing results")
                        if self._multi_stream:
                            self.latest_results[detector_type] = {}
                            self.sticky_results[detector_type] = {}
                            self.active_alerts[detector_type] = {}
                            self.triggered_alarms[detector_type] = {}
                        else:
                            self.latest_results[detector_type] = None
                            self.sticky_results[detector_type] = None
                            self.active_alerts[detector_type] = {}
                            self.triggered_alarms[detector_type] = set()
                        self._active_result_streams[detector_type].clear()
                        self._result_by_frame.pop(detector_type, None)
                    else:
                        if self._multi_stream:
                            if not isinstance(self.latest_results[detector_type], dict):
                                self.latest_results[detector_type] = {}
                            self.latest_results[detector_type][stream_id] = result
                        else:
                            self.latest_results[detector_type] = result
                        # 方案 C：按帧号缓存，供渲染时帧-结果对齐
                        fn = result.get('frame_number')
                        if fn is not None:
                            if detector_type not in self._result_by_frame:
                                self._result_by_frame[detector_type] = {}
                            if stream_id not in self._result_by_frame[detector_type]:
                                self._result_by_frame[detector_type][stream_id] = deque(maxlen=self._result_by_frame_maxlen)
                            buf = self._result_by_frame[detector_type][stream_id]
                            if buf and buf[-1][0] == fn:
                                buf[-1] = (fn, result)
                            else:
                                buf.append((fn, result))
                        self._active_result_streams[detector_type].add(stream_id)
                        # 空结果也要覆盖旧结果，否则旧框会被“粘住”继续画出来。
                        if self._multi_stream:
                            self.sticky_results[detector_type][stream_id] = result
                        else:
                            self.sticky_results[detector_type] = result
                        self._bump_result_version(detector_type, stream_id)
                        self._debug_log_detection_result(detector_type, stream_id, result)
                
                # 检测结果超时：多路时按 stream_id 检查，一路时按单结果检查
                if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
                    for sid in list(self.latest_results[detector_type].keys()):
                        r = self.latest_results[detector_type][sid]
                        if r and r.get('timestamp'):
                            time_diff = datetime.now(BEIJING_TZ) - r['timestamp']
                            if time_diff > timedelta(seconds=3):
                                self._clear_sticky_for_detector_stream(detector_type, sid)
                elif not self._multi_stream and self.latest_results.get(detector_type) is not None:
                    r = self.latest_results[detector_type]
                    if r.get('timestamp'):
                        time_diff = datetime.now(BEIJING_TZ) - r['timestamp']
                        if time_diff > timedelta(seconds=3):
                            logger.warning(f"⚠️ [{detector_type.upper()}] No update for {time_diff.total_seconds():.1f}s, clearing results")
                            self._clear_sticky_for_detector_stream(detector_type, 0)
            except Exception:
                pass

    def _alarm_buffer_due(self):
        """Limit expensive alarm frame copies; preview FPS is controlled separately."""
        now = time.monotonic()
        if now - self._last_alarm_buffer_mono < self._alarm_buffer_interval_s:
            return False
        self._last_alarm_buffer_mono = now
        return True

    def _should_buffer_for_alerts(self):
        """
        是否需要执行告警缓冲（video_buffer.add_frame/update 等重操作）：
        - 若存在活跃录制：始终需要
        - 若最近 1 秒内有检测结果输出：需要（保证告警触发时有完整预录）
        - 否则：跳过缓冲以提升纯预览 FPS（此时也不会触发告警）
        """
        try:
            if self.video_buffer.get_active_recordings() > 0:
                return True
        except Exception:
            pass
        ts = self._last_detector_result_ts
        if ts is None:
            return False
        return (time.time() - ts) <= 1.0

    def _result_has_visuals(self, detector_type, result):
        del detector_type
        if not result:
            return False
        for key in ("display_alerts", "detections", "current_detections"):
            value = result.get(key)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return True
        return False
    
    def _get_result_for_stream(self, detector_type, stream_id):
        """按 stream_id 取检测结果（一路时 stream_id 忽略）"""
        if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
            return self.latest_results[detector_type].get(stream_id)
        return self.latest_results.get(detector_type)
    
    # 仅显示“足够新”的检测结果：允许跳帧显示（检测结果不必每帧都有）。
    # 统一阈值以保证各检测器显示策略一致。
    _MAX_FRAME_DIFF_FOR_DISPLAY_DEFAULT = 12.0
    _MAX_FRAME_DIFF_FOR_DISPLAY_BY_DETECTOR = {}

    def _result_is_fresh_for_frame(self, result, frame_number):
        if not result:
            return False
        result_frame = result.get('frame_number')
        if frame_number is not None and result_frame is not None:
            try:
                if int(frame_number) - int(result_frame) > self._display_max_frame_lag:
                    return False
            except Exception:
                pass
        result_ts = result.get('timestamp')
        if result_ts is not None:
            try:
                if (datetime.now(BEIJING_TZ) - result_ts).total_seconds() > self._display_result_ttl_s:
                    return False
            except Exception:
                pass
        return True

    def _get_result_for_frame(self, detector_type, stream_id, frame_number):
        """只取足够新的检测结果用于更新报警框缓存。"""
        if self._multi_stream and isinstance(self.sticky_results.get(detector_type), dict):
            result = self.sticky_results[detector_type].get(stream_id)
        else:
            result = self.sticky_results.get(detector_type)
        if not self._result_is_fresh_for_frame(result, frame_number):
            return None
        return result
    
    def _clear_bbox_smooth(self, detector_type, stream_id, track_id):
        """track 消失时清理平滑缓存，避免下次复用时沿用陈旧 bbox。"""
        key = (detector_type, stream_id, track_id)
        self._bbox_smooth_cache.pop(key, None)

    def _smooth_bbox(self, detector_type, stream_id, track_id, bbox, bbox_format='xyxy'):
        """方案 A：对报警框 bbox 做指数滑动平均，减少抖动、使框更稳定跟随目标。"""
        if bbox is None or len(bbox) < 4:
            return bbox
        try:
            if bbox_format == 'xywh':
                x, y, w, h = [float(b) for b in bbox[:4]]
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                x1, y1, x2, y2 = [float(b) for b in bbox[:4]]
            key = (detector_type, stream_id, track_id)
            prev = self._bbox_smooth_cache.get(key)
            if prev is None:
                out = [x1, y1, x2, y2]
                self._bbox_smooth_cache[key] = {'bbox': out}
                if bbox_format == 'xywh':
                    return [out[0], out[1], out[2] - out[0], out[3] - out[1]]
                return out
            a = self._bbox_smooth_alpha
            p = prev['bbox']
            out = [
                a * p[0] + (1 - a) * x1,
                a * p[1] + (1 - a) * y1,
                a * p[2] + (1 - a) * x2,
                a * p[3] + (1 - a) * y2,
            ]
            self._bbox_smooth_cache[key] = {'bbox': out}
            if bbox_format == 'xywh':
                return [out[0], out[1], out[2] - out[0], out[3] - out[1]]
            return out
        except Exception:
            return bbox
    
    def _build_composite_frame(self):
        """合成预览画面：单路/多路均使用统一降采样；输出固定 1280x720。单路时单格放大至满屏，多路时 2x2 四宫格。"""
        blank_cell = self._blank_cell

        if self.stream_count == 1:
            fd = self.latest_frames.get(0)
            if not fd or 'frame' not in fd or fd.get('frame') is None:
                out = cv2.resize(blank_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                return out
            frame_cell = fd['frame']
            orig_wh = (fd.get('original_width'), fd.get('original_height'))
            if None in orig_wh:
                orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
            frame_cell = self._render_results(
                frame_cell, fd.get('frame_number'), fd.get('timestamp'), stream_id=0, original_size=orig_wh
            )
            frame_cell = self._overlay_stream_status(frame_cell, 0)
            return frame_cell

        # 多路：固定四宫格，每格 CELL_W x CELL_H。
        cells = []
        for idx in range(GRID_CELL_COUNT):
            if idx < self.stream_count:
                frame_cell = self._render_cell_for_stream(idx, self.latest_frames.get(idx))
            else:
                frame_cell = blank_cell.copy()
            cells.append(frame_cell)

        composite = self._composite_canvas
        composite[:] = blank_cell[0, 0]
        for idx, frame_cell in enumerate(cells):
            row = idx // GRID_COLUMNS
            col = idx % GRID_COLUMNS
            y0 = row * CELL_H
            x0 = col * CELL_W
            composite[y0:y0 + CELL_H, x0:x0 + CELL_W] = frame_cell
        if composite.shape[1] != COMPOSITE_OUTPUT_W or composite.shape[0] != COMPOSITE_OUTPUT_H:
            return cv2.resize(composite, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        return composite.copy()
    
    def _render_results(self, frame, frame_number, timestamp, stream_id=0, original_size=None):
        """渲染所有检测结果到画面。original_size=(w,h) 为原始分辨率，用于将检测框坐标缩放到当前显示尺寸。"""
        self._begin_alert_render_for_stream(stream_id)
        self.expire_stale_alert_boxes(stream_id)
        lookup_start = time.perf_counter()
        detector_results = {
            detector_type: self._get_result_for_frame(detector_type, stream_id, frame_number)
            for detector_type in self._RENDER_DETECTORS
        }
        self._perf_add('result_lookup_ms', (time.perf_counter() - lookup_start) * 1000.0)
        has_new_result = any(result is not None for result in detector_results.values())
        if not has_new_result and not self._has_active_alert_boxes(stream_id):
            return frame

        rendered = frame.copy()
        disp_wh = (frame.shape[1], frame.shape[0])
        orig_wh = original_size if original_size and None not in original_size else disp_wh
        coord_wh = {
            detector_type: self._result_coord_wh(result, orig_wh)
            for detector_type, result in detector_results.items()
        }
        for detector_type, result in detector_results.items():
            if result is not None:
                rendered = self._draw_raw_model_detections(
                    rendered, result, coord_wh[detector_type], disp_wh
                )
        if detector_results['fall'] is not None:
            rendered = self._render_fall_detections(
                rendered, detector_results['fall'], stream_id, coord_wh['fall'], disp_wh
            )
        if detector_results['ventilator'] is not None:
            rendered = self._render_ventilator_detections(
                rendered, detector_results['ventilator'], stream_id, coord_wh['ventilator'], disp_wh
            )
        if detector_results['fight'] is not None:
            rendered = self._render_fight_detections(
                rendered, detector_results['fight'], stream_id, coord_wh['fight'], disp_wh
            )
        if detector_results['crowd'] is not None:
            rendered = self._render_crowd_detections(
                rendered, detector_results['crowd'], stream_id, coord_wh['crowd'], disp_wh
            )
        if detector_results['helmet'] is not None:
            rendered = self._render_helmet_detections(
                rendered, detector_results['helmet'], stream_id, coord_wh['helmet'], disp_wh
            )
        if detector_results['window_door_inside'] is not None:
            rendered = self._render_window_door_detections(
                rendered,
                detector_results['window_door_inside'],
                stream_id,
                detector_key='window_door_inside',
                orig_wh=coord_wh['window_door_inside'],
                disp_wh=disp_wh,
            )
        if detector_results['window_door_outside'] is not None:
            rendered = self._render_window_door_detections(
                rendered,
                detector_results['window_door_outside'],
                stream_id,
                detector_key='window_door_outside',
                orig_wh=coord_wh['window_door_outside'],
                disp_wh=disp_wh,
            )
        return self.draw_active_alert_boxes(rendered, stream_id, frame_number=frame_number)
    
    def _render_fall_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """渲染跌倒检测结果 - 模型检出即画框，与 fight 同构"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        detections = result.get('display_alerts') or result.get('detections', [])

        for idx, det in enumerate(detections):
            alert_id = det.get('alert_id', f"current_{idx}")
            self.update_active_alert_box(
                'fall',
                stream_id,
                alert_id,
                det.get('bbox'),
                label=("跌倒警报", "fall"),
                bbox_format='xyxy',
                result=result,
                coord_wh=orig_wh,
                info=det,
                is_recording=det.get('is_recording', False),
                disable_flow_tracking=True,
            )

        return frame
    
    def _render_ventilator_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """
        🎯 渲染呼吸机检测结果 - 增强版（包含调试可视化）
        """
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts['ventilator'].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts['ventilator']
        all_persons = result.get('persons', [])
        detections = result.get('detections', [])  # 用于触发录制
        display_alerts = result.get('display_alerts', [])  # 🎯 用于画面显示（所有未佩戴的人）
        
        # ========== 🆕 调试可视化开关 ==========
        # 设置为 True 可以看到面罩、氧气瓶、头部框（用于调试匹配逻辑）
        # 生产环境建议设置为 False（只显示报警红框，画面更简洁）
        DEBUG_MODE = False  # 🔧 关闭调试模式（不显示面罩/氧气瓶/头部框）
        
        # ========== 🆕 调试可视化：绘制检测框 ==========
        if DEBUG_MODE:
            masks = result.get('masks', [])
            tanks = result.get('tanks', [])
            
            # 🟢 绘制面罩（绿色框）
            for idx, mask in enumerate(masks):
                bbox = self._scale_bbox_to_display(mask['bbox'], orig_wh, disp_wh, 'xyxy')
                x1, y1, x2, y2 = map(int, bbox)
                confidence = mask['confidence']
                
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签背景
                label = f"Mask {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            (0, 255, 0), -1)
                
                # 绘制文字
                cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # 绘制中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # 🔵 绘制氧气瓶（蓝色框）
            for idx, tank in enumerate(tanks):
                bbox = self._scale_bbox_to_display(tank['bbox'], orig_wh, disp_wh, 'xyxy')
                x1, y1, x2, y2 = map(int, bbox)
                confidence = tank['confidence']
                
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 绘制标签背景
                label = f"Tank {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            (255, 0, 0), -1)
                
                # 绘制文字
                cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 绘制中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # 🟡 绘制所有头部框（黄色框）
            for person in all_persons:
                tracker_id = person['tracker_id']
                bbox = self._scale_bbox_to_display(person['bbox'], orig_wh, disp_wh, 'xyxy')
                x1, y1, x2, y2 = map(int, bbox)
                
                # 判断是否已触发报警（避免重复绘制）
                is_alerted = tracker_id in alerts_dict
                
                if not is_alerted:  # 只绘制未报警的头部框
                    # 绘制边框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # 获取佩戴状态
                    wearing_rate = person.get('mask_wearing_rate', 0)
                    obs_count = person.get('observation_count', 0)
                    check_completed = person.get('check_completed', False)
                    
                    # 绘制标签
                    if check_completed:
                        status = "✅ Pass" if person.get('check_passed', False) else "❌ Fail"
                        label = f"Head {tracker_id} {status}"
                    else:
                        label = f"Head {tracker_id} {wearing_rate:.1%} ({obs_count}f)"
                    
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), 
                                (0, 255, 255), -1)
                    
                    cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # 绘制中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
            
            # 🆕 在画面左上角显示统计信息 (已禁用 - 不需要显示)
            # stats_text = [
            #     f"Masks: {len(masks)}",
            #     f"Tanks: {len(tanks)}",
            #     f"Heads: {len(all_persons)}",
            #     f"Alerts: {len(display_alerts)}"
            # ]
            # 
            # y_offset = 30
            # for text in stats_text:
            #     # 文字背景
            #     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            #     cv2.rectangle(frame, (10, y_offset - text_size[1] - 5), 
            #                 (10 + text_size[0] + 10, y_offset + 5), 
            #                 (0, 0, 0), -1)
            #     
            #     # 文字
            #     cv2.putText(frame, text, (15, y_offset), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            #     y_offset += 30
        
        # ========== 原有逻辑：处理报警 ==========
        if logger.isEnabledFor(logging.DEBUG) and display_alerts:
            logger.debug("[VENTILATOR DEBUG] display_alerts=%d stream=%s", len(display_alerts), stream_id)
        if logger.isEnabledFor(logging.DEBUG) and detections:
            logger.debug("[VENTILATOR DEBUG] detections=%d stream=%s", len(detections), stream_id)
        
        # 🎯 使用 display_alerts 更新报警框缓存，显示由 draw_active_alert_boxes 每帧统一完成。
        for alert in display_alerts:
            tracker_id = alert['tracker_id']
            english_text = "no ventilator"
            if alert.get('is_recording', False):
                english_text = f"🎬 {english_text}"
            self.update_active_alert_box(
                'ventilator',
                stream_id,
                tracker_id,
                alert.get('bbox'),
                label=("未佩戴呼吸机", english_text),
                bbox_format='xyxy',
                result=result,
                coord_wh=orig_wh,
                info=alert,
                is_recording=alert.get('is_recording', False),
            )

        for person in all_persons:
            tracker_id = person.get('tracker_id')
            if not (person.get('alarm_triggered') and person.get('check_completed') and not person.get('check_passed')):
                continue
            self._refresh_active_alert_box(
                'ventilator',
                stream_id,
                tracker_id,
                person.get('bbox'),
                result=result,
                coord_wh=orig_wh,
                bbox_format='xyxy',
            )

        return frame

    def _iter_active_results(self, detector_type):
        if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
            active_streams = self._active_result_streams.get(detector_type, set())
            for sid in tuple(active_streams):
                res = self.latest_results[detector_type].get(sid)
                if res:
                    yield sid, res
        elif self.latest_results.get(detector_type):
            yield 0, self.latest_results[detector_type]
    
    def _render_fight_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """渲染打架检测结果 - 修复版：支持检测框实时跟随 + 自动清理过期警报"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        detections = result.get('display_alerts') or result.get('detections', [])
        
        for idx, det in enumerate(detections):
            alert_id = det.get('alert_id', f"current_{idx}")
            self.update_active_alert_box(
                'fight',
                stream_id,
                alert_id,
                det.get('bbox'),
                label=("打架警报", "fight"),
                bbox_format='xyxy',
                result=result,
                coord_wh=orig_wh,
                info=det,
                is_ongoing=det.get('is_ongoing', False),
            )
        
        return frame
    
    def _render_crowd_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """🎯 渲染人员聚集检测结果 - 使用display_alerts持续显示"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        display_alerts = result.get('display_alerts', [])
        
        for alert in display_alerts:
            cluster_id = alert['cluster_id']
            count = alert['info']['count']
            self.update_active_alert_box(
                'crowd',
                stream_id,
                cluster_id,
                alert.get('bbox'),
                label=("人群聚集", f"crowd {count} people"),
                bbox_format='xyxy',
                result=result,
                coord_wh=orig_wh,
                info=alert,
                extend_y2_by_height_ratio=1.5,
            )
        
        return frame
    
    def _render_helmet_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """🎯 渲染安全帽检测结果 - 使用display_alerts显示所有人"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        display_alerts = result.get('display_alerts', [])
        
        for alert in display_alerts:
            track_id = alert['track_id']
            self.update_active_alert_box(
                'helmet',
                stream_id,
                track_id,
                alert.get('bbox'),
                label=("未佩戴安全帽", "no helmet"),
                bbox_format='xywh',
                result=result,
                coord_wh=orig_wh,
                info=alert,
                is_recording=alert.get('is_recording', False),
            )

        tracks = result.get('tracks') or {}
        if isinstance(tracks, dict):
            track_items = tracks.items()
        else:
            track_items = []
        for track_id, track in track_items:
            try:
                class_id = int(track.get('class', track.get('class_id', -1)))
            except Exception:
                class_id = -1
            if class_id != 1:
                continue
            self._refresh_active_alert_box(
                'helmet',
                stream_id,
                track_id,
                track.get('bbox'),
                result=result,
                coord_wh=orig_wh,
                bbox_format='xywh',
            )
        
        return frame
    
    def _render_window_door_detections(self, frame, result, stream_id=0, detector_key='window_door_inside', orig_wh=None, disp_wh=None):
        """🎯 渲染窗户门检测结果 - 使用display_alerts显示所有打开的窗户/门"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        display_alerts = result.get('display_alerts', [])
        
        for alert in display_alerts:
            track_id = alert['track_id']
            display_name = alert['info'].get('display_name', '警报')
            self.update_active_alert_box(
                detector_key,
                stream_id,
                track_id,
                alert.get('bbox'),
                label=(display_name, "alert!"),
                bbox_format='xyxy',
                result=result,
                coord_wh=orig_wh,
                info=alert,
                is_recording=alert.get('is_recording', False),
            )
        
        return frame
    
    def _process_detections(self, timestamp):
        """🎯 改进的警报处理逻辑（支持多路 stream_id）"""
        def _triggered_set(detector_type, stream_id):
            if self._multi_stream:
                return self.triggered_alarms[detector_type].setdefault(stream_id, set())
            return self.triggered_alarms[detector_type]
        
        # 跌倒检测：使用 display_alerts 中标记了 is_recording 的告警，确保遵循检测器内部冷却期
        for stream_id, fall_res in self._iter_active_results('fall'):
            alerts = fall_res.get('display_alerts') or fall_res.get('detections', [])
            current_tracker_ids = set()
            triggered = _triggered_set('fall', stream_id)
            for alert in alerts:
                # 仅对 is_recording=True 的告警触发录制，避免重复触发
                if not alert.get('is_recording', True):
                    continue
                tracker_id = alert.get('tracker_id', 0)
                current_tracker_ids.add(tracker_id)
                if tracker_id not in triggered:
                    alarm_id = f"fall_ch{stream_id + 1}_{tracker_id}_{int(timestamp.timestamp())}"
                    triggered.add(tracker_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='fall', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_tracker_ids)
        
        # 呼吸机检测：同样以 display_alerts + is_recording 为准，多通道互不干扰
        for stream_id, vent_res in self._iter_active_results('ventilator'):
            alerts = vent_res.get('display_alerts') or vent_res.get('detections', [])
            current_tracker_ids = set()
            triggered = _triggered_set('ventilator', stream_id)
            for alert in alerts:
                if not alert.get('is_recording', True):
                    continue
                tracker_id = alert.get('tracker_id', 0)
                current_tracker_ids.add(tracker_id)
                if tracker_id not in triggered:
                    alarm_id = f"ventilator_ch{stream_id + 1}_{tracker_id}_{int(timestamp.timestamp())}"
                    triggered.add(tracker_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='ventilator', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_tracker_ids)
        
        for stream_id, fight_res in self._iter_active_results('fight'):
            detections = fight_res.get('detections', [])
            current_alert_ids = set()
            triggered = _triggered_set('fight', stream_id)
            for det in detections:
                alert_id = det.get('alert_id', 0)
                current_alert_ids.add(alert_id)
                if alert_id not in triggered:
                    alarm_id = f"fight_ch{stream_id + 1}_{alert_id}_{int(timestamp.timestamp())}"
                    triggered.add(alert_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='fight', alarm_info=det, stream_id=stream_id)
            triggered -= (triggered - current_alert_ids)
        
        for stream_id, crowd_res in self._iter_active_results('crowd'):
            detections = crowd_res.get('detections', [])
            current_alert_ids = set()
            triggered = _triggered_set('crowd', stream_id)
            for det in detections:
                alert_id = det.get('alert_id', 0)
                current_alert_ids.add(alert_id)
                if alert_id not in triggered:
                    alarm_id = f"crowd_ch{stream_id + 1}_{alert_id}_{int(timestamp.timestamp())}"
                    triggered.add(alert_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='crowd', alarm_info=det, stream_id=stream_id)
            triggered -= (triggered - current_alert_ids)
        
        # 安全帽检测：使用 display_alerts 中 is_recording=True 的目标
        for stream_id, helmet_res in self._iter_active_results('helmet'):
            alerts = helmet_res.get('display_alerts') or helmet_res.get('detections', [])
            current_track_ids = set()
            triggered = _triggered_set('helmet', stream_id)
            for alert in alerts:
                if not alert.get('is_recording', True):
                    continue
                track_id = alert.get('track_id', alert.get('tracker_id', 0))
                current_track_ids.add(track_id)
                if track_id not in triggered:
                    alarm_id = f"helmet_ch{stream_id + 1}_{track_id}_{int(timestamp.timestamp())}"
                    triggered.add(track_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='helmet', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_track_ids)
        
        # 窗户门检测：仓内/仓外分别触发录制（互不干扰）
        for detector_key in ("window_door_inside", "window_door_outside"):
            for stream_id, wd_res in self._iter_active_results(detector_key):
                alerts = wd_res.get('display_alerts') or wd_res.get('detections', [])
                current_track_ids = set()
                triggered = _triggered_set(detector_key, stream_id)
                for alert in alerts:
                    if not alert.get('is_recording', True):
                        continue
                    track_id = alert.get('track_id', 0)
                    current_track_ids.add(track_id)
                    if track_id not in triggered:
                        alarm_id = f"{detector_key}_ch{stream_id + 1}_{track_id}_{int(timestamp.timestamp())}"
                        triggered.add(track_id)
                        self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type=detector_key, alarm_info=alert, stream_id=stream_id)
                triggered -= (triggered - current_track_ids)
    
    def _on_video_saved(self, video_path, image_path, alarm_type, alarm_info):
        """视频和图片保存回调 - 告警事件推送到 Web 层（SSE），视频/图片通过 Web 端下载"""
        # 告警事件推送到 Web 层（SSE）
        if self.alert_queue is not None:
            try:
                from core.events import alert_event
                ev = alert_event(
                    alarm_id=alarm_type + "_" + datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S"),
                    alarm_type=alarm_type,
                    alarm_info={k: v for k, v in (alarm_info or {}).items()
                               if isinstance(v, (str, int, float, bool, type(None)))},
                    video_path=video_path,
                    image_path=image_path,
                )
                self.alert_queue.put_nowait(ev)
            except Exception as e:
                logger.warning("Failed to push alert event to queue: %s", e)
    
    def _check_control_commands(self):
        """检查控制命令"""
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                if cmd == 'stop':
                    self.stop()
                elif isinstance(cmd, dict) and cmd.get('cmd') == 'set_preview':
                    self.preview_mode = 'merged'
                elif isinstance(cmd, dict) and cmd.get('cmd') == 'clear_sticky':
                    detector_type = str(cmd.get('detector') or '')
                    sid = max(0, min(self.stream_count - 1, int(cmd.get('stream_id', 0))))
                    self._clear_sticky_for_detector_stream(detector_type, sid)
        except Exception as e:
            pass
    
    def stop(self):
        """停止展示服务"""
        logger.info("Stopping display service...")
        self.running = False
        self._stop_jpeg_encoder()
        self._stop_raw_jpeg_encoder()
        self._close_local_preview()
        
        # 等待所有视频保存完成
        if hasattr(self, 'video_buffer'):
            self.video_buffer.shutdown(timeout=30)


def run_display_service(frame_queue, result_queues, output_queue, control_queue,
                        video_output_dir, fps, stream_count=1, alert_queue=None, font_path_config=None,
                        preview_status=None, preview_cfg=None, alert_box_mode='follow',
                        alarm_retention_days=7, raw_output_queue=None):
    """进程入口函数。stream_count: 1~4 路；alert_queue: 告警事件队列；font_path_config: 中文字体路径。"""
    ensure_root_logging()
    service = DisplayService(
        frame_queue,
        result_queues,
        output_queue,
        control_queue,
        video_output_dir,
        fps,
        stream_count,
        alert_queue,
        font_path_config,
        preview_status,
        preview_cfg,
        alarm_retention_days,
        raw_output_queue,
    )
    # 根据配置设置报警框显示模式
    try:
        service.alert_box_mode = str(alert_box_mode or 'follow').strip().lower()
        if service.alert_box_mode not in ('follow', 'blink'):
            service.alert_box_mode = 'follow'
    except Exception:
        service.alert_box_mode = 'follow'
    service.start()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Display Service')
    parser.add_argument('--output-dir', type=str, default='./alarm_videos', help='Video output directory')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    args = parser.parse_args()
    
    frame_queue = mp.Queue(maxsize=10)
    result_queues = {
        'fall': mp.Queue(maxsize=20),
        'ventilator': mp.Queue(maxsize=20),
        'fight': mp.Queue(maxsize=20),
        'crowd': mp.Queue(maxsize=20),
        'helmet': mp.Queue(maxsize=20),
        'window_door_inside': mp.Queue(maxsize=20),
        'window_door_outside': mp.Queue(maxsize=20),
    }
    output_queue = mp.Queue(maxsize=5)
    control_queue = mp.Queue()
    
    run_display_service(frame_queue, result_queues, output_queue, control_queue, 
                       args.output_dir, args.fps)
