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
from collections import deque
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Threading 异步保存支持
import threading
from queue import Queue, Full, Empty

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
                 output_dir='./alarm_videos', on_video_saved=None):
        self.fps = fps
        self.pre_alarm_seconds = pre_alarm_seconds
        self.post_alarm_seconds = post_alarm_seconds
        
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
        
        logger.info(f"VideoBufferManager initialized (Threading Async):")
        logger.info(f"   ├─ {pre_alarm_seconds}s pre + {post_alarm_seconds}s post")
        logger.info(f"   ├─ Max pre-buffer: {max_buffer_frames} frames")
        logger.info(f"   ├─ Output: {self.output_dir}")
        logger.info(f"   └─ ✅ Async save thread started")
        if SOPHON_AVAILABLE:
            logger.info("Sophon SAIL detected: alarm videos can use hardware encoder like yolov8_bmcv")
    
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
        if trigger_frame is not None:
            if stream_id is not None:
                image_path = self.output_dir / f"{alarm_type}_ch{stream_id + 1}_{timestamp}_frame.jpg"
            else:
                image_path = self.output_dir / f"{alarm_type}_{timestamp}_frame.jpg"
            try:
                cv2.imwrite(str(image_path), trigger_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image_size = image_path.stat().st_size
                logger.info(f"📸 [IMAGE SAVE] Frame saved: {image_path.name} ({image_size/1024:.1f}KB)")
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
# 方案 A：统一降采样 + 固定 960x540 九宫格合成（进一步降低像素量与编码压力）
COMPOSITE_OUTPUT_W = 960
COMPOSITE_OUTPUT_H = 540
CELL_W = COMPOSITE_OUTPUT_W // 3   # 320
CELL_H = COMPOSITE_OUTPUT_H // 3   # 180


class DisplayService:
    """展示服务 - 独立进程"""
    
    def __init__(self, frame_queue, result_queues, output_queue, control_queue,
                 video_output_dir='./alarm_videos', fps=30, stream_count=1, alert_queue=None,
                font_path_config=None, encoder_queue=None):
        self.frame_queue = frame_queue
        self.result_queues = result_queues
        self.output_queue = output_queue
        self.control_queue = control_queue
        self.alert_queue = alert_queue  # 告警事件队列，供 Web 层实时推送
        self.encoder_queue = encoder_queue  # 已标注帧队列，供编码服务生成 HLS 流（解码-处理-编码流水线）
        self.fps = fps
        # 支持 1~9 路视频流；预览采用九宫格布局（多路时）
        self.stream_count = max(1, min(9, int(stream_count)))
        self._multi_stream = (self.stream_count > 1)
        self.latest_frames = {}  # stream_id -> frame_data（多路时使用）
        
        self.video_buffer = VideoBufferManager(
            fps=fps, 
            output_dir=video_output_dir,
            on_video_saved=self._on_video_saved
        )
        
        # 中文字体（用于报警标签，避免中文乱码）
        self.font_path = None
        self._load_chinese_font(font_path_config)
        
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
        
        # 警报状态管理。多路时 active_alerts[det][stream_id][tracker_id]=...
        self.active_alerts = {
            'fall': {} if self._multi_stream else {},
            'ventilator': {} if self._multi_stream else {},
            'fight': {} if self._multi_stream else {},
            'crowd': {} if self._multi_stream else {},
            'helmet': {} if self._multi_stream else {},
            'window_door_inside': {} if self._multi_stream else {},
            'window_door_outside': {} if self._multi_stream else {},
        }
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
        self._result_by_frame = {}  # detector_type -> stream_id -> {frame_number: result}
        self._result_by_frame_maxlen = 60
        
        # 方案 A：报警框 bbox 平滑缓存 (detector_type, stream_id, track_id) -> {'bbox': [x1,y1,x2,y2]}
        self._bbox_smooth_cache = {}
        self._bbox_smooth_alpha = 0.1  # 上一帧权重，0.4=更快跟随目标，减少行走时滞后

        # 预览模式：merged=九宫格合流，task=单路任务通道
        self.preview_mode = 'merged'
        self.preview_task_stream_id = 0  # task 模式下显示的 stream_id（0-based）
        
        # 中文字体由 _load_chinese_font(font_path_config) 在 __init__ 中设置

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
    
    def _draw_bilingual_label(self, frame, bbox, chinese_text, english_text, color=(0, 0, 255)):
        """绘制双语标签（中文在上，英文在下）。无中文字体时仅绘制英文。字号随画面尺寸自适应。"""
        x1, y1, x2, y2 = map(int, bbox)
        use_chinese = bool(self.font_path and chinese_text)
        # 字号自适应：320x180 小格也清晰，960x540 不至于过大
        fh, fw = frame.shape[:2]
        base = int(max(9, min(18, round(min(fw, fh) * 0.04))))
        chinese_font_size = base
        english_font_size = max(8, int(round(base * 0.95)))
        
        font_cn = None
        if self.font_path and chinese_text:
            try:
                font_cn = ImageFont.truetype(self.font_path, chinese_font_size)
            except Exception:
                use_chinese = False
        
        font_en = None
        english_font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/arialbd.ttf'
        ]
        for fp in english_font_paths:
            if os.path.exists(fp):
                try:
                    font_en = ImageFont.truetype(fp, english_font_size)
                    break
                except Exception:
                    continue
        if font_en is None:
            font_en = ImageFont.load_default()
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        if use_chinese and font_cn:
            bbox_cn = draw.textbbox((0, 0), chinese_text, font=font_cn)
            text_height_cn = bbox_cn[3] - bbox_cn[1]
        else:
            text_height_cn = 0
        
        bbox_en = draw.textbbox((0, 0), english_text, font=font_en)
        text_height_en = bbox_en[3] - bbox_en[1]
        total_height = (text_height_cn + 2 + text_height_en) if use_chinese else text_height_en
        text_y = y1 - total_height - 4
        text_color = (255, 255, 255)
        
        if use_chinese and font_cn:
            draw.text((x1, text_y), chinese_text, font=font_cn, fill=text_color)
            text_y += text_height_cn + 2
        draw.text((x1, text_y), english_text, font=font_en, fill=text_color)
        
        frame_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return frame_result

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

        # 细边框 + 角标
        thickness = 2 if min(fw, fh) <= 400 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        corner = max(8, int(min(x2 - x1, y2 - y1) * 0.18))
        # 左上
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)
        # 右上
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)
        # 左下
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)
        # 右下
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)

        frame = self._draw_bilingual_label(frame, [x1, y1, x2, y2], chinese_text, english_text, color=color)
        return frame
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
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
        """主循环：汇总结果、渲染、处理警报（支持一路/两路/四路/九路/十六路）"""
        logger.info("Display service running... (stream_count=%d)" % self.stream_count)
        
        fps_start_time = time.time()
        fps_frame_count = 0
        processing_times = []
        # 预览与编码输出目标 FPS（节流避免过载）
        target_output_fps = min(25.0, float(self.fps))
        min_output_interval = 1.0 / target_output_fps if target_output_fps > 0 else 0.0
        last_output_ts = 0.0
        
        while self.running:
            self._check_control_commands()
            
            frame_start_time = time.time()
            
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
            except Exception:
                continue
            
            stream_id = frame_data.get('stream_id', 0)
            # 方案 A：入口统一降采样，latest_frames 只存小分辨率帧，减少内存与合成/编码压力
            frame = frame_data.get('frame')
            if frame is not None and frame.size > 0:
                orig_h, orig_w = frame.shape[:2]
                if orig_w != CELL_W or orig_h != CELL_H:
                    resized = cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR)
                    frame_data = {**frame_data, 'frame': resized, 'original_width': orig_w, 'original_height': orig_h}
                else:
                    frame_data = {**frame_data, 'original_width': orig_w, 'original_height': orig_h}
            self.latest_frames[stream_id] = frame_data
            # 更新每路 FPS/连接状态统计
            try:
                self._update_stream_stats(stream_id, frame_data.get('frame_number'), frame_data.get('timestamp'))
            except Exception:
                pass
            # 排空队列中当前可用帧，按 stream_id 保留最新一帧，避免多路时队列积压导致 Full
            while True:
                try:
                    extra = self.frame_queue.get_nowait()
                    frame = extra.get('frame')
                    if frame is not None and frame.size > 0:
                        orig_h, orig_w = frame.shape[:2]
                        if orig_w != CELL_W or orig_h != CELL_H:
                            resized = cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR)
                            extra = {**extra, 'frame': resized, 'original_width': orig_w, 'original_height': orig_h}
                        else:
                            extra = {**extra, 'original_width': orig_w, 'original_height': orig_h}
                    self.latest_frames[extra.get('stream_id', 0)] = extra
                    try:
                        self._update_stream_stats(extra.get('stream_id', 0), extra.get('frame_number'), extra.get('timestamp'))
                    except Exception:
                        pass
                except Exception:
                    break
            
            if self._multi_stream:
                # 允许“部分通道在线”：只要任意一路有帧就开始输出合成画面，
                # 缺失通道在 _build_composite_frame 内用黑屏占位，避免因 RTSP 失败导致整屏无输出。
                if not self.latest_frames:
                    continue
                # 选择一个参考帧用于 frame_number/timestamp（渲染时每格仍按各自帧号取结果）
                frame_data = self.latest_frames.get(0) or next(iter(self.latest_frames.values()))
            
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            if self._multi_stream:
                collect_start = time.time()
                self._collect_detection_results()
                collect_time = (time.time() - collect_start) * 1000
                render_start = time.time()
                rendered_frame = self._build_composite_frame()
                render_time = (time.time() - render_start) * 1000
                # 使用参考帧的时间戳（避免 dict 顺序导致跳变）
                frame_number = frame_data.get('frame_number')
                timestamp = frame_data.get('timestamp')
            else:
                self.frame_count += 1
                collect_start = time.time()
                self._collect_detection_results()
                collect_time = (time.time() - collect_start) * 1000
                render_start = time.time()
                orig_wh = (frame_data.get('original_width'), frame_data.get('original_height'))
                if None in orig_wh:
                    orig_wh = (frame.shape[1], frame.shape[0])
                rendered_frame = self._render_results(
                    frame, frame_number, timestamp, stream_id=0, original_size=orig_wh
                )
                # 一路模式同样叠加连接状态 + FPS，并统一缩放到 1280x720
                try:
                    rendered_frame = self._overlay_stream_status(rendered_frame, 0)
                except Exception:
                    pass
                if rendered_frame.shape[1] != COMPOSITE_OUTPUT_W or rendered_frame.shape[0] != COMPOSITE_OUTPUT_H:
                    rendered_frame = cv2.resize(rendered_frame, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                render_time = (time.time() - render_start) * 1000
            
            fps_frame_count += 1
            
            buffer_start = time.time()
            if self._should_buffer_for_alerts():
                self.video_buffer.add_frame(rendered_frame, frame_number, timestamp)
                self._process_detections(timestamp)
                self.video_buffer.update(rendered_frame, frame_number, timestamp)
            else:
                # 无检测输出时跳过告警缓冲，提升预览帧率
                # 仍保留 _process_detections 的跳过（无检测结果也不会触发告警）
                pass
            buffer_time = (time.time() - buffer_start) * 1000
            
            # 输出节流：保持不高于 target_output_fps，避免渲染/编码层过载
            output_start = time.time()
            now_ts = output_start
            should_output = (min_output_interval <= 0.0) or ((now_ts - last_output_ts) >= min_output_interval)
            if should_output:
                last_output_ts = now_ts
                try:
                    if self.output_queue.full():
                        self.output_queue.get_nowait()
                    self.output_queue.put(rendered_frame, block=False)
                    # 解码-处理-编码流水线：已标注帧同时送入编码器，生成 HLS 流供前端低延迟播放
                    if self.encoder_queue is not None:
                        try:
                            if self.encoder_queue.full():
                                self.encoder_queue.get_nowait()
                            self.encoder_queue.put(rendered_frame.copy(), block=False)
                        except Exception:
                            pass
                except Exception:
                    pass
            output_time = (time.time() - output_start) * 1000
            
            frame_process_time = (time.time() - frame_start_time) * 1000
            processing_times.append(frame_process_time)
            
            if fps_frame_count >= 30:
                elapsed_time = time.time() - fps_start_time
                actual_fps = fps_frame_count / elapsed_time
                avg_process_time = sum(processing_times) / len(processing_times)
                max_process_time = max(processing_times)
                
                active_recordings = self.video_buffer.get_active_recordings()
                
                logger.info(f"🖥️ [DISPLAY SERVICE] Frames: {self.frame_count}")
                logger.info(f"   ├─ Actual FPS: {actual_fps:.2f}")
                logger.info(f"   ├─ Avg process time: {avg_process_time:.1f}ms "
                        f"(collect: {collect_time:.1f}ms, render: {render_time:.1f}ms, "
                        f"buffer: {buffer_time:.1f}ms, output: {output_time:.1f}ms)")
                logger.info(f"   ├─ Max process time: {max_process_time:.1f}ms")
                logger.info(f"   ├─ Active recordings: {active_recordings}")
                logger.info(f"   └─ Queue sizes: frame={self.frame_queue.qsize()}, "
                        f"output={self.output_queue.qsize()}")
                
                fps_start_time = time.time()
                fps_frame_count = 0
                processing_times = []
        
        logger.info("Display service stopped")
    
    def _collect_detection_results(self):
        """收集所有检测服务的最新结果（支持多路 stream_id）"""
        for detector_type, result_queue in self.result_queues.items():
            try:
                last_result = None
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    last_result = result
                    if not result:
                        continue
                    # 记录最近一次“有检测结果输出”的时间，用于纯预览性能优化
                    try:
                        if result.get('enabled', True):
                            self._last_detector_result_ts = time.time()
                    except Exception:
                        pass
                    stream_id = result.get('stream_id', 0)
                    if not result.get('enabled', True):
                        logger.info(f"🔕 [{detector_type.upper()}] Detector disabled, clearing results")
                        if self._multi_stream:
                            self.latest_results[detector_type] = {}
                            self.active_alerts[detector_type] = {}
                            self.triggered_alarms[detector_type] = {}
                        else:
                            self.latest_results[detector_type] = None
                            self.active_alerts[detector_type] = {}
                            self.triggered_alarms[detector_type] = set()
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
                                self._result_by_frame[detector_type][stream_id] = {}
                            buf = self._result_by_frame[detector_type][stream_id]
                            buf[fn] = result
                            if len(buf) > self._result_by_frame_maxlen:
                                for k in sorted(buf.keys())[:-self._result_by_frame_maxlen]:
                                    del buf[k]
                
                # 检测结果超时：多路时按 stream_id 检查，一路时按单结果检查
                if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
                    for sid in list(self.latest_results[detector_type].keys()):
                        r = self.latest_results[detector_type][sid]
                        if r and r.get('timestamp'):
                            time_diff = datetime.now(BEIJING_TZ) - r['timestamp']
                            if time_diff > timedelta(seconds=3):
                                del self.latest_results[detector_type][sid]
                                if sid in self.active_alerts.get(detector_type, {}):
                                    del self.active_alerts[detector_type][sid]
                                if sid in self.triggered_alarms.get(detector_type, {}):
                                    del self.triggered_alarms[detector_type][sid]
                elif not self._multi_stream and self.latest_results.get(detector_type) is not None:
                    r = self.latest_results[detector_type]
                    if r.get('timestamp'):
                        time_diff = datetime.now(BEIJING_TZ) - r['timestamp']
                        if time_diff > timedelta(seconds=3):
                            logger.warning(f"⚠️ [{detector_type.upper()}] No update for {time_diff.total_seconds():.1f}s, clearing results")
                            self.latest_results[detector_type] = None
                            self.active_alerts[detector_type] = {}
                            self.triggered_alarms[detector_type] = set()
            except Exception:
                pass

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
    
    def _get_result_for_stream(self, detector_type, stream_id):
        """按 stream_id 取检测结果（一路时 stream_id 忽略）"""
        if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
            return self.latest_results[detector_type].get(stream_id)
        return self.latest_results.get(detector_type)
    
    # 仅显示“足够新”的检测结果：允许跳帧显示（检测结果不必每帧都有）。
    # 统一阈值以保证各检测器显示策略一致。
    _MAX_FRAME_DIFF_FOR_DISPLAY_DEFAULT = 12.0
    _MAX_FRAME_DIFF_FOR_DISPLAY_BY_DETECTOR = {}

    def _get_result_for_frame(self, detector_type, stream_id, frame_number):
        """仅显示新鲜检测结果：帧差超过阈值则不显示，避免目标移开后框残留。"""
        buf = self._result_by_frame.get(detector_type, {}).get(stream_id, {})
        max_diff = self._MAX_FRAME_DIFF_FOR_DISPLAY_BY_DETECTOR.get(
            detector_type, self._MAX_FRAME_DIFF_FOR_DISPLAY_DEFAULT
        )
        if buf and frame_number is not None:
            if frame_number in buf:
                return buf[frame_number]
            keys = sorted(buf.keys(), reverse=True)
            for fn in keys:
                if fn <= frame_number and (frame_number - fn) <= max_diff:
                    return buf[fn]
            closest = min(keys, key=lambda k: abs(k - frame_number))
            if abs(closest - frame_number) <= max_diff:
                return buf[closest]
            # 有帧缓存但无新鲜结果：不显示，避免目标移开后框残留原地
            return None
        # 无帧缓存（兼容）：退回到 stream 最新结果
        return self._get_result_for_stream(detector_type, stream_id)
    
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
        """合成预览画面：单路/多路均使用统一降采样；输出固定 1280x720。单路时单格放大至满屏，多路时 3x3 九宫格。task 模式下仅显示选中任务对应通道。"""
        blank_color = (42, 23, 15)  # BGR
        blank_cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        blank_cell[:] = blank_color

        # 任务通道模式：多路时仅显示选中的 stream_id 单格放大至满屏
        if self._multi_stream and self.preview_mode == 'task':
            sid = self.preview_task_stream_id
            fd = self.latest_frames.get(sid)
            if not fd or 'frame' not in fd or fd.get('frame') is None:
                out = cv2.resize(blank_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                return out
            frame_cell = fd['frame'].copy()
            orig_wh = (fd.get('original_width'), fd.get('original_height'))
            if None in orig_wh:
                orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
            frame_cell = self._render_results(
                frame_cell, fd.get('frame_number'), fd.get('timestamp'), stream_id=sid, original_size=orig_wh
            )
            frame_cell = self._overlay_stream_status(frame_cell, sid)
            return cv2.resize(frame_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        if self.stream_count == 1:
            fd = self.latest_frames.get(0)
            if not fd or 'frame' not in fd or fd.get('frame') is None:
                out = cv2.resize(blank_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                return out
            frame_cell = fd['frame'].copy()
            orig_wh = (fd.get('original_width'), fd.get('original_height'))
            if None in orig_wh:
                orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
            frame_cell = self._render_results(
                frame_cell, fd.get('frame_number'), fd.get('timestamp'), stream_id=0, original_size=orig_wh
            )
            frame_cell = self._overlay_stream_status(frame_cell, 0)
            return cv2.resize(frame_cell, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        # 多路：固定九宫格，每格 CELL_W x CELL_H，合成后 1278x720 再缩放到 1280x720
        cells = []
        for idx in range(9):
            if idx < self.stream_count:
                sid = idx
                fd = self.latest_frames.get(sid)
                if not fd or 'frame' not in fd or fd.get('frame') is None:
                    frame_cell = blank_cell.copy()
                else:
                    frame_cell = fd['frame'].copy()
                    orig_wh = (fd.get('original_width'), fd.get('original_height'))
                    if None in orig_wh:
                        orig_wh = (frame_cell.shape[1], frame_cell.shape[0])
                    frame_cell = self._render_results(
                        frame_cell, fd.get('frame_number'), fd.get('timestamp'), stream_id=sid, original_size=orig_wh
                    )
            else:
                frame_cell = blank_cell.copy()

            try:
                frame_cell = self._overlay_stream_status(frame_cell, idx)
            except Exception:
                pass
            cells.append(frame_cell)

        row0 = np.hstack(cells[0:3])
        row1 = np.hstack(cells[3:6])
        row2 = np.hstack(cells[6:9])
        composite = np.vstack([row0, row1, row2])
        if composite.shape[1] != COMPOSITE_OUTPUT_W or composite.shape[0] != COMPOSITE_OUTPUT_H:
            composite = cv2.resize(composite, (COMPOSITE_OUTPUT_W, COMPOSITE_OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        return composite
    
    def _render_results(self, frame, frame_number, timestamp, stream_id=0, original_size=None):
        """渲染所有检测结果到画面。original_size=(w,h) 为原始分辨率，用于将检测框坐标缩放到当前显示尺寸。"""
        rendered = frame.copy()
        disp_wh = (frame.shape[1], frame.shape[0])
        orig_wh = original_size if original_size and None not in original_size else disp_wh
        get_res = lambda det: self._get_result_for_frame(det, stream_id, frame_number)
        if get_res('fall') is not None:
            rendered = self._render_fall_detections(rendered, get_res('fall'), stream_id, orig_wh, disp_wh)
        if get_res('ventilator') is not None:
            rendered = self._render_ventilator_detections(rendered, get_res('ventilator'), stream_id, orig_wh, disp_wh)
        if get_res('fight') is not None:
            rendered = self._render_fight_detections(rendered, get_res('fight'), stream_id, orig_wh, disp_wh)
        if get_res('crowd') is not None:
            rendered = self._render_crowd_detections(rendered, get_res('crowd'), stream_id, orig_wh, disp_wh)
        if get_res('helmet') is not None:
            rendered = self._render_helmet_detections(rendered, get_res('helmet'), stream_id, orig_wh, disp_wh)
        if get_res('window_door_inside') is not None:
            rendered = self._render_window_door_detections(rendered, get_res('window_door_inside'), stream_id, detector_key='window_door_inside', orig_wh=orig_wh, disp_wh=disp_wh)
        if get_res('window_door_outside') is not None:
            rendered = self._render_window_door_detections(rendered, get_res('window_door_outside'), stream_id, detector_key='window_door_outside', orig_wh=orig_wh, disp_wh=disp_wh)
        return rendered
    
    def _render_fall_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """🎯 渲染跌倒检测结果 - 使用display_alerts显示所有跌倒的人"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts['fall'].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts['fall']
        display_alerts = result.get('display_alerts', [])
        current_display_ids = set()
        
        for alert in display_alerts:
            tracker_id = alert['tracker_id']
            current_display_ids.add(tracker_id)
            if tracker_id not in alerts_dict:
                alerts_dict[tracker_id] = {
                    'bbox': alert['bbox'],
                    'start_time': result.get('timestamp'),
                    'info': alert,
                    'is_recording': alert.get('is_recording', False)
                }
            else:
                alerts_dict[tracker_id]['bbox'] = alert['bbox']
                alerts_dict[tracker_id]['is_recording'] = alert.get('is_recording', False)
        
        active_ids = list(alerts_dict.keys())
        for tracker_id in active_ids:
            if tracker_id not in current_display_ids:
                self._clear_bbox_smooth('fall', stream_id, tracker_id)
                del alerts_dict[tracker_id]
        
        for tracker_id, alert in alerts_dict.items():
            bbox = alert['bbox']
            bbox = self._smooth_bbox('fall', stream_id, tracker_id, bbox, 'xyxy')
            if bbox is None:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xyxy')
            x1, y1, x2, y2 = map(int, bbox)
            
            chinese_text = "跌倒警报"
            english_text = "fall"
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2], chinese_text, english_text, color=(0, 0, 255))
        
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
        # 🔍 调试日志
        if display_alerts:
            logger.info(f"🔍 [VENTILATOR DEBUG] Received {len(display_alerts)} display alerts:")
            for idx, alert in enumerate(display_alerts):
                is_rec = "🎬录制" if alert.get('is_recording', False) else "👁️仅显示"
                logger.info(f"     Alert {idx}: tracker_id={alert.get('tracker_id')}, "
                        f"bbox={alert.get('bbox')}, {is_rec}")
        
        if detections:
            logger.info(f"🔍 [VENTILATOR DEBUG] {len(detections)} detections will trigger recording")
        
        # 🎯 使用 display_alerts 来维护活跃警报（用于显示所有未佩戴的人）
        current_display_ids = set()
        
        for alert in display_alerts:
            tracker_id = alert['tracker_id']
            current_display_ids.add(tracker_id)
            
            if tracker_id not in alerts_dict:
                alerts_dict[tracker_id] = {
                    'bbox': alert['bbox'],
                    'start_time': result.get('timestamp'),
                    'info': alert,
                    'is_recording': alert.get('is_recording', False)
                }
                status = "录制中" if alert.get('is_recording', False) else "仅显示"
                logger.info(f"🚨 [VENTILATOR] New alert added: ID={tracker_id}, [{status}]")
            else:
                # 更新bbox和录制状态
                alerts_dict[tracker_id]['bbox'] = alert['bbox']
                alerts_dict[tracker_id]['is_recording'] = alert.get('is_recording', False)
        
        active_ids = list(alerts_dict.keys())
        for tracker_id in active_ids:
            if tracker_id not in current_display_ids:
                self._clear_bbox_smooth('ventilator', stream_id, tracker_id)
                del alerts_dict[tracker_id]
        
        # ========== 绘制所有活跃警报（红色框 - 最高优先级，最后绘制） ==========
        logger.info(f"🎨 [VENTILATOR DEBUG] Drawing {len(self.active_alerts['ventilator'])} alerts")
        for tracker_id, alert in alerts_dict.items():
            bbox = alert['bbox']
            bbox = self._smooth_bbox('ventilator', stream_id, tracker_id, bbox, 'xyxy')
            if bbox is None:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xyxy')
            is_recording = alert.get('is_recording', False)
            
            logger.info(f"     Drawing alert: ID={tracker_id}, bbox={bbox}, "
                    f"{'🎬录制中' if is_recording else '👁️仅显示'}")
            
            x1, y1, x2, y2 = map(int, bbox)
            # 裁剪到画面内，避免越界导致“画了但看不见/异常”
            fh, fw = frame.shape[:2]
            x1 = int(max(0, min(fw - 1, x1)))
            y1 = int(max(0, min(fh - 1, y1)))
            x2 = int(max(0, min(fw - 1, x2)))
            y2 = int(max(0, min(fh - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            # 绘制标签
            chinese_text = "未佩戴呼吸机"
            english_text = "no ventilator"
            
            # 如果正在录制，添加特殊标记
            if is_recording:
                english_text = f"🎬 {english_text}"
            
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2], chinese_text, english_text, color=(0, 0, 255))
        
        return frame
    
    def _render_fight_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """渲染打架检测结果 - 修复版：支持检测框实时跟随 + 自动清理过期警报"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts['fight'].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts['fight']
        detections = result.get('detections', [])
        current_alert_ids = set()
        
        for det in detections:
            alert_id = det.get('alert_id', 0)
            current_alert_ids.add(alert_id)
            alerts_dict[alert_id] = {
                'bbox': det['bbox'],
                'start_time': result.get('timestamp'),
                'info': det,
                'is_ongoing': det.get('is_ongoing', False)
            }
        
        active_ids = list(alerts_dict.keys())
        for alert_id in active_ids:
            if alert_id not in current_alert_ids:
                self._clear_bbox_smooth('fight', stream_id, alert_id)
                del alerts_dict[alert_id]
        
        for alert_id, alert in alerts_dict.items():
            bbox = alert['bbox']
            bbox = self._smooth_bbox('fight', stream_id, alert_id, bbox, 'xyxy')
            if bbox is None:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xyxy')
            x1, y1, x2, y2 = map(int, bbox)

            chinese_text = "打架警报"
            english_text = "fight"
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2], chinese_text, english_text, color=(0, 0, 255))
        
        return frame
    
    def _render_crowd_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """🎯 渲染人员聚集检测结果 - 使用display_alerts持续显示"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts['crowd'].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts['crowd']
        display_alerts = result.get('display_alerts', [])
        current_display_ids = set()
        
        for alert in display_alerts:
            cluster_id = alert['cluster_id']
            current_display_ids.add(cluster_id)
            if cluster_id not in alerts_dict:
                alerts_dict[cluster_id] = {
                    'bbox': alert['bbox'],
                    'start_time': result.get('timestamp'),
                    'info': alert
                }
            else:
                alerts_dict[cluster_id]['bbox'] = alert['bbox']
                alerts_dict[cluster_id]['info'] = alert
        
        active_ids = list(alerts_dict.keys())
        for cluster_id in active_ids:
            if cluster_id not in current_display_ids:
                self._clear_bbox_smooth('crowd', stream_id, cluster_id)
                del alerts_dict[cluster_id]
        
        h, w = frame.shape[:2]
        for cluster_id, alert in alerts_dict.items():
            bbox = alert['bbox']
            bbox = self._smooth_bbox('crowd', stream_id, cluster_id, bbox, 'xyxy')
            if bbox is None:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xyxy')
            x1, y1, x2, y2 = map(int, bbox)
            # 垂直拉伸报警框：在头部聚集框基础上向下扩展，尽量覆盖身体区域
            box_h = max(1, y2 - y1)
            stretch = int(box_h * 1.5)
            y2_ext = min(h - 1, y2 + stretch)
            count = alert['info']['count']
            
            chinese_text = "人群聚集"
            english_text = f"crowd {count} people"
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2_ext], chinese_text, english_text, color=(0, 0, 255))
        
        return frame
    
    def _render_helmet_detections(self, frame, result, stream_id=0, orig_wh=None, disp_wh=None):
        """🎯 渲染安全帽检测结果 - 使用display_alerts显示所有人"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts['helmet'].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts['helmet']
        display_alerts = result.get('display_alerts', [])
        current_display_ids = set()
        
        for alert in display_alerts:
            track_id = alert['track_id']
            current_display_ids.add(track_id)
            if track_id not in alerts_dict:
                alerts_dict[track_id] = {
                    'bbox': alert['bbox'],
                    'start_time': result.get('timestamp'),
                    'info': alert,
                    'is_recording': alert.get('is_recording', False)
                }
            else:
                alerts_dict[track_id]['bbox'] = alert['bbox']
                alerts_dict[track_id]['is_recording'] = alert.get('is_recording', False)
        
        active_ids = list(alerts_dict.keys())
        for track_id in active_ids:
            if track_id not in current_display_ids:
                self._clear_bbox_smooth('helmet', stream_id, track_id)
                del alerts_dict[track_id]
        fh, fw = frame.shape[:2]
        for track_id, alert in alerts_dict.items():
            bbox = alert.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            bbox = self._smooth_bbox('helmet', stream_id, track_id, bbox, 'xywh')
            if bbox is None or len(bbox) < 4:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xywh')
            try:
                x, y, w, h = [float(b) for b in bbox[:4]]
                x1 = int(max(0, x))
                y1 = int(max(0, y))
                x2 = int(min(fw, x + w))
                y2 = int(min(fh, y + h))
            except (ValueError, TypeError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            
            chinese_text = "未佩戴安全帽"
            english_text = "no helmet"
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2], chinese_text, english_text, color=(0, 0, 255))
        
        return frame
    
    def _render_window_door_detections(self, frame, result, stream_id=0, detector_key='window_door_inside', orig_wh=None, disp_wh=None):
        """🎯 渲染窗户门检测结果 - 使用display_alerts显示所有打开的窗户/门"""
        disp_wh = disp_wh or (frame.shape[1], frame.shape[0])
        orig_wh = orig_wh or disp_wh
        alerts_dict = self.active_alerts[detector_key].setdefault(stream_id, {}) if self._multi_stream else self.active_alerts[detector_key]
        display_alerts = result.get('display_alerts', [])
        current_display_ids = set()
        
        for alert in display_alerts:
            track_id = alert['track_id']
            current_display_ids.add(track_id)
            if track_id not in alerts_dict:
                alerts_dict[track_id] = {
                    'bbox': alert['bbox'],
                    'start_time': result.get('timestamp'),
                    'info': alert,
                    'is_recording': alert.get('is_recording', False)
                }
            else:
                alerts_dict[track_id]['bbox'] = alert['bbox']
                alerts_dict[track_id]['is_recording'] = alert.get('is_recording', False)
        
        active_ids = list(alerts_dict.keys())
        for track_id in active_ids:
            if track_id not in current_display_ids:
                self._clear_bbox_smooth(detector_key, stream_id, track_id)
                del alerts_dict[track_id]
        fh, fw = frame.shape[:2]
        for track_id, alert in alerts_dict.items():
            bbox = alert.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            bbox = self._smooth_bbox(detector_key, stream_id, track_id, bbox, 'xyxy')
            if bbox is None or len(bbox) < 4:
                continue
            bbox = self._scale_bbox_to_display(bbox, orig_wh, disp_wh, 'xyxy')
            try:
                x1 = int(max(0, min(fw - 1, float(bbox[0]))))
                y1 = int(max(0, min(fh - 1, float(bbox[1]))))
                x2 = int(max(0, min(fw, float(bbox[2]))))
                y2 = int(max(0, min(fh, float(bbox[3]))))
            except (ValueError, TypeError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            
            display_name = alert['info'].get('display_name', '警报')
            
            chinese_text = display_name
            english_text = "alert!"
            frame = self._draw_unified_alert_box(frame, [x1, y1, x2, y2], chinese_text, english_text, color=(0, 0, 255))
        
        return frame
    
    def _process_detections(self, timestamp):
        """🎯 改进的警报处理逻辑（支持多路 stream_id）"""
        def _iter_results(detector_type):
            if self._multi_stream and isinstance(self.latest_results.get(detector_type), dict):
                for sid, res in self.latest_results[detector_type].items():
                    if res:
                        yield sid, res
            elif self.latest_results.get(detector_type):
                yield 0, self.latest_results[detector_type]
        
        def _triggered_set(detector_type, stream_id):
            if self._multi_stream:
                return self.triggered_alarms[detector_type].setdefault(stream_id, set())
            return self.triggered_alarms[detector_type]
        
        # 跌倒检测：使用 display_alerts 中标记了 is_recording 的告警，确保遵循检测器内部冷却期
        for stream_id, fall_res in _iter_results('fall'):
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
                    alarm_id = f"fall_ch{stream_id}_{tracker_id}_{int(timestamp.timestamp())}"
                    triggered.add(tracker_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='fall', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_tracker_ids)
        
        # 呼吸机检测：同样以 display_alerts + is_recording 为准，多通道互不干扰
        for stream_id, vent_res in _iter_results('ventilator'):
            alerts = vent_res.get('display_alerts') or vent_res.get('detections', [])
            current_tracker_ids = set()
            triggered = _triggered_set('ventilator', stream_id)
            for alert in alerts:
                if not alert.get('is_recording', True):
                    continue
                tracker_id = alert.get('tracker_id', 0)
                current_tracker_ids.add(tracker_id)
                if tracker_id not in triggered:
                    alarm_id = f"ventilator_ch{stream_id}_{tracker_id}_{int(timestamp.timestamp())}"
                    triggered.add(tracker_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='ventilator', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_tracker_ids)
        
        for stream_id, fight_res in _iter_results('fight'):
            detections = fight_res.get('detections', [])
            current_alert_ids = set()
            triggered = _triggered_set('fight', stream_id)
            for det in detections:
                alert_id = det.get('alert_id', 0)
                current_alert_ids.add(alert_id)
                if alert_id not in triggered:
                    alarm_id = f"fight_ch{stream_id}_{alert_id}_{int(timestamp.timestamp())}"
                    triggered.add(alert_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='fight', alarm_info=det, stream_id=stream_id)
            triggered -= (triggered - current_alert_ids)
        
        for stream_id, crowd_res in _iter_results('crowd'):
            detections = crowd_res.get('detections', [])
            current_alert_ids = set()
            triggered = _triggered_set('crowd', stream_id)
            for det in detections:
                alert_id = det.get('alert_id', 0)
                current_alert_ids.add(alert_id)
                if alert_id not in triggered:
                    alarm_id = f"crowd_ch{stream_id}_{alert_id}_{int(timestamp.timestamp())}"
                    triggered.add(alert_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='crowd', alarm_info=det, stream_id=stream_id)
            triggered -= (triggered - current_alert_ids)
        
        # 安全帽检测：使用 display_alerts 中 is_recording=True 的目标
        for stream_id, helmet_res in _iter_results('helmet'):
            alerts = helmet_res.get('display_alerts') or helmet_res.get('detections', [])
            current_track_ids = set()
            triggered = _triggered_set('helmet', stream_id)
            for alert in alerts:
                if not alert.get('is_recording', True):
                    continue
                track_id = alert.get('track_id', alert.get('tracker_id', 0))
                current_track_ids.add(track_id)
                if track_id not in triggered:
                    alarm_id = f"helmet_ch{stream_id}_{track_id}_{int(timestamp.timestamp())}"
                    triggered.add(track_id)
                    self.video_buffer.trigger_alarm(alarm_id=alarm_id, alarm_type='helmet', alarm_info=alert, stream_id=stream_id)
            triggered -= (triggered - current_track_ids)
        
        # 窗户门检测：仓内/仓外分别触发录制（互不干扰）
        for detector_key in ("window_door_inside", "window_door_outside"):
            for stream_id, wd_res in _iter_results(detector_key):
                alerts = wd_res.get('display_alerts') or wd_res.get('detections', [])
                current_track_ids = set()
                triggered = _triggered_set(detector_key, stream_id)
                for alert in alerts:
                    if not alert.get('is_recording', True):
                        continue
                    track_id = alert.get('track_id', 0)
                    current_track_ids.add(track_id)
                    if track_id not in triggered:
                        alarm_id = f"{detector_key}_ch{stream_id}_{track_id}_{int(timestamp.timestamp())}"
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
                    mode = cmd.get('mode', 'merged')
                    self.preview_mode = mode if mode in ('merged', 'task') else 'merged'
                    sid = cmd.get('stream_id', 0)
                    self.preview_task_stream_id = max(0, min(self.stream_count - 1, int(sid)))
        except Exception as e:
            pass
    
    def stop(self):
        """停止展示服务"""
        logger.info("Stopping display service...")
        self.running = False
        
        # 等待所有视频保存完成
        if hasattr(self, 'video_buffer'):
            self.video_buffer.shutdown(timeout=30)


def run_display_service(frame_queue, result_queues, output_queue, control_queue,
                        video_output_dir, fps, stream_count=1, alert_queue=None, font_path_config=None,
                        encoder_queue=None, alert_box_mode='blink'):
    """进程入口函数。stream_count: 1=一路, 2=两路, 4=四路, 9=九路, 16=十六路；alert_queue: 告警事件队列；font_path_config: 中文字体路径；encoder_queue: 已标注帧队列供 HLS 编码。"""
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
        encoder_queue,
    )
    # 根据配置设置报警框显示模式
    try:
        service.alert_box_mode = str(alert_box_mode or 'blink').strip().lower()
    except Exception:
        service.alert_box_mode = 'blink'
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