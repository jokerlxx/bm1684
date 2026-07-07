"""
第一层：流服务 (Stream Service)
职责：
1. 读取RTSP视频流或本地视频文件
2. 通过共享队列分发帧数据给检测服务
3. 支持启动、停止、重连
"""

import cv2
import time
import multiprocessing as mp
from datetime import datetime, timezone, timedelta

# 北京时间（UTC+8），用于帧时间戳与告警时间一致
BEIJING_TZ = timezone(timedelta(hours=8))
import logging
import signal
import sys
import os
import queue

from core.logging_utils import log_pull_fps

try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [StreamService] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StreamService:
    """流服务 - 支持RTSP流和视频文件"""
    
    def __init__(self, source, frame_queue, control_queue, fps=30, input_mode=0, stream_id=0, stream_count=1):
        """
        Args:
            source: 视频源（RTSP URL或视频文件路径）
            frame_queue: multiprocessing.Queue 或 list[Queue]，用于输出帧数据（可同时投递到多个消费者队列）
            control_queue: multiprocessing.Queue，用于接收控制命令
            fps: 目标帧率
            input_mode: 输入模式 (0=RTSP, 1=视频文件)
            stream_id: 流编号（多路时 0,1,2,3）
            stream_count: 总路数（用于低延迟策略判断）
        """
        self.source = source
        # 兼容单队列/多队列：多队列用于“展示队列/检测队列”分离，避免抢帧导致预览 FPS 降低
        if isinstance(frame_queue, (list, tuple)):
            self.frame_queues = list(frame_queue)
        else:
            self.frame_queues = [frame_queue]
        self.control_queue = control_queue
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.input_mode = input_mode
        self.stream_id = stream_id
        self.stream_count = max(1, int(stream_count))
        
        self.running = False
        self.cap = None
        self.decoder = None
        # 与 yolov8_bmcv 示例一致：如有 Sophon 环境，则创建设备句柄，便于硬件编解码
        self.device_id = 0
        self.sophon_handle = sail.Handle(self.device_id) if SOPHON_AVAILABLE else None
        self.frame_count = 0
        
        self.is_video_file = (input_mode == 1)
        self.video_loop = True
        self.total_frames = 0
        self._full_drop_count = 0
        self._last_full_log_time = 0.0
        self.realtime_drop_mode = (not self.is_video_file and self.stream_count >= 9)

    def _qsize_safe(self, q):
        try:
            return q.qsize()
        except Exception:
            return 0

    def _full_safe(self, q):
        try:
            return q.full()
        except Exception:
            return False

    def _put_frame_to_queues(self, frame_data):
        """将帧投递到所有输出队列（各队列独立做低延迟与满队列丢帧处理）。"""
        for q in self.frame_queues:
            try:
                # 一路时尽量只保留最新帧（对展示与检测都降低延迟）
                if self.stream_count == 1:
                    try:
                        while True:
                            q.get_nowait()
                    except Exception:
                        pass
                elif self._full_safe(q):
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                q.put(frame_data, block=False)
            except Exception:
                # 单个队列失败不影响其它队列
                pass
        
    def setup_signal_handlers(self):
        """设置信号处理器，支持优雅退出"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"Received signal {signum}, stopping gracefully...")
        self.stop()
        sys.exit(0)
    
    def connect_stream(self):
        """连接视频源（RTSP流或视频文件）"""
        if self.is_video_file:
            return self._connect_video_file()
        else:
            return self._connect_rtsp_stream()
    
    def _connect_rtsp_stream(self):
        """连接RTSP流"""
        logger.info(f"🎥 [Stream {self.stream_id}] Connecting to RTSP: {self.source[:60]}...")
        # 优先尝试 Sophon 硬件解码器（与 yolov8_bmcv 示例一致），失败时退回 OpenCV
        if SOPHON_AVAILABLE and self.sophon_handle is not None:
            try:
                self.decoder = sail.Decoder(self.source, True, self.device_id)
                if not self.decoder.is_opened():
                    logger.error("❌ Failed to open RTSP stream with Sophon Decoder")
                    self.decoder.release()
                    self.decoder = None
                else:
                    try:
                        shape = self.decoder.get_frame_shape()
                        if len(shape) >= 2:
                            height, width = int(shape[0]), int(shape[1])
                        else:
                            height = width = 0
                    except Exception:
                        height = width = 0
                    try:
                        actual_fps = float(self.decoder.get_fps())
                    except Exception:
                        actual_fps = 0.0
                    logger.info(
                        f"✅ [Stream {self.stream_id}] RTSP connected via Sophon Decoder: "
                        f"{width}x{height} @ {actual_fps}fps"
                    )
                    return True
            except Exception as e:
                logger.error(f"❌ Sophon Decoder open failed for RTSP: {e}")
                self.decoder = None

        # 尽量降低 FFmpeg 内部缓存（OpenCV FFmpeg 后端生效）。不覆盖用户自定义环境变量。
        # 参考：OPENCV_FFMPEG_CAPTURE_OPTIONS 形如 "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
        if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

        self.cap = cv2.VideoCapture(self.source)
        
        # 优化RTSP参数降低延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            logger.error("❌ Failed to open RTSP stream")
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return False
        
        # 获取流信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"✅ [Stream {self.stream_id}] RTSP connected: {width}x{height} @ {actual_fps}fps")
        return True
    
    def _connect_video_file(self):
        """连接视频文件"""
        logger.info(f"📹 [Stream {self.stream_id}] Opening video file: {self.source}")
        
        # 检查文件是否存在
        if not os.path.exists(self.source):
            logger.error(f"❌ Video file not found: {self.source}")
            return False
        # 优先使用 Sophon 硬件解码器；失败时退回 OpenCV VideoCapture
        if SOPHON_AVAILABLE and self.sophon_handle is not None:
            try:
                self.decoder = sail.Decoder(self.source, True, self.device_id)
                if not self.decoder.is_opened():
                    logger.error(f"❌ Failed to open video file with Sophon Decoder: {self.source}")
                    self.decoder.release()
                    self.decoder = None
                else:
                    try:
                        shape = self.decoder.get_frame_shape()
                        if len(shape) >= 2:
                            height, width = int(shape[0]), int(shape[1])
                        else:
                            height = width = 0
                    except Exception:
                        height = width = 0
                    try:
                        actual_fps = float(self.decoder.get_fps())
                    except Exception:
                        actual_fps = float(self.fps)
                    self.total_frames = 0
                    duration = 0.0
                    logger.info(f"✅ Video file opened successfully via Sophon Decoder:")
                    logger.info(f"   ├─ Resolution: {width}x{height}")
                    logger.info(f"   ├─ FPS (decoder): {actual_fps}")
                    logger.info(f"   ├─ Total frames: {self.total_frames or 'unknown'}")
                    logger.info(f"   ├─ Duration: {duration:.2f}s")
                    logger.info(f"   └─ Loop mode: {'ON' if self.video_loop else 'OFF'}")
                    return True
            except Exception as e:
                logger.error(f"❌ Sophon Decoder open failed for video file: {e}")
                self.decoder = None

        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"❌ Failed to open video file: {self.source}")
            return False
        
        # 获取视频信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = self.total_frames / actual_fps if actual_fps > 0 else 0
        
        logger.info(f"✅ Video file opened successfully:")
        logger.info(f"   ├─ Resolution: {width}x{height}")
        logger.info(f"   ├─ FPS: {actual_fps}")
        logger.info(f"   ├─ Total frames: {self.total_frames}")
        logger.info(f"   ├─ Duration: {duration:.2f}s")
        logger.info(f"   └─ Loop mode: {'ON' if self.video_loop else 'OFF'}")
        
        return True
    
    def reconnect_stream(self):
        """重新连接流"""
        logger.warning("⚠️ Attempting to reconnect...")
        if self.decoder is not None:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        time.sleep(2)  # 等待2秒后重连
        return self.connect_stream()
    
    def start(self):
        """启动流服务"""
        mode_name = "VIDEO FILE" if self.is_video_file else "RTSP STREAM"
        logger.info(f"🚀 Stream service starting ({mode_name} mode)...")
        if self.realtime_drop_mode:
            logger.info("⚡ [Stream %s] High-density realtime mode enabled: keep latest frames, skip input throttling", self.stream_id)
        self.setup_signal_handlers()
        
        # 🆕 启动前检查队列健康状态
        is_healthy, status_msg = self._check_queue_health()
        if is_healthy:
            logger.info(f"✅ Queue health check passed: {status_msg}")
        else:
            logger.error(f"❌ Queue health check failed: {status_msg}")
            logger.error("   └─ Cannot start stream service with unhealthy queue")
            return
        
        # 先置 running=True，便于在连接阶段也能响应 stop/信号
        self.running = True

        # RTSP 在 stop->start 的短窗口内容易出现 NVR 侧 500/资源未释放，使用退避重试提高启动稳定性
        connected = False
        if self.is_video_file:
            connected = self.connect_stream()
        else:
            attempt = 0
            max_attempts = 30  # 约 1~2 分钟内的退避重试；避免 URL 错误时无限卡死
            while self.running and not connected and attempt < max_attempts:
                connected = self.connect_stream()
                if connected:
                    break
                attempt += 1

                # 连接失败也检查是否收到 stop，避免阻塞退出
                self._check_control_commands()
                if not self.running:
                    break

                sleep_s = min(10.0, 0.5 * (2 ** min(attempt, 5)))
                logger.warning(
                    f"⚠️ [Stream {self.stream_id}] RTSP connect failed, retry {attempt}/{max_attempts} in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)

        if not connected:
            logger.error("❌ Failed to start stream service")
            self.stop()
            return

        self.run()
    
    def run(self):
        """主循环：读取帧并分发（低延迟策略：先取流再限速，RTSP 丢旧取新）"""
        reconnect_count = 0
        max_reconnect = 5 if not self.is_video_file else 0  # 视频文件不重连
        last_tick = time.time()
        
        # 帧率统计
        fps_start_time = time.time()
        fps_frame_count = 0
        
        # 🆕 视频文件进度统计
        last_progress_report = 0
        
        # 🆕 队列健康检查计数器（每100帧检查一次）
        health_check_counter = 0
        health_check_interval = 100
        
        while self.running:
            # 检查控制命令（stop 会置 self.cap/decoder=None，必须在本轮内退出，否则下一行读取会崩溃）
            self._check_control_commands()
            if not self.running or (self.cap is None and self.decoder is None):
                break

            # 低延迟策略：先读取再按需 sleep，保证入队帧尽量是“当前最新”
            # 读取帧：若使用 Sophon 解码，则直接 Decoder.read -> BMImage -> asmat()；
            # 否则保持原有 OpenCV 逻辑（RTSP 模式下尽量“丢旧取新”以降低延迟）
            if self.decoder is not None:
                bmimg = sail.BMImage()
                ret_code = self.decoder.read(self.sophon_handle, bmimg)
                if ret_code != 0:
                    ret, frame = False, None
                else:
                    try:
                        frame = bmimg.asmat()
                        ret = frame is not None
                    except Exception as e:
                        logger.error(f"❌ Failed to convert BMImage to numpy: {e}")
                        ret, frame = False, None
            else:
                if not self.is_video_file:
                    try:
                        # 当队列已有帧（消费者稍慢）时，多 grab 几次跳到最新帧，避免堆积旧帧
                        try:
                            qsize = max(self._qsize_safe(q) for q in self.frame_queues) if self.frame_queues else 0
                        except Exception:
                            qsize = 0
                        skip = 1  # 队列空时只丢 1 帧，尽量低延迟
                        if qsize > 0 or any(self._full_safe(q) for q in self.frame_queues):
                            if self.realtime_drop_mode:
                                skip = min(12, 2 + qsize * 2)
                            else:
                                skip = min(8, 1 + qsize)  # 有积压时多丢几帧以追上实时
                        for _ in range(skip):
                            self.cap.grab()
                        ret, frame = self.cap.retrieve()
                        if not ret:
                            ret, frame = self.cap.read()
                    except Exception:
                        if self.cap is None:
                            break
                        ret, frame = self.cap.read()
                else:
                    ret, frame = self.cap.read()

            # 读取后再做帧率限制，避免“等完 sleep 才读”带来的额外延迟
            now = time.time()
            elapsed = now - last_tick
            if (not self.realtime_drop_mode) and elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            last_tick = time.time()
            
            if not ret:
                # 🆕 视频文件读取结束，循环播放
                if self.is_video_file:
                    if self.video_loop:
                        logger.info(f"🔄 Video file ended, restarting from beginning...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头
                        self.frame_count = 0
                        continue
                    else:
                        logger.info("✅ Video file playback completed (no loop)")
                        self.stop()
                        break
                
                # RTSP流读取失败，尝试重连
                logger.warning(f"⚠️ Failed to read frame (attempt {reconnect_count + 1}/{max_reconnect})")
                reconnect_count += 1
                
                if reconnect_count >= max_reconnect:
                    logger.error("❌ Max reconnect attempts reached, stopping...")
                    self.stop()
                    break
                
                if not self.reconnect_stream():
                    continue
                else:
                    reconnect_count = 0
                    continue
            
            # 重置重连计数
            reconnect_count = 0
            self.frame_count += 1
            fps_frame_count += 1
            health_check_counter += 1
            
            # 🆕 定期检查队列健康状态
            if health_check_counter >= health_check_interval:
                is_healthy, status_msg = self._check_queue_health()
                if not is_healthy:
                    logger.error(f"❌ Queue health check failed: {status_msg}")
                    logger.error("   └─ Stopping stream service due to unhealthy queue...")
                    self.stop()
                    break
                health_check_counter = 0
            
            # 准备帧数据（含 stream_id 供多路检测与展示）
            frame_data = {
                'frame': frame,
                'frame_number': self.frame_count,
                'timestamp': datetime.now(BEIJING_TZ),
                'shape': frame.shape,
                'stream_id': self.stream_id
            }
            
            # 非阻塞方式放入队列
            try:
                # 低延迟策略：一路时尽量只保留“最新帧”，避免队列积压导致播放延迟
                # 多路时多个采集进程共享同一队列，不能在此彻底清空（会误删其它路帧），仍使用“满则丢一帧”策略
                self._put_frame_to_queues(frame_data)
                
                # 🆕 打印帧率统计
                if fps_frame_count >= 30:
                    elapsed_time = time.time() - fps_start_time
                    actual_fps = fps_frame_count / elapsed_time
                    
                    if self.is_video_file:
                        # 视频文件：显示进度
                        progress = (self.frame_count % self.total_frames) / self.total_frames * 100 if self.total_frames > 0 else 0
                        logger.info(f"📹 [VIDEO FILE] Frames: {self.frame_count}, "
                                f"Progress: {progress:.1f}%, "
                                f"Actual FPS: {actual_fps:.2f}, "
                                f"Queue(max): {max(self._qsize_safe(q) for q in self.frame_queues) if self.frame_queues else 0}")
                    else:
                        # RTSP流：显示帧率
                        logger.info(f"📹 [RTSP STREAM] Frames: {self.frame_count}, "
                                f"Actual FPS: {actual_fps:.2f}, "
                                f"Target FPS: {self.fps}, "
                                f"Queue(max): {max(self._qsize_safe(q) for q in self.frame_queues) if self.frame_queues else 0}")
                    log_pull_fps(
                        self.stream_id,
                        actual_fps,
                        target_fps=self.fps,
                        queue_size=max(self._qsize_safe(q) for q in self.frame_queues) if self.frame_queues else 0,
                        source_type="视频文件" if self.is_video_file else "RTSP拉流",
                    )
                    
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
            except BrokenPipeError as e:
                # 管道断开错误（消费者进程可能已退出）
                logger.error(f"❌ BrokenPipeError: Consumer process may have crashed!")
                logger.error(f"   ├─ Error details: {type(e).__name__}: {str(e) if str(e) else 'No details'}")
                logger.error(f"   ├─ Frame number: {self.frame_count}")
                logger.error(f"   └─ Stopping stream service...")
                self.stop()
                break
                
            except OSError as e:
                # 操作系统错误（可能是队列相关）
                logger.error(f"❌ OSError when putting frame to queue!")
                logger.error(f"   ├─ Error type: {type(e).__name__}")
                logger.error(f"   ├─ Error code: {e.errno if hasattr(e, 'errno') else 'N/A'}")
                logger.error(f"   ├─ Error message: {str(e) if str(e) else 'No message'}")
                logger.error(f"   ├─ Frame number: {self.frame_count}")
                try:
                    sizes = [self._qsize_safe(q) for q in self.frame_queues]
                    fulls = [self._full_safe(q) for q in self.frame_queues]
                    logger.error(f"   ├─ Queue sizes: {sizes}")
                    logger.error(f"   └─ Queue fulls: {fulls}")
                except:
                    logger.error(f"   └─ Cannot get queue status (queue may be closed)")
                
                # 如果是严重错误，停止服务
                if e.errno in [32, 104]:  # EPIPE, ECONNRESET
                    logger.error("   └─ Critical error detected, stopping stream service...")
                    self.stop()
                    break
                    
            except ValueError as e:
                # 值错误（可能是队列已关闭）
                logger.error(f"❌ ValueError when putting frame to queue!")
                logger.error(f"   ├─ Error message: {str(e) if str(e) else 'No message'}")
                logger.error(f"   ├─ Frame number: {self.frame_count}")
                logger.error(f"   └─ Queue may be closed, stopping stream service...")
                self.stop()
                break
            
            except queue.Full:
                # 队列满：限频打一条 WARNING，避免刷屏
                self._full_drop_count += 1
                now = time.time()
                if now - self._last_full_log_time >= 5.0:
                    logger.warning(
                        f"⚠️ Frame queue full (stream_id={self.stream_id}), "
                        f"dropped {self._full_drop_count} frames in last 5s - consumers may be slow"
                    )
                    self._full_drop_count = 0
                    self._last_full_log_time = now
                
            except Exception as e:
                # 其他未知错误 - 添加更详细的诊断
                logger.error(f"❌ Unexpected error putting frame to queue!")
                logger.error(f"   ├─ Error type: {type(e).__name__}")
                logger.error(f"   ├─ Error message: {str(e) if str(e) else 'No message'}")
                logger.error(f"   ├─ Error repr: {repr(e)}")
                logger.error(f"   ├─ Frame number: {self.frame_count}")
                
                # 尝试获取队列状态以诊断问题
                try:
                    sizes = [self._qsize_safe(q) for q in self.frame_queues]
                    fulls = [self._full_safe(q) for q in self.frame_queues]
                    logger.error(f"   ├─ Queue sizes: {sizes}")
                    logger.error(f"   ├─ Queue fulls: {fulls}")
                    if any(fulls):
                        logger.error(f"   └─ Some queues are full - consumers may be too slow or blocked")
                except Exception as qe:
                    logger.error(f"   └─ Cannot get queue status: {type(qe).__name__}")
                    logger.error(f"      (Queue may be closed or corrupted)")
                
                # 根据错误类型决定是否继续
                if isinstance(e, (BrokenPipeError, ConnectionResetError)):
                    logger.error("🛑 Fatal queue error - stopping stream service")
                    self.stop()
                    break
                else:
                    logger.warning("⚠️ Non-fatal error, continuing (skipped 1 frame)...")
        
        logger.info("✅ Stream service stopped")
    
    def _check_queue_health(self):
        """
        检查队列健康状态
        Returns:
            tuple: (is_healthy: bool, status_message: str)
        """
        try:
            # 尝试获取队列大小
            sizes = [self._qsize_safe(q) for q in self.frame_queues]
            fulls = [self._full_safe(q) for q in self.frame_queues]
            return True, f"Queues healthy (sizes: {sizes}, fulls: {fulls})"

        except Exception as e:
            return False, f"Queue health check failed: {type(e).__name__}: {str(e)}"
    
    def _check_control_commands(self):
        """检查控制命令"""
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                logger.info(f"Received command: {cmd}")
                
                if cmd == 'stop':
                    self.stop()
                elif cmd == 'reconnect':
                    self.reconnect_stream()
                    
        except Exception as e:
            pass
    
    def stop(self):
        """停止流服务"""
        logger.info("Stopping stream service...")
        self.running = False
        if self.decoder is not None:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Stream service stopped")


def run_stream_service(source, frame_queue, control_queue, fps=30, input_mode=0, stream_id=0, stream_count=1):
    """
    进程入口函数
    
    Args:
        source: 视频源（RTSP URL或视频文件路径）
        frame_queue: 输出帧的队列
        control_queue: 接收控制命令的队列
        fps: 目标帧率
        input_mode: 输入模式 (0=RTSP, 1=视频文件)
        stream_id: 流编号（一路=0，多路=0,1,...,N-1，如四路=0/1/2/3，九路=0..8，十六路=0..15）
    """
    service = StreamService(source, frame_queue, control_queue, fps, input_mode, stream_id, stream_count=stream_count)
    service.start()


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream Service (RTSP or Video File)')
    parser.add_argument('--source', type=str, required=True, 
                       help='Video source (RTSP URL or video file path)')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1],
                       help='Input mode: 0=RTSP, 1=Video file')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    args = parser.parse_args()
    
    # 创建队列
    frame_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    # 启动服务
    run_stream_service(args.source, frame_queue, control_queue, args.fps, args.mode)
