"""
第一层：流服务 (Stream Service)
职责：
1. 读取RTSP视频流或本地视频文件
2. 通过共享队列分发帧数据给检测服务
3. 支持启动、停止、重连

本文件已按 `yolov8_bmcv.py` 的逻辑对“解码”部分进行了深度改造：
- 优先使用 Sophon SAIL 的 `sail.Decoder` 在 BM1684X 上进行硬件解码
- 解码得到的 `sail.BMImage` 通过 BMCV 转换为 BGR ndarray，再交给现有检测&展示链路
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
import numpy as np

# 尝试导入 Sophon SAIL，用于硬件编解码
try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except Exception:
    sail = None
    SOPHON_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [StreamService] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StreamService:
    """流服务 - 支持RTSP流和视频文件，优先使用 Sophon SAIL 硬件解码"""
    
    def __init__(self, source, frame_queue, control_queue, fps=30,
                 input_mode=0, stream_id=0, stream_count=1, device_id=0):
        """
        Args:
            source: 视频源（RTSP URL或视频文件路径）
            frame_queue: multiprocessing.Queue 或 list[Queue]，用于输出帧数据（可同时投递到多个消费者队列）
            control_queue: multiprocessing.Queue，用于接收控制命令
            fps: 目标帧率
            input_mode: 输入模式 (0=RTSP, 1=视频文件)
            stream_id: 流编号（多路时 0,1,2,3）
            stream_count: 总路数（用于低延迟策略判断）
            device_id: BM1684X 设备号，用于 Sophon SAIL 编解码
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
        self.device_id = int(device_id)
        
        self.running = False
        # OpenCV 后端（仅在未启用 Sophon 或回退时使用）
        self.cap = None
        # Sophon SAIL 相关句柄
        self.handle = None
        self.bmcv = None
        self.decoder = None  # sail.Decoder
        self._first_frame_numpy = None  # Sophon 解码读取的首帧（用于获取宽高的同时复用）
        self.frame_count = 0
        
        self.is_video_file = (input_mode == 1)
        self.video_loop = True
        self.total_frames = 0
        self._full_drop_count = 0
        self._last_full_log_time = 0.0

        # 是否启用 Sophon 硬件解码：SAIL 可用时默认开启，失败时自动回退到 OpenCV
        self.use_sophon_decoder = SOPHON_AVAILABLE

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
        """连接视频源（RTSP流或视频文件），优先使用 Sophon SAIL 解码"""
        # 优先尝试 Sophon SAIL 硬件解码
        if self.use_sophon_decoder and SOPHON_AVAILABLE:
            ok = self._connect_sophon_decoder()
            if ok:
                return True
            # 若 Sophon 打开失败，自动回退到 OpenCV
            logger.warning("⚠️ Sophon SAIL decoder open failed, falling back to OpenCV VideoCapture")
            self.use_sophon_decoder = False
        
        # 回退：使用 OpenCV 软解码
        if self.is_video_file:
            return self._connect_video_file()
        else:
            return self._connect_rtsp_stream()

    def _connect_sophon_decoder(self):
        """
        使用 Sophon SAIL 的 `sail.Decoder` 连接视频源（遵循 yolov8_bmcv.py 的解码逻辑）。
        解码得到 BMImage 后，通过 BMCV 转成 BGR ndarray，整体链路仍然兼容现有检测与展示代码。
        """
        if not SOPHON_AVAILABLE:
            return False

        try:
            logger.info(f"🎥 [Stream {self.stream_id}] Connecting via Sophon sail.Decoder (device={self.device_id}): {self.source}")
            self.handle = sail.Handle(self.device_id)
            self.bmcv = sail.Bmcv(self.handle)
            self.decoder = sail.Decoder(self.source, True, self.device_id)
        except Exception as e:
            logger.error(f"❌ Failed to create Sophon decoder: {e}")
            self.decoder = None
            self.handle = None
            self.bmcv = None
            return False

        if not self.decoder.is_opened():
            logger.error("❌ Sophon decoder can not open the video/stream")
            self.decoder = None
            return False

        # 读取一帧用于确认分辨率，并作为后续 run() 循环的第一帧
        bmimg = sail.BMImage()
        ret = self.decoder.read(self.handle, bmimg)
        if ret != 0:
            logger.error("❌ Sophon decoder first read failed")
            return False

        try:
            frame = self._bmimage_to_bgr_numpy(bmimg)
        except Exception as e:
            logger.error(f"❌ Failed to convert BMImage to numpy BGR: {e}")
            return False

        self._first_frame_numpy = frame
        h, w = frame.shape[:2]
        logger.info(f"✅ [Stream {self.stream_id}] Sophon decoder connected: {w}x{h}")
        return True

    def _bmimage_to_bgr_numpy(self, bmimg):
        """
        将 BMImage 转换为 BGR ndarray。
        步骤：
        1. 统一转换为 FORMAT_BGR_PLANAR
        2. 使用 bm_image_to_tensor 得到 Tensor，再 asnumpy 得到 NCHW
        3. 转置为 HWC，得到 OpenCV 兼容的 BGR 图像
        """
        if self.bmcv is None:
            self.bmcv = sail.Bmcv(self.handle)

        h = bmimg.height()
        w = bmimg.width()

        # 统一转换为 BGR planar
        bgr_planar = sail.BMImage(
            self.handle,
            h,
            w,
            sail.Format.FORMAT_BGR_PLANAR,
            sail.DATA_TYPE_EXT_1N_BYTE,
        )
        self.bmcv.convert_format(bmimg, bgr_planar)

        # BMImage → Tensor → numpy，形状为 [N, C, H, W]
        tensor = self.bmcv.bm_image_to_tensor(bgr_planar)
        np_tensor = tensor.asnumpy()  # [N, C, H, W]
        if np_tensor.ndim == 4:
            np_chw = np_tensor[0]  # [C, H, W]
        elif np_tensor.ndim == 3:
            np_chw = np_tensor
        else:
            raise RuntimeError(f"Unexpected tensor shape from BMImage: {np_tensor.shape}")

        # CHW → HWC（BGR）
        np_hwc = np.transpose(np_chw, (1, 2, 0)).astype('uint8', copy=False)
        return np_hwc
    
    def _connect_rtsp_stream(self):
        """连接RTSP流（OpenCV 软解码分支）"""
        logger.info(f"🎥 [Stream {self.stream_id}] Connecting to RTSP (OpenCV): {self.source[:60]}...")

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
            return False
        
        # 获取流信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"✅ [Stream {self.stream_id}] RTSP connected: {width}x{height} @ {actual_fps}fps")
        return True
    
    def _connect_video_file(self):
        """连接视频文件（OpenCV 软解码分支）"""
        logger.info(f"📹 [Stream {self.stream_id}] Opening video file: {self.source}")
        
        # 检查文件是否存在
        if not os.path.exists(self.source):
            logger.error(f"❌ Video file not found: {self.source}")
            return False
        
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
        """重新连接流（根据当前解码后端选择 Sophon 或 OpenCV）"""
        logger.warning("⚠️ Attempting to reconnect...")

        # 先清理旧的解码资源
        if self.decoder:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._first_frame_numpy = None

        time.sleep(2)  # 等待2秒后重连
        return self.connect_stream()
    
    def start(self):
        """启动流服务"""
        mode_name = "VIDEO FILE" if self.is_video_file else "RTSP STREAM"
        logger.info(f"🚀 Stream service starting ({mode_name} mode)...")
        self.setup_signal_handlers()
        
        # 🆕 启动前检查队列健康状态
        is_healthy, status_msg = self._check_queue_health()
        if is_healthy:
            logger.info(f"✅ Queue health check passed: {status_msg}")
        else:
            logger.error(f"❌ Queue health check failed: {status_msg}")
            logger.error("   └─ Cannot start stream service with unhealthy queue")
            return
        
        if not self.connect_stream():
            logger.error("❌ Failed to start stream service")
            return
        
        self.running = True
        self.run()
    
    def run(self):
        """主循环：读取帧并分发（低延迟策略：先取流再限速，RTSP 丢旧取新，解码逻辑对齐 yolov8_bmcv.py）"""
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
            # 检查控制命令
            self._check_control_commands()

            # 低延迟策略：先读取再按需 sleep，保证入队帧尽量是“当前最新”
            # 解码逻辑：
            # - 若启用 Sophon：通过 sail.Decoder 解码为 BMImage，再转成 BGR ndarray
            # - 否则：保留原有 OpenCV 软解码逻辑
            if self.use_sophon_decoder and self.decoder is not None:
                # Sophon 硬件解码分支
                if self._first_frame_numpy is not None:
                    # 首帧已经在 _connect_sophon_decoder 中解码并转换，这里直接复用
                    frame = self._first_frame_numpy
                    ret = True
                    self._first_frame_numpy = None
                else:
                    bmimg = sail.BMImage()
                    try:
                        ret_code = self.decoder.read(self.handle, bmimg)
                    except Exception as e:
                        logger.error(f"❌ Sophon decoder read error: {e}")
                        ret_code = -1
                    if ret_code != 0:
                        ret = False
                        frame = None
                    else:
                        try:
                            frame = self._bmimage_to_bgr_numpy(bmimg)
                            ret = True
                        except Exception as e:
                            logger.error(f"❌ Failed to convert BMImage to numpy BGR: {e}")
                            ret = False
                            frame = None
            else:
                # OpenCV 软解码分支（保留原有低延迟逻辑）
                if not self.is_video_file:
                    try:
                        # 当队列已有帧（消费者稍慢）时，多 grab 几次跳到最新帧，避免堆积旧帧
                        try:
                            qsize = max(self._qsize_safe(q) for q in self.frame_queues) if self.frame_queues else 0
                        except Exception:
                            qsize = 0
                        skip = 1  # 队列空时只丢 1 帧，尽量低延迟
                        if qsize > 0 or any(self._full_safe(q) for q in self.frame_queues):
                            skip = min(8, 1 + qsize)  # 有积压时多丢几帧以追上实时
                        for _ in range(skip):
                            self.cap.grab()
                        ret, frame = self.cap.retrieve()
                        if not ret:
                            ret, frame = self.cap.read()
                    except Exception:
                        ret, frame = self.cap.read()
                else:
                    ret, frame = self.cap.read()

            # 读取后再做帧率限制，避免“等完 sleep 才读”带来的额外延迟
            now = time.time()
            elapsed = now - last_tick
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            last_tick = time.time()
            
            if not ret:
                # 🆕 视频文件读取结束
                if self.is_video_file:
                    if self.video_loop and not self.use_sophon_decoder:
                        # 仅在 OpenCV 模式下支持 loop；Sophon Decoder 暂不做 seek，避免引入未知 API
                        logger.info(f"🔄 Video file ended, restarting from beginning (OpenCV)...")
                        if self.cap is not None:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头
                        self.frame_count = 0
                        continue
                    else:
                        logger.info("✅ Video file playback completed")
                        self.stop()
                        break
                
                # RTSP流或其他模式读取失败，尝试重连
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
        
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.decoder:
            try:
                self.decoder.release()
            except Exception:
                pass
            self.decoder = None
        self.handle = None
        self.bmcv = None
        
        logger.info("Stream service stopped")


def run_stream_service(source, frame_queue, control_queue, fps=30,
                       input_mode=0, stream_id=0, stream_count=1, device_id=0):
    """
    进程入口函数
    
    Args:
        source: 视频源（RTSP URL或视频文件路径）
        frame_queue: 输出帧的队列
        control_queue: 接收控制命令的队列
        fps: 目标帧率
        input_mode: 输入模式 (0=RTSP, 1=视频文件)
        stream_id: 流编号（一路=0，多路=0,1,...,N-1，如四路=0/1/2/3，九路=0..8，十六路=0..15）
        stream_count: 总路数
        device_id: BM1684X 设备号（供 Sophon SAIL 硬件解码使用）
    """
    service = StreamService(
        source,
        frame_queue,
        control_queue,
        fps,
        input_mode,
        stream_id,
        stream_count=stream_count,
        device_id=device_id,
    )
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