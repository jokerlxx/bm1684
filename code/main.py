"""
主控程序 (Main Scheduler)
"""

import multiprocessing as mp
import threading
import time
from datetime import datetime, timedelta
import logging
import sys
import signal
from pathlib import Path
import json
import pytz

# 导入服务（视频流接入 / 检测引擎 / 展示服务）
from ingestion import run_stream_service
from detection import (
    run_fall_detector,
    run_ventilator_detector,
    run_fight_detector,
    run_crowd_detector,
    run_helmet_detector,
    run_window_door_inside_detector,
    run_window_door_outside_detector,
)
from core import run_display_service
from core.encoder_service import run_encoder_service

logger = logging.getLogger('StreamService')


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')


# ==================== 加载配置文件 ====================
def load_config(config_file='config_bm1684x.json'):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Using default configuration")
        return {
            'fps': 25,
            'models': {
                "fall_detection": "/home/admin/zsh/final/final/model/falldetection_BIG_fp32_1b.bmodel",
                "ventilator_equipment": "/home/admin/zsh/final/final/model/huxiji_fp32_1b.bmodel",
                "ventilator_helmet": "/home/admin/zsh/final/final/model/mao_fp32_1b.bmodel",
                "fight_detection": "/home/admin/zsh/final/final/model/fight_runv15_BIG_fp32_1b.bmodel",
                "crowd_person": "/home/admin/zsh/final/final/model/mao_fp32_1b.bmodel",
                "helmet_detection": "/home/admin/zsh/final/final/model/mao_fp32_1b.bmodel",
                "window_door_detection": "/home/admin/zsh/final/final/model/windoes_BIG_fp32_1b.bmodel"
            },
            'output': {
                'video_output_dir': './alarm_videos',
                'display_port': 5000,
                'font_path': 'simhei.ttf',
                'hls_output_dir': './hls_output',
                'hls_time': 1,
                'hls_list_size': 5
            },
            'queue_sizes': {
                'frame_queue': 60,
                'result_queue': 10,
                'display_queue': 5,
                'encoder_queue': 5
            },
            'bm1684x': {
                'device_id': 0,
                'enable_sophon': True
            }
        }


# 加载配置
CONFIG = load_config()

# 全局调度器实例
scheduler = None


# ==================== 主调度器类 ====================
class MainScheduler:
    """主调度器 - 管理所有服务进程"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        self.processes = {}
        # 流服务控制队列：每路独立，避免“单队列无法广播 stop”以及 stop 残留影响下一次启动
        self.stream_control_queues = [mp.Queue() for _ in range(9)]
        
        # 采集输出队列分离：一个给展示/编码（保证预览 FPS），一个给检测器（避免抢帧）
        self.display_frame_queue = mp.Queue(maxsize=config['queue_sizes']['frame_queue'])
        self.detect_frame_queue = mp.Queue(maxsize=config['queue_sizes']['frame_queue'])
        self.result_queues = {
            'fall': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'ventilator': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'fight': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'crowd': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'helmet': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'window_door_inside': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'window_door_outside': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
        }
        self.output_queue = mp.Queue(maxsize=config['queue_sizes']['display_queue'])
        # 已标注帧队列：展示服务写入，编码服务读取，用于 HLS 低延迟流（解码-处理-编码流水线）
        self.encoder_queue = mp.Queue(maxsize=config['queue_sizes'].get('encoder_queue', 5))
        # 告警事件队列：展示服务写入，Web 层读取并推送至前端（SSE/WebSocket）
        self.alert_queue = mp.Queue(maxsize=200)
        
        self.control_queues = {
            'fall': mp.Queue(),
            'ventilator': mp.Queue(),
            'fight': mp.Queue(),
            'crowd': mp.Queue(),
            'helmet': mp.Queue(),
            'window_door_inside': mp.Queue(),
            'window_door_outside': mp.Queue(),
            'display': mp.Queue(),
            'encoder': mp.Queue()
        }
        
        self.detector_running = {
            'fall': False,
            'ventilator': False,
            'fight': False,
            'crowd': False,
            'helmet': False,
            'window_door_inside': False,
            'window_door_outside': False,
        }
        # 任务级启停：task_id -> bool；每个检测器当前处理的 stream_id 集合
        self.task_running = {}
        self.detector_streams = {}  # detector_name -> set(stream_id)
        # 防止 reload_streams 与 stop 并发执行导致 dictionary changed size during iteration
        self._reload_stop_lock = threading.Lock()
        
        logger.info("MainScheduler initialized")

    def _drain_queue(self, q, max_items=10000):
        """清空 multiprocessing 队列中的残留消息（避免 stop 等旧命令影响下一次启动）。"""
        drained = 0
        try:
            while drained < max_items:
                q.get_nowait()
                drained += 1
        except Exception:
            pass
        return drained
    
    def start(self):
        """启动核心服务"""
        logger.info("=" * 60)
        logger.info("Starting Multi-Task Detection System")
        logger.info("=" * 60)
        
        try:
            import os
            # 优先从「视频流服务」配置 video_streams 读取；无则回退到旧字段（兼容）
            video_sources = []
            stream_count = 1
            input_mode = 0
            streams = self.config.get('video_streams')
            if isinstance(streams, list) and len(streams) > 0:
                video_sources = [str(s.get('ip', '')).strip() for s in streams[:9] if isinstance(s, dict)]
                if not video_sources:
                    video_sources = ['']
                stream_count = min(9, max(1, len(video_sources)))
                logger.info(f"📡 Video sources from config video_streams: {stream_count} stream(s)")
                for i, src in enumerate(video_sources):
                    logger.info(f"   Stream {i}: {src[:70]}..." if src else f"   Stream {i}: (empty)")
            else:
                # 无 video_streams 时使用单路空源，避免启动报错；用户需在「视频流」页添加并保存
                video_sources = ['']
                stream_count = 1
                logger.info("📡 No video_streams in config; using 1 empty stream (add streams in 视频流 page and save)")
            if not video_sources:
                video_sources = ['']
                stream_count = 1

            # 启动前清空控制队列，避免上一次 terminate 后残留 stop 命令导致新进程立刻退出
            try:
                for i in range(stream_count):
                    self._drain_queue(self.stream_control_queues[i])
                self._drain_queue(self.control_queues['display'])
                self._drain_queue(self.control_queues['encoder'])
            except Exception:
                pass
            
            # 1. 启动流服务（一路一进程，多路多进程）
            logger.info("Starting Stream Service(s)...")
            if stream_count == 1:
                self.processes['stream'] = mp.Process(
                    target=run_stream_service,
                    args=(
                        video_sources[0],
                        [self.display_frame_queue, self.detect_frame_queue],
                        self.stream_control_queues[0],
                        self.config['fps'],
                        input_mode,
                        0,
                        stream_count
                    ),
                    daemon=True
                )
                self.processes['stream'].start()
            else:
                for i in range(stream_count):
                    key = 'stream' if i == 0 else f'stream_{i}'
                    self.processes[key] = mp.Process(
                        target=run_stream_service,
                        args=(
                            video_sources[i],
                            [self.display_frame_queue, self.detect_frame_queue],
                            self.stream_control_queues[i],
                            self.config['fps'],
                            input_mode,
                            i,
                            stream_count
                        ),
                        daemon=True
                    )
                    self.processes[key].start()
                    # 避免同一时刻对 NVR 发起过多 RTSP SETUP，降低 500 概率
                    time.sleep(0.15)
            time.sleep(2)
            
            # 2. 启动显示服务（传入 stream_count、alert_queue、font_path、encoder_queue 用于实时告警与 HLS 流水线）
            logger.info("Starting Display Service...")
            out_cfg = self.config.get('output', {})
            font_path = out_cfg.get('font_path', 'simhei.ttf')
            alert_box_mode = str(out_cfg.get('alert_box_mode', 'blink')).strip().lower()
            if alert_box_mode not in ('blink', 'follow'):
                alert_box_mode = 'blink'
            hls_dir = out_cfg.get('hls_output_dir', './hls_output')
            self.processes['display'] = mp.Process(
                target=run_display_service,
                args=(
                    self.display_frame_queue,
                    self.result_queues,
                    self.output_queue,
                    self.control_queues['display'],
                    out_cfg['video_output_dir'],
                    self.config['fps'],
                    stream_count,
                    self.alert_queue,
                    font_path,
                    self.encoder_queue,
                    alert_box_mode,
                ),
                daemon=True
            )
            self.processes['display'].start()
            
            # 3. 启动编码服务（解码-处理-编码流水线：将已标注帧编码为 HLS 流供前端低延迟播放）
            logger.info("Starting Encoder Service (HLS)...")
            hls_time = out_cfg.get('hls_time', 1)
            hls_list_size = out_cfg.get('hls_list_size', 5)
            self.processes['encoder'] = mp.Process(
                target=run_encoder_service,
                args=(
                    self.encoder_queue,
                    self.control_queues['encoder'],
                    hls_dir,
                    self.config['fps'],
                    hls_time,
                    hls_list_size,
                ),
                daemon=True
            )
            self.processes['encoder'].start()
            
            # 检测器不再随系统启动自动开启，由「任务管理」页对每个任务单独启动/停止
            logger.info("   - Detectors: use Task Management to start/stop per task")
            
            self.running = True
            logger.info("=" * 60)
            logger.info("✅ Core services started successfully!")
            logger.info(f"   - Input Mode: {'RTSP Stream' if input_mode == 0 else 'Video File'}")
            logger.info(f"   - Stream count: {stream_count}")
            logger.info("   - Stream Service(s): Running")
            logger.info("   - Display Service: Running")
            logger.info("   - Encoder Service (HLS): Running")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            self.stop()
            raise
    
    def start_detector(self, detector_name, enabled_streams=None):
        """启动单个检测器。enabled_streams: 仅处理这些 stream_id（0-based）的帧；None 表示处理全部（兼容旧开关）。"""
        if detector_name in self.processes and self.processes[detector_name].is_alive():
            logger.warning(f"{detector_name} detector is already running")
            return False
        
        logger.info(f"Starting {detector_name} detector...")
        
        # 清理旧的结果队列，避免积压
        try:
            while not self.result_queues[detector_name].empty():
                self.result_queues[detector_name].get_nowait()
            logger.info(f"Cleared result queue for {detector_name}")
        except:
            pass

        # 清空该检测器控制队列（上次 stop+terminate 可能留下未消费的 stop）
        try:
            self._drain_queue(self.control_queues[detector_name])
        except Exception:
            pass
        
        try:
            if detector_name == 'fall':
                process = mp.Process(
                    target=run_fall_detector,
                    args=(
                        self.config['models']['fall_detection'],
                        self.detect_frame_queue,
                        self.result_queues['fall'],
                        self.control_queues['fall'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'ventilator':
                process = mp.Process(
                    target=run_ventilator_detector,
                    args=(
                        self.config['models']['ventilator_equipment'],
                        self.config['models']['ventilator_helmet'],
                        self.detect_frame_queue,
                        self.result_queues['ventilator'],
                        self.control_queues['ventilator'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'fight':
                process = mp.Process(
                    target=run_fight_detector,
                    args=(
                        self.config['models']['fight_detection'],
                        self.detect_frame_queue,
                        self.result_queues['fight'],
                        self.control_queues['fight'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'crowd':
                process = mp.Process(
                    target=run_crowd_detector,
                    args=(
                        self.config['models']['crowd_person'],
                        self.detect_frame_queue,
                        self.result_queues['crowd'],
                        self.control_queues['crowd'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'helmet':
                process = mp.Process(
                    target=run_helmet_detector,
                    args=(
                        self.config['models']['helmet_detection'],
                        self.detect_frame_queue,
                        self.result_queues['helmet'],
                        self.control_queues['helmet'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'window_door_inside':
                model_path = (self.config['models'].get('window_door_inside')
                             or self.config['models'].get('window_door_detection'))
                if not model_path:
                    logger.error("门窗仓内检测器需配置 models.window_door_inside 或 models.window_door_detection")
                    return False
                process = mp.Process(
                    target=run_window_door_inside_detector,
                    args=(
                        model_path,
                        self.detect_frame_queue,
                        self.result_queues['window_door_inside'],
                        self.control_queues['window_door_inside'],
                        self.config
                    ),
                    daemon=True
                )
            elif detector_name == 'window_door_outside':
                model_path = (self.config['models'].get('window_door_outside')
                             or self.config['models'].get('window_door_detection'))
                if not model_path:
                    logger.error("门窗仓外检测器需配置 models.window_door_outside 或 models.window_door_detection")
                    return False
                process = mp.Process(
                    target=run_window_door_outside_detector,
                    args=(
                        model_path,
                        self.detect_frame_queue,
                        self.result_queues['window_door_outside'],
                        self.control_queues['window_door_outside'],
                        self.config
                    ),
                    daemon=True
                )
            else:
                logger.error(f"Unknown detector: {detector_name}")
                return False
            
            process.start()
            self.processes[detector_name] = process
            self.detector_running[detector_name] = True
            
            # 等待进程初始化
            time.sleep(1.0)
            if enabled_streams is not None and len(enabled_streams) > 0:
                try:
                    self.control_queues[detector_name].put({'cmd': 'set_streams', 'stream_ids': list(enabled_streams)})
                except Exception:
                    pass
            self.control_queues[detector_name].put('enable')
            
            # 再等待一下确保启动完成
            time.sleep(0.5)
            
            logger.info(f"✅ {detector_name} detector started (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {detector_name} detector: {e}")
            return False
    
    def stop_detector(self, detector_name):
        """停止单个检测器"""
        if detector_name not in self.processes or not self.processes[detector_name].is_alive():
            logger.warning(f"{detector_name} detector is not running")
            return False
        
        logger.info(f"Stopping {detector_name} detector...")
        
        try:
            # 🔧 新增：在停止前先发送一个disabled结果，通知display service清空渲染
            try:
                disabled_result = {
                    'detector_type': detector_name,
                    'enabled': False,
                    'frame': None,
                    'timestamp': datetime.now(),
                    'detections': []
                }
                # 清空旧结果
                while not self.result_queues[detector_name].empty():
                    self.result_queues[detector_name].get_nowait()
                # 发送disabled通知（连续发送3次确保被接收）
                for _ in range(3):
                    self.result_queues[detector_name].put(disabled_result, block=False)
                logger.info(f"Sent disabled notification for {detector_name} (x3)")
                time.sleep(0.3)  # 给display service充足时间处理
            except Exception as e:
                logger.warning(f"Failed to send disabled notification: {e}")
            
            # 发送停止命令
            self.control_queues[detector_name].put('stop')
            time.sleep(0.5)
            
            # 终止进程
            process = self.processes[detector_name]
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
                
                if process.is_alive():
                    process.kill()
                    process.join()
            
            # 🔧 修改：不再清空result_queue，避免清除disabled通知
            # 让display service自己处理队列中的消息
            
            self.detector_running[detector_name] = False
            # 清空残留控制命令，避免下次启动立刻读到 stop
            try:
                self._drain_queue(self.control_queues[detector_name])
            except Exception:
                pass
            logger.info(f"✅ {detector_name} detector stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {detector_name} detector: {e}")
            return False
    
    def get_task_running_states(self):
        """返回 task_id -> bool，供前端展示每任务运行状态"""
        return dict(self.task_running)
    
    def start_task(self, task_id):
        """启动单个任务：为该任务绑定的通道启用对应检测器（或合并到已有检测器的 stream 集合）"""
        tasks = self.config.get('tasks') or []
        task = next((t for t in tasks if isinstance(t, dict) and str(t.get('id')) == str(task_id)), None)
        if not task:
            return {'status': 'error', 'message': '任务不存在'}
        stream_index = int(task.get('stream_index', 1))
        if stream_index < 1 or stream_index > 9:
            return {'status': 'error', 'message': '无效的通道'}
        stream_id_0 = stream_index - 1
        detectors = [d for d in (task.get('detectors') or []) if d in VALID_DETECTORS]
        if not detectors:
            return {'status': 'error', 'message': '任务未配置检测器'}
        for det in detectors:
            self.detector_streams.setdefault(det, set()).add(stream_id_0)
            streams = self.detector_streams[det]
            if det in self.processes and self.processes[det].is_alive():
                try:
                    self.control_queues[det].put({'cmd': 'set_streams', 'stream_ids': list(streams)})
                except Exception:
                    pass
            else:
                self.start_detector(det, enabled_streams=streams)
        self.task_running[str(task_id)] = True
        logger.info("Task started: %s (stream %s, detectors: %s)", task_id, stream_index, detectors)
        return {'status': 'success', 'message': '任务已启动'}
    
    def stop_task(self, task_id):
        """停止单个任务：从对应检测器中移除该任务的通道"""
        tasks = self.config.get('tasks') or []
        task = next((t for t in tasks if isinstance(t, dict) and str(t.get('id')) == str(task_id)), None)
        if not task:
            return {'status': 'error', 'message': '任务不存在'}
        stream_index = int(task.get('stream_index', 1))
        stream_id_0 = stream_index - 1 if 1 <= stream_index <= 9 else 0
        detectors = [d for d in (task.get('detectors') or []) if d in VALID_DETECTORS]
        for det in detectors:
            if det in self.detector_streams:
                self.detector_streams[det].discard(stream_id_0)
                streams = self.detector_streams[det]
                if not streams:
                    self.detector_streams.pop(det, None)
                    self.stop_detector(det)
                elif det in self.processes and self.processes[det].is_alive():
                    try:
                        self.control_queues[det].put({'cmd': 'set_streams', 'stream_ids': list(streams)})
                    except Exception:
                        pass
        self.task_running[str(task_id)] = False
        logger.info("Task stopped: %s", task_id)
        return {'status': 'success', 'message': '任务已停止'}
    
    def reload_streams(self, config=None):
        """热重载视频流：重启流服务与显示服务（用于保存视频流配置后无需整系统重启）。加锁避免与 stop 并发。"""
        with self._reload_stop_lock:
            if not self.running:
                if config is not None:
                    self.config = config
                return
            if config is not None:
                self.config = config
            streams = self.config.get('video_streams')
            if not isinstance(streams, list) or len(streams) == 0:
                video_sources = ['']
                stream_count = 1
            else:
                video_sources = [str(s.get('ip', '')).strip() for s in streams[:9] if isinstance(s, dict)]
                if not video_sources:
                    video_sources = ['']
                stream_count = min(9, max(1, len(video_sources)))
            input_mode = 0
            # 停止流与显示、编码
            for key in list(self.processes.keys()):
                if key == 'stream' or key.startswith('stream_') or key == 'display' or key == 'encoder':
                    p = self.processes.get(key)
                    if p and p.is_alive():
                        try:
                            if key == 'stream' or key.startswith('stream_'):
                                sid = 0 if key == 'stream' else int(key.split('_', 1)[1])
                                self.stream_control_queues[sid].put('stop')
                            elif key == 'display':
                                self.control_queues['display'].put('stop')
                            elif key == 'encoder':
                                self.control_queues['encoder'].put('stop')
                        except Exception:
                            pass
                        try:
                            p.terminate()
                            p.join(timeout=3)
                            if p.is_alive():
                                p.kill()
                                p.join()
                        except Exception:
                            pass
                    self.processes.pop(key, None)
            time.sleep(1)

            # 清空控制队列与帧队列，避免残留 stop 命令和旧帧影响新进程
            try:
                for i in range(9):
                    self._drain_queue(self.stream_control_queues[i])
                self._drain_queue(self.control_queues['display'])
                self._drain_queue(self.control_queues['encoder'])
                self._drain_queue(self.display_frame_queue)
                self._drain_queue(self.detect_frame_queue)
                self._drain_queue(self.output_queue)
                self._drain_queue(self.encoder_queue)
            except Exception:
                pass

            # 重新启动流
            if stream_count == 1:
                self.processes['stream'] = mp.Process(
                    target=run_stream_service,
                    args=(
                        video_sources[0],
                        [self.display_frame_queue, self.detect_frame_queue],
                        self.stream_control_queues[0],
                        self.config['fps'],
                        input_mode,
                        0,
                        stream_count
                    ),
                    daemon=True
                )
                self.processes['stream'].start()
            else:
                for i in range(stream_count):
                    key = 'stream' if i == 0 else f'stream_{i}'
                    self.processes[key] = mp.Process(
                        target=run_stream_service,
                        args=(
                            video_sources[i],
                            [self.display_frame_queue, self.detect_frame_queue],
                            self.stream_control_queues[i],
                            self.config['fps'],
                            input_mode,
                            i,
                            stream_count
                        ),
                        daemon=True
                    )
                    self.processes[key].start()
                    time.sleep(0.15)
            time.sleep(1)
            font_path = self.config.get('output', {}).get('font_path', 'simhei.ttf')
            out_cfg = self.config.get('output', {})
            self.processes['display'] = mp.Process(
                target=run_display_service,
                args=(
                    self.display_frame_queue,
                    self.result_queues,
                    self.output_queue,
                    self.control_queues['display'],
                    self.config['output']['video_output_dir'],
                    self.config['fps'],
                    stream_count,
                    self.alert_queue,
                    font_path,
                    self.encoder_queue,
                ),
                daemon=True
            )
            self.processes['display'].start()
            hls_dir = out_cfg.get('hls_output_dir', './hls_output')
            hls_time = out_cfg.get('hls_time', 1)
            hls_list_size = out_cfg.get('hls_list_size', 5)
            self.processes['encoder'] = mp.Process(
                target=run_encoder_service,
                args=(
                    self.encoder_queue,
                    self.control_queues['encoder'],
                    hls_dir,
                    self.config['fps'],
                    hls_time,
                    hls_list_size,
                ),
                daemon=True
            )
            self.processes['encoder'].start()
            logger.info("Video streams reloaded: %s stream(s)", stream_count)
    
    def stop(self):
        """停止所有服务。使用锁与快照迭代，避免与 reload_streams 并发时 dictionary changed size during iteration。"""
        with self._reload_stop_lock:
            logger.info("Stopping all services...")
            self.running = False
            
            # 逐路广播 stop（multiprocessing.Queue 是“工作队列”语义，不能用单队列广播）
            for k in list(self.processes.keys()):
                if k == 'stream' or k.startswith('stream_'):
                    try:
                        sid = 0 if k == 'stream' else int(k.split('_', 1)[1])
                        self.stream_control_queues[sid].put('stop')
                    except Exception:
                        pass

            for name, queue in list(self.control_queues.items()):
                try:
                    queue.put('stop')
                except Exception:
                    pass
            
            time.sleep(1)
            
            # 使用快照迭代，避免 dict 在迭代中被 reload_streams 等修改
            for name, process in list(self.processes.items()):
                if process and process.is_alive():
                    logger.info(f"Terminating {name} service...")
                    try:
                        process.terminate()
                        process.join(timeout=3)
                        if process.is_alive():
                            process.kill()
                            process.join()
                    except Exception as e:
                        logger.warning("Error terminating %s: %s", name, e)
            
            self.processes.clear()
            
            for key in list(self.detector_running.keys()):
                self.detector_running[key] = False
            
            logger.info("✅ All services stopped")
    
    def get_status(self):
        """获取所有服务状态"""
        status = {
            'system_running': self.running,
            'detectors': {}
        }
        
        for name in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door_inside', 'window_door_outside']:
            is_running = (name in self.processes and 
                         self.processes[name].is_alive() and 
                         self.detector_running[name])
            
            status['detectors'][name] = {
                'running': is_running,
                'pid': self.processes[name].pid if is_running else None
            }
        
        return status


# ==================== API 实现（供后端路由调用） ====================
def start_system_impl():
    """启动系统 - 返回可 jsonify 的字典"""
    global scheduler
    try:
        if scheduler is not None and scheduler.running:
            return {'status': 'error', 'message': '系统已在运行中'}
        logger.info("Starting system...")
        scheduler = MainScheduler(CONFIG)
        scheduler.start()
        return {'status': 'success', 'message': '系统启动成功'}
    except Exception as e:
        logger.error("Failed to start system: %s", e)
        return {'status': 'error', 'message': str(e)}


def stop_system_impl():
    """停止系统 - 返回可 jsonify 的字典"""
    global scheduler
    try:
        if scheduler is None:
            return {'status': 'error', 'message': '系统未运行'}
        logger.info("Stopping system...")
        scheduler.stop()
        scheduler = None
        return {'status': 'success', 'message': '系统停止成功'}
    except Exception as e:
        logger.error("Failed to stop system: %s", e)
        return {'status': 'error', 'message': str(e)}


def toggle_detector_impl(data):
    """切换检测器 - 接收 request 数据，返回可 jsonify 的字典"""
    global scheduler
    try:
        if scheduler is None or not scheduler.running:
            return {'status': 'error', 'message': '系统未运行，请先启动系统'}
        detector = data.get('detector')
        enabled = data.get('enabled', True)
        if detector not in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door_inside', 'window_door_outside']:
            return {'status': 'error', 'message': f'未知的检测器: {detector}'}
        if enabled:
            success = scheduler.start_detector(detector)
            action = '启动'
        else:
            success = scheduler.stop_detector(detector)
            action = '停止'
        if success:
            logger.info("✅ %s detector %s成功 (手动)", detector, action)
            return {'status': 'success', 'message': f'{detector} detector {action}成功'}
        return {'status': 'error', 'message': f'{detector} detector {action}失败'}
    except Exception as e:
        logger.error("Failed to toggle detector: %s", e)
        return {'status': 'error', 'message': str(e)}


def get_status_impl():
    """获取系统状态 - 返回可 jsonify 的字典"""
    global scheduler
    if scheduler is None:
        return {'system_running': False, 'detectors': {}, 'streams': []}
    # 在原有状态基础上补充视频流进程状态
    base = scheduler.get_status()
    streams = []
    try:
        for name, proc in scheduler.processes.items():
            if name == 'stream' or name.startswith('stream_'):
                running = proc.is_alive()
                streams.append({
                    'name': name,
                    'running': running,
                    'pid': proc.pid if running else None,
                })
    except Exception:
        pass
    base['streams'] = streams
    return base


def get_alerts_history_impl():
    """历史告警文件列表 - 用于历史数据回溯 API（含 category）；列表前自动删除超过 7 天的文件"""
    try:
        from storage import list_alarm_files, get_alarm_output_dir, cleanup_old_alarm_files
        from shutil import disk_usage
        output_dir = get_alarm_output_dir(CONFIG.get('output'))
        cleanup_old_alarm_files(output_dir, max_age_days=7)
        items = list_alarm_files(output_dir, limit=500)
        # 统计总大小与磁盘空间
        total_size = sum((it.get("size") or 0) for it in items)
        disk_total = None
        disk_used = None
        disk_free = None
        try:
            usage = disk_usage(output_dir)
            disk_total = usage.total
            disk_used = usage.used
            disk_free = usage.free
        except Exception:
            pass
        return {
            'status': 'success',
            'items': items,
            'output_dir': output_dir,
            'total_size': total_size,
            'disk_total': disk_total,
            'disk_used': disk_used,
            'disk_free': disk_free,
        }
    except Exception as e:
        logger.error("Failed to get alerts history: %s", e)
        return {'status': 'error', 'message': str(e), 'items': [], 'output_dir': ''}


def get_alerts_cleanup_impl(request=None):
    """清除超过 7 天的告警文件；可选 body.max_days 覆盖默认 7 天"""
    try:
        from storage import get_alarm_output_dir, cleanup_old_alarm_files
        output_dir = get_alarm_output_dir(CONFIG.get('output'))
        max_days = 7
        if request:
            d = request.get_json(silent=True)
            if isinstance(d, dict) and isinstance(d.get('max_days'), (int, float)) and 1 <= d['max_days'] <= 365:
                max_days = int(d['max_days'])
        deleted = cleanup_old_alarm_files(output_dir, max_age_days=max_days)
        return {'status': 'success', 'message': f'已删除 {deleted} 个过期文件', 'deleted': deleted}
    except Exception as e:
        logger.error("Failed to cleanup old alerts: %s", e)
        return {'status': 'error', 'message': str(e), 'deleted': 0}


def get_alerts_clear_impl(request=None):
    """清除全部告警视频和图片"""
    try:
        from storage import get_alarm_output_dir, clear_all_alarm_files
        output_dir = get_alarm_output_dir(CONFIG.get('output'))
        deleted = clear_all_alarm_files(output_dir)
        return {'status': 'success', 'message': f'已删除 {deleted} 个文件', 'deleted': deleted}
    except Exception as e:
        logger.error("Failed to clear alerts: %s", e)
        return {'status': 'error', 'message': str(e), 'deleted': 0}


def delete_alerts_batch_impl(data):
    """批量删除指定的告警文件（视频/图片），按文件名列表删除。"""
    try:
        from storage import get_alarm_output_dir, _is_alarm_file  # _is_alarm_file 为内部函数，这里仅用于类型校验
        output_dir = Path(get_alarm_output_dir(CONFIG.get('output')))
        names = data.get('names') if isinstance(data, dict) else None
        if not isinstance(names, list) or not names:
            return {'status': 'error', 'message': '缺少要删除的文件列表'}
        deleted = 0
        for name in names:
            name = (str(name) or '').strip()
            if not name or '..' in name or '/' in name or '\\' in name:
                continue
            f = output_dir / name
            try:
                if f.is_file() and _is_alarm_file(f):
                    f.unlink()
                    deleted += 1
            except Exception:
                continue
        return {'status': 'success', 'message': f'已删除 {deleted} 个文件', 'deleted': deleted}
    except Exception as e:
        logger.error("Failed to delete alerts batch: %s", e)
        return {'status': 'error', 'message': str(e), 'deleted': 0}


def get_alerts_file_impl(request):
    """告警文件下载 - 按文件名安全返回文件，供浏览器下载"""
    import os
    from storage import get_alarm_output_dir
    name = (request.args.get('name') or '').strip()
    if not name or '..' in name or '/' in name or '\\' in name:
        return {'status': 'error', 'message': '无效文件名'}
    output_dir = get_alarm_output_dir(CONFIG.get('output'))
    output_path = Path(output_dir)
    file_path = output_path / name
    try:
        file_path = file_path.resolve()
        output_abs = output_path.resolve()
        if not file_path.is_file() or not str(file_path).startswith(str(output_abs)):
            return {'status': 'error', 'message': '文件不存在'}
        return (str(file_path), name)
    except Exception as e:
        logger.error("Failed to serve alert file: %s", e)
        return {'status': 'error', 'message': str(e)}


def get_video_streams_impl():
    """获取视频流配置（名称 + IP），供“视频流”页面使用。最多 9 路。"""
    try:
        streams = CONFIG.get('video_streams')

        # 若尚未保存过 video_streams，返回单行空占位，由用户在「视频流」页添加
        if not isinstance(streams, list) or len(streams) == 0:
            streams = [{'name': '通道 1', 'ip': ''}]

        # 兜底：清洗结构，限制 9 路
        cleaned_streams = []
        for i, item in enumerate(streams[:9]):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or f'通道 {i + 1}')
            ip = str(item.get('ip') or '').strip()
            cleaned_streams.append({'name': name, 'ip': ip})

        return {
            'status': 'success',
            'streams': cleaned_streams,
            'max_streams': 9,
        }
    except Exception as e:
        logger.error("Failed to get video streams: %s", e)
        return {'status': 'error', 'message': str(e), 'streams': [], 'max_streams': 9}


def save_video_streams_impl(data):
    """保存视频流配置到 config_bm1684x.json，并在系统运行时热重载视频流。"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        raw_streams = data.get('streams') or []
        if not isinstance(raw_streams, list):
            return {'status': 'error', 'message': '无效的视频流数据'}

        # 清洗并限制为 1~9 路，IP 不能为空
        cleaned_streams = []
        for i, item in enumerate(raw_streams[:9]):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or f'通道 {i + 1}')
            ip = str(item.get('ip') or '').strip()
            if not ip:
                continue
            cleaned_streams.append({'name': name, 'ip': ip})

        if not cleaned_streams:
            return {'status': 'error', 'message': '至少需要配置一路有效的视频流 IP 地址'}

        # 读取原配置，仅写入 video_streams（启动时仅据此构建视频源）
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config['video_streams'] = cleaned_streams

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 重新加载全局 CONFIG
        CONFIG = load_config()
        logger.info("Video stream config saved to config_bm1684x.json")
        # 系统运行中则热重载视频流，无需用户重启
        try:
            if scheduler is not None and getattr(scheduler, 'running', False):
                scheduler.reload_streams(CONFIG)
        except Exception as ex:
            logger.error("Failed to reload streams: %s", ex)
        return {'status': 'success', 'message': '视频流配置已保存'}
    except Exception as e:
        logger.error("Failed to save video streams: %s", e)
        return {'status': 'error', 'message': str(e)}


# ==================== 任务管理 API ====================
VALID_DETECTORS = ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door_inside', 'window_door_outside']


def get_tasks_impl():
    """获取任务列表（从 config 的 tasks 字段）"""
    try:
        tasks = CONFIG.get('tasks')
        if not isinstance(tasks, list):
            tasks = []
        # 确保每条有 id
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if 'id' not in t or t['id'] is None:
                t['id'] = 't_' + str(int(time.time() * 1000))
        return {'status': 'success', 'tasks': tasks}
    except Exception as e:
        logger.error("Failed to get tasks: %s", e)
        return {'status': 'error', 'message': str(e), 'tasks': []}


def add_task_impl(data):
    """新增任务：name, stream_index (1-9), detectors[]"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        name = (data.get('name') or '').strip()
        if not name:
            return {'status': 'error', 'message': '任务名称不能为空'}
        stream_index = int(data.get('stream_index', 1))
        if stream_index < 1 or stream_index > 9:
            stream_index = 1
        raw = data.get('detectors') or []
        detectors = [d for d in raw if d in VALID_DETECTORS]
        if not detectors:
            return {'status': 'error', 'message': '请至少选择一个检测器'}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        tasks = config.get('tasks')
        if not isinstance(tasks, list):
            tasks = []
        task_id = 't_' + str(int(time.time() * 1000))
        tasks.append({
            'id': task_id,
            'name': name,
            'stream_index': stream_index,
            'detectors': detectors,
        })
        config['tasks'] = tasks
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        CONFIG = load_config()
        # 若系统正在运行，刷新调度器内存配置，新增任务可立即被启动/停止使用
        try:
            if scheduler is not None and getattr(scheduler, 'running', False):
                scheduler.config = CONFIG
        except Exception:
            pass
        logger.info("Task added: %s", task_id)
        return {'status': 'success', 'message': '任务已添加', 'id': task_id}
    except Exception as e:
        logger.error("Failed to add task: %s", e)
        return {'status': 'error', 'message': str(e)}


def update_task_impl(data):
    """更新任务：id, name, stream_index, detectors"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        task_id = data.get('id')
        if not task_id:
            return {'status': 'error', 'message': '缺少任务 id'}
        name = (data.get('name') or '').strip()
        if not name:
            return {'status': 'error', 'message': '任务名称不能为空'}
        stream_index = int(data.get('stream_index', 1))
        if stream_index < 1 or stream_index > 9:
            stream_index = 1
        raw = data.get('detectors') or []
        detectors = [d for d in raw if d in VALID_DETECTORS]
        if not detectors:
            return {'status': 'error', 'message': '请至少选择一个检测器'}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        tasks = config.get('tasks')
        if not isinstance(tasks, list):
            return {'status': 'error', 'message': '无任务列表'}
        found = False
        for t in tasks:
            if isinstance(t, dict) and str(t.get('id')) == str(task_id):
                t['name'] = name
                t['stream_index'] = stream_index
                t['detectors'] = detectors
                found = True
                break
        if not found:
            return {'status': 'error', 'message': '任务不存在'}
        config['tasks'] = tasks
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        CONFIG = load_config()
        # 若系统正在运行，刷新调度器内存配置，更新后的任务参数立刻生效
        try:
            if scheduler is not None and getattr(scheduler, 'running', False):
                scheduler.config = CONFIG
        except Exception:
            pass
        logger.info("Task updated: %s", task_id)
        return {'status': 'success', 'message': '任务已更新'}
    except Exception as e:
        logger.error("Failed to update task: %s", e)
        return {'status': 'error', 'message': str(e)}


def delete_task_impl(data):
    """删除任务：id；若任务正在运行则先停止再删除"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        task_id = data.get('id')
        if not task_id:
            return {'status': 'error', 'message': '缺少任务 id'}
        if scheduler is not None and getattr(scheduler, 'task_running', {}).get(str(task_id)):
            try:
                scheduler.stop_task(task_id)
            except Exception:
                pass
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        tasks = config.get('tasks')
        if not isinstance(tasks, list):
            return {'status': 'success', 'message': '已删除'}
        new_tasks = [t for t in tasks if not (isinstance(t, dict) and str(t.get('id')) == str(task_id))]
        config['tasks'] = new_tasks
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        CONFIG = load_config()
        # 若系统正在运行，同步调度器内存配置，避免引用已删除任务
        try:
            if scheduler is not None and getattr(scheduler, 'running', False):
                scheduler.config = CONFIG
        except Exception:
            pass
        logger.info("Task deleted: %s", task_id)
        return {'status': 'success', 'message': '已删除'}
    except Exception as e:
        logger.error("Failed to delete task: %s", e)
        return {'status': 'error', 'message': str(e)}


def set_preview_mode_impl(data):
    """设置预览模式：merged=九宫格，task=单路任务通道。data: {mode, task_id?}"""
    try:
        mode = (data.get('mode') or 'merged').lower()
        if mode not in ('merged', 'task'):
            mode = 'merged'
        stream_id = 0
        if mode == 'task':
            task_id = data.get('task_id')
            if task_id:
                res = get_tasks_impl()
                if res.get('status') == 'success' and res.get('tasks'):
                    t = next((x for x in res['tasks'] if str(x.get('id')) == str(task_id)), None)
                    if t and t.get('running'):
                        stream_index = int(t.get('stream_index', 1))
                        stream_id = max(0, min(8, stream_index - 1))
        if scheduler and hasattr(scheduler, 'control_queues'):
            try:
                scheduler.control_queues['display'].put_nowait({
                    'cmd': 'set_preview', 'mode': mode, 'stream_id': stream_id
                })
            except Exception:
                pass
        return {'status': 'success', 'mode': mode, 'stream_id': stream_id}
    except Exception as e:
        logger.error("set_preview_mode: %s", e)
        return {'status': 'error', 'message': str(e)}


# 后端路由使用的 handlers 与 Flask 应用（事件驱动微服务 - Web 层）
HANDLERS = {
    'start': start_system_impl,
    'stop': stop_system_impl,
    'toggle_detector': toggle_detector_impl,
    'status': get_status_impl,
    'alerts_history': get_alerts_history_impl,
    'alerts_file': get_alerts_file_impl,
    'alerts_cleanup': get_alerts_cleanup_impl,
    'alerts_clear': get_alerts_clear_impl,
    'alerts_delete_batch': delete_alerts_batch_impl,
    'video_streams_get': get_video_streams_impl,
    'video_streams_save': save_video_streams_impl,
    'tasks_get': get_tasks_impl,
    'tasks_add': add_task_impl,
    'tasks_update': update_task_impl,
    'tasks_delete': delete_task_impl,
    'preview_mode_set': set_preview_mode_impl,
}

from backend.app import create_app
app = create_app(
    HANDLERS,
    get_scheduler=lambda: scheduler,
    get_alert_queue=lambda: getattr(scheduler, 'alert_queue', None) if scheduler else None,
    get_hls_dir=lambda: (Path(CONFIG.get('output', {}).get('hls_output_dir', 'hls_output')).resolve()
                         if CONFIG else None),
)


# ==================== 主程序入口 ====================
def main():
    """主程序入口"""
    global scheduler
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if scheduler:
            scheduler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    port = CONFIG['output'].get('display_port', 5000)
    logger.info(f"🌐 Starting web interface on http://0.0.0.0:{port}")
    logger.info(f"📹 Video output directory: {CONFIG['output']['video_output_dir']}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if scheduler:
            scheduler.stop()


if __name__ == '__main__':
    main()