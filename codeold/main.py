"""
主控程序 (Main Scheduler) - 时间段自动开关版
新功能：
1. 为每个检测器设置时间段（开始时间-结束时间）
2. 自动识别跨天时间段
3. 到时间自动开启/关闭检测器
4. 支持手动操作，不影响自动调度
5. 使用北京时间（Asia/Shanghai）
"""

import multiprocessing as mp
import time
from datetime import datetime, timedelta
import logging
import sys
import signal
from pathlib import Path
import json
from threading import Thread
import pytz
try:
    import psutil  # 可选：用于 CPU/内存/磁盘等系统信息
except Exception:
    psutil = None

# 导入服务（视频流接入 / 检测引擎 / 展示服务）
from ingestion import run_stream_service
from detection import (
    run_fall_detector,
    run_ventilator_detector,
    run_fight_detector,
    run_crowd_detector,
    run_helmet_detector,
    run_window_door_detector,
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

# 应用启动时间 & 版本号（用于“系统信息”页面）
APP_START_TIME = time.time()
APP_VERSION = "1.0.0"


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
            'input_mode': 0,
            'stream_count': 1,
            'rtsp_url': 'rtsp://admin:1q2w3e4r@192.168.150.64:554/Streaming/Channels/102',
            'rtsp_urls': ['rtsp://admin:1q2w3e4r@192.168.150.64:554/Streaming/Channels/102'],
            'input_video_path': '',
            'input_video_paths': [],
            'video_loop': True,
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


# ==================== 时间段配置管理 ====================
class TimeslotConfig:
    """时间段配置管理器"""
    
    def __init__(self, config_file='timeslot_config.json'):
        self.config_file = config_file
        self.config = self.load()
    
    def load(self):
        """加载时间段配置"""
        default_config = {
            'fall': {'enabled': False, 'start': '08:00', 'end': '18:00'},
            'ventilator': {'enabled': False, 'start': '08:00', 'end': '18:00'},
            'fight': {'enabled': False, 'start': '08:00', 'end': '18:00'},
            'crowd': {'enabled': False, 'start': '08:00', 'end': '18:00'},
            'helmet': {'enabled': False, 'start': '08:00', 'end': '18:00'},
            'window_door': {'enabled': False, 'start': '08:00', 'end': '18:00'}
        }
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # 合并默认配置，确保所有检测器都有配置
                for detector in default_config:
                    if detector not in loaded:
                        loaded[detector] = default_config[detector]
                return loaded
        except FileNotFoundError:
            logger.info(f"Timeslot config not found, creating default: {self.config_file}")
            self.save(default_config)
            return default_config
        except Exception as e:
            logger.error(f"Failed to load timeslot config: {e}")
            return default_config
    
    def save(self, config=None):
        """保存时间段配置"""
        if config is not None:
            self.config = config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Timeslot config saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save timeslot config: {e}")
            return False
    
    def get(self, detector):
        """获取指定检测器的时间段配置"""
        return self.config.get(detector, {
            'enabled': False,
            'start': '08:00',
            'end': '18:00'
        })
    
    def set(self, detector, enabled, start_time, end_time):
        """设置指定检测器的时间段"""
        self.config[detector] = {
            'enabled': enabled,
            'start': start_time,
            'end': end_time
        }
        return self.save()


# ==================== 时间段调度器 ====================
class TimeslotScheduler:
    """时间段调度器 - 自动开关检测器"""
    
    def __init__(self, main_scheduler, timeslot_config):
        self.main_scheduler = main_scheduler
        self.timeslot_config = timeslot_config
        self.running = False
        self.thread = None
        
        # 记录上一次的状态，避免重复操作
        self.last_states = {}
    
    def start(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.thread = Thread(target=self._schedule_loop, daemon=True)
        self.thread.start()
        logger.info("⏰ Timeslot scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("⏰ Timeslot scheduler stopped")
    
    def _schedule_loop(self):
        """调度循环 - 每10秒检查一次（更快响应）"""
        while self.running:
            try:
                # 检查所有检测器
                for detector in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door']:
                    self._check_detector(detector)
                
                # 等待10秒（快速响应）
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}")
                time.sleep(10)
    
    def _check_detector(self, detector):
        """检查单个检测器的时间段状态"""
        # 获取配置
        config = self.timeslot_config.get(detector)
        
        # 如果未启用时间段调度，跳过
        if not config.get('enabled', False):
            return
        
        # 判断当前是否应该开启
        should_be_running = self._is_in_timeslot(config['start'], config['end'])
        
        # 获取当前实际运行状态
        is_running = self.main_scheduler.detector_running.get(detector, False)
        
        # 获取上一次的状态
        last_state = self.last_states.get(detector, None)
        
        # 状态发生变化时才操作
        if should_be_running != last_state:
            current_time = datetime.now(BEIJING_TZ).strftime('%H:%M')
            
            if should_be_running and not is_running:
                # 应该运行但未运行 - 自动启动
                logger.info(f"⏰ [AUTO] {current_time} - Starting {detector} detector (time slot: {config['start']}-{config['end']})")
                self.main_scheduler.start_detector(detector)
                
            elif not should_be_running and is_running:
                # 不应该运行但正在运行 - 自动停止
                logger.info(f"⏰ [AUTO] {current_time} - Stopping {detector} detector (outside time slot: {config['start']}-{config['end']})")
                self.main_scheduler.stop_detector(detector)
            
            # 更新状态记录
            self.last_states[detector] = should_be_running
    
    def _is_in_timeslot(self, start_time_str, end_time_str):
        """
        判断当前时间是否在时间段内
        
        Args:
            start_time_str: 开始时间 "HH:MM"
            end_time_str: 结束时间 "HH:MM"
        
        Returns:
            bool: 是否在时间段内
        """
        # 获取当前北京时间
        now = datetime.now(BEIJING_TZ)
        current_time = now.time()
        
        # 解析时间字符串
        start_hour, start_min = map(int, start_time_str.split(':'))
        end_hour, end_min = map(int, end_time_str.split(':'))
        
        from datetime import time as dt_time
        start_time = dt_time(start_hour, start_min)
        end_time = dt_time(end_hour, end_min)
        
        # 判断是否跨天
        if start_time <= end_time:
            # 不跨天: 18:30-22:40
            # 当前时间 >= 开始时间 AND 当前时间 <= 结束时间
            return start_time <= current_time <= end_time
        else:
            # 跨天: 22:00-06:30
            # 当前时间 >= 开始时间 OR 当前时间 <= 结束时间
            return current_time >= start_time or current_time <= end_time


# 加载配置
CONFIG = load_config()

# 全局调度器实例
scheduler = None
timeslot_config = TimeslotConfig()
timeslot_scheduler = None


# ==================== 主调度器类 ====================
class MainScheduler:
    """主调度器 - 管理所有服务进程"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        self.processes = {}
        
        # 采集输出队列分离：一个给展示/编码（保证预览 FPS），一个给检测器（避免抢帧）
        self.display_frame_queue = mp.Queue(maxsize=config['queue_sizes']['frame_queue'])
        self.detect_frame_queue = mp.Queue(maxsize=config['queue_sizes']['frame_queue'])
        self.result_queues = {
            'fall': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'ventilator': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'fight': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'crowd': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'helmet': mp.Queue(maxsize=config['queue_sizes']['result_queue']),
            'window_door': mp.Queue(maxsize=config['queue_sizes']['result_queue'])
        }
        self.output_queue = mp.Queue(maxsize=config['queue_sizes']['display_queue'])
        # 已标注帧队列：展示服务写入，编码服务读取，用于 HLS 低延迟流（解码-处理-编码流水线）
        self.encoder_queue = mp.Queue(maxsize=config['queue_sizes'].get('encoder_queue', 5))
        # 告警事件队列：展示服务写入，Web 层读取并推送至前端（SSE/WebSocket）
        self.alert_queue = mp.Queue(maxsize=200)
        
        self.control_queues = {
            'stream': mp.Queue(),
            'fall': mp.Queue(),
            'ventilator': mp.Queue(),
            'fight': mp.Queue(),
            'crowd': mp.Queue(),
            'helmet': mp.Queue(),
            'window_door': mp.Queue(),
            'display': mp.Queue(),
            'encoder': mp.Queue()
        }
        
        self.detector_running = {
            'fall': False,
            'ventilator': False,
            'fight': False,
            'crowd': False,
            'helmet': False,
            'window_door': False
        }
        
        logger.info("MainScheduler initialized")
    
    def start(self):
        """启动核心服务"""
        logger.info("=" * 60)
        logger.info("Starting Multi-Task Detection System - Timeslot Version")
        logger.info("=" * 60)

        try:
            import os
            input_mode = self.config.get('input_mode', 0)
            stream_count = int(self.config.get('stream_count', 1))
            if stream_count not in (1, 2, 4, 9, 16):
                stream_count = 1
                logger.warning("stream_count must be 1, 2, 4, 9, or 16; using 1")

            # 从配置中读取 BM1684X 设备号，后续传递给视频流服务，便于使用 Sophon SAIL 硬件编解码
            bm_cfg = self.config.get('bm1684x', {}) or {}
            device_id = int(bm_cfg.get('device_id', 0))
            
            # 构建多路视频源列表
            if input_mode == 0:
                if stream_count == 1:
                    video_sources = [self.config.get('rtsp_url', '')]
                else:
                    video_sources = self.config.get('rtsp_urls', [])
                    if len(video_sources) < stream_count:
                        video_sources = video_sources + [video_sources[-1]] * (stream_count - len(video_sources)) if video_sources else [self.config.get('rtsp_url', '')] * stream_count
                    video_sources = video_sources[:stream_count]
                logger.info(f"📡 Input Mode: RTSP Stream, streams: {stream_count}")
                for i, src in enumerate(video_sources):
                    logger.info(f"   Stream {i}: {src[:70]}...")
            else:
                if stream_count == 1:
                    path = self.config.get('input_video_path', '')
                    if not path or not os.path.exists(path):
                        raise FileNotFoundError(f"Video file not found: {path}")
                    video_sources = [path]
                else:
                    video_sources = self.config.get('input_video_paths', [])
                    if len(video_sources) < stream_count:
                        video_sources = (video_sources + [video_sources[-1]] * (stream_count - len(video_sources))) if video_sources else []
                    video_sources = video_sources[:stream_count]
                    for p in video_sources:
                        if not os.path.exists(p):
                            raise FileNotFoundError(f"Video file not found: {p}")
                logger.info(f"📹 Input Mode: Video File, streams: {stream_count}")
                for i, src in enumerate(video_sources):
                    logger.info(f"   Stream {i}: {src}")
            
            # 1. 启动流服务（一路一进程，多路多进程）
            logger.info("Starting Stream Service(s)...")
            if stream_count == 1:
                self.processes['stream'] = mp.Process(
                    target=run_stream_service,
                    args=(
                        video_sources[0],
                        [self.display_frame_queue, self.detect_frame_queue],
                        self.control_queues['stream'],
                        self.config['fps'],
                        input_mode,
                        0,              # stream_id
                        stream_count,
                        device_id,      # 传递 BM1684X 设备号，供流服务选择 Sophon 解码
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
                            self.control_queues['stream'],
                            self.config['fps'],
                            input_mode,
                            i,            # stream_id
                            stream_count,
                            device_id,    # 传递 BM1684X 设备号
                        ),
                        daemon=True
                    )
                    self.processes[key].start()
            time.sleep(2)
            
            # 2. 启动显示服务（传入 stream_count、alert_queue、font_path、encoder_queue 用于实时告警与 HLS 流水线）
            logger.info("Starting Display Service...")
            font_path = self.config.get('output', {}).get('font_path', 'simhei.ttf')
            out_cfg = self.config.get('output', {})
            hls_dir = out_cfg.get('hls_output_dir', './hls_output')
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
            
            self.running = True
            logger.info("=" * 60)
            logger.info("✅ Core services started successfully!")
            logger.info(f"   - Input Mode: {'RTSP Stream' if input_mode == 0 else 'Video File'}")
            logger.info(f"   - Stream count: {stream_count}")
            logger.info("   - Stream Service(s): Running")
            logger.info("   - Display Service: Running")
            logger.info("   - Encoder Service (HLS): Running")
            logger.info("   - Timeslot Scheduler: Will start automatically")
            logger.info("   - Detectors: Waiting for manual or auto start")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            self.stop()
            raise
    
    def start_detector(self, detector_name):
        """启动单个检测器"""
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
            elif detector_name == 'window_door':
                process = mp.Process(
                    target=run_window_door_detector,
                    args=(
                        self.config['models']['window_door_detection'],
                        self.detect_frame_queue,
                        self.result_queues['window_door'],
                        self.control_queues['window_door'],
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
            logger.info(f"✅ {detector_name} detector stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {detector_name} detector: {e}")
            return False
    
    def stop(self):
        """停止所有服务"""
        logger.info("Stopping all services...")
        
        self.running = False
        
        for name, queue in self.control_queues.items():
            try:
                if name == 'stream':
                    n = sum(1 for k in self.processes if k == 'stream' or k.startswith('stream_'))
                    for _ in range(max(1, n)):
                        queue.put('stop')
                else:
                    queue.put('stop')
            except Exception:
                pass
        
        time.sleep(1)
        
        for name, process in self.processes.items():
            if process.is_alive():
                logger.info(f"Terminating {name} service...")
                process.terminate()
                process.join(timeout=3)
                
                if process.is_alive():
                    process.kill()
                    process.join()
        
        self.processes.clear()
        
        for key in self.detector_running:
            self.detector_running[key] = False
        
        logger.info("✅ All services stopped")
    
    def get_status(self):
        """获取所有服务状态"""
        status = {
            'system_running': self.running,
            'detectors': {}
        }
        
        for name in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door']:
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
    global scheduler, timeslot_scheduler
    try:
        if scheduler is not None and scheduler.running:
            return {'status': 'error', 'message': '系统已在运行中'}
        logger.info("Starting system with timeslot scheduler...")
        scheduler = MainScheduler(CONFIG)
        scheduler.start()
        timeslot_scheduler = TimeslotScheduler(scheduler, timeslot_config)
        timeslot_scheduler.start()
        return {'status': 'success', 'message': '系统启动成功，时间段调度器已启动'}
    except Exception as e:
        logger.error("Failed to start system: %s", e)
        return {'status': 'error', 'message': str(e)}


def stop_system_impl():
    """停止系统 - 返回可 jsonify 的字典"""
    global scheduler, timeslot_scheduler
    try:
        if scheduler is None:
            return {'status': 'error', 'message': '系统未运行'}
        logger.info("Stopping system...")
        if timeslot_scheduler:
            timeslot_scheduler.stop()
            timeslot_scheduler = None
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
        if detector not in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door']:
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
    # 在原有状态基础上补充视频流进程状态，供“系统信息”页面使用
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


def _get_system_resource_info():
    """
    收集系统资源信息（CPU / 内存 / 存储 / TPU）。
    若 psutil 不可用，则尽量返回基础信息。
    """
    # CPU / 内存
    cpu_percent = None
    mem_used = None
    mem_total = None
    mem_percent = None
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            mem_used = vm.used
            mem_total = vm.total
            mem_percent = vm.percent
        except Exception:
            pass

    # 存储：以告警视频目录所在分区为准，并估算 alarm_videos 目录占用率
    from shutil import disk_usage
    output_cfg = CONFIG.get('output', {}) if CONFIG else {}
    alarm_dir = output_cfg.get('video_output_dir', 'alarm_videos')
    alarm_path = Path(alarm_dir).resolve()

    disk_total = None
    disk_used = None
    disk_percent = None
    dir_size = 0
    dir_usage_percent = None
    try:
        usage = disk_usage(str(alarm_path if alarm_path.exists() else alarm_path.parent))
        disk_total = usage.total
        disk_used = usage.used
        disk_percent = usage.used * 100.0 / usage.total if usage.total else None
    except Exception:
        pass

    # 计算目录自身大小（可能稍慢，但目录一般不至于特别大）
    try:
        if alarm_path.exists():
            for p in alarm_path.rglob('*'):
                if p.is_file():
                    dir_size += p.stat().st_size
        if disk_total:
            dir_usage_percent = dir_size * 100.0 / disk_total
    except Exception:
        pass

    # BM1684X TPU 使用率：优先尝试通过命令行工具获取（如 bm-smi），失败则保持为空
    tpu_info = {
        "name": "BM1684X",
        "utilization_percent": None,
        "memory_used": None,
        "memory_total": None,
        "note": "如已安装 bm-smi/bmsmi 等监控工具，可在此处扩展解析以获得更精确的 TPU 使用率",
    }
    try:
        import subprocess
        # 常见 Sophon 工具命令名尝试；如需适配其他命令可在此扩展
        candidates = [
            ["bm-smi", "--json"],
            ["bmsmi", "--json"],
        ]
        raw = None
        for cmd in candidates:
            try:
                raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0)
                if raw:
                    break
            except Exception:
                raw = None
        if raw:
            try:
                info = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                info = None
            # 具体 JSON 结构依赖实际工具版本，以下为尽量稳健的“猜测式”解析：
            if isinstance(info, dict):
                devs = info.get("devs") or info.get("devices") or []
                if isinstance(devs, list) and devs:
                    d0 = devs[0]
                    util = d0.get("utilization") or d0.get("tpu_util") or d0.get("tpuUtil")
                    mem_used = d0.get("mem_used") or d0.get("tpu_mem_used")
                    mem_total = d0.get("mem_total") or d0.get("tpu_mem_total")
                    # 允许字符串百分号形式，如 "35%"
                    if isinstance(util, str) and util.strip().endswith("%"):
                        try:
                            util = float(util.strip().rstrip("%"))
                        except Exception:
                            util = None
                    if isinstance(util, (int, float)):
                        tpu_info["utilization_percent"] = float(util)
                    if isinstance(mem_used, str):
                        try:
                            mem_used = float(mem_used)
                        except Exception:
                            mem_used = None
                    if isinstance(mem_total, str):
                        try:
                            mem_total = float(mem_total)
                        except Exception:
                            mem_total = None
                    if isinstance(mem_used, (int, float)):
                        tpu_info["memory_used"] = mem_used
                    if isinstance(mem_total, (int, float)):
                        tpu_info["memory_total"] = mem_total
                    if tpu_info["utilization_percent"] is not None:
                        tpu_info["note"] = "通过 bm-smi/bmsmi 解析得到的实时估算值"
    except Exception:
        # 完全兼容：若环境不存在相关工具，则保留默认提示信息
        pass

    return {
        "cpu": {
            "percent": cpu_percent,
        },
        "memory": {
            "used": mem_used,
            "total": mem_total,
            "percent": mem_percent,
        },
        "storage": {
            "alarm_videos_path": str(alarm_path),
            "dir_size": dir_size,
            "dir_usage_percent": dir_usage_percent,
            "disk_total": disk_total,
            "disk_used": disk_used,
            "disk_percent": disk_percent,
        },
        "tpu": tpu_info,
    }


def get_system_info_impl():
    """
    “系统信息”接口：
    - 基础信息：运行时间 / 北京时间 / 应用版本
    - 资源信息：CPU/内存/存储/TPU
    - 应用状态：主程序 / 检测器 / 视频流
    """
    try:
        now = datetime.now(BEIJING_TZ)
        uptime_seconds = max(0, time.time() - APP_START_TIME)
        # 状态信息复用 get_status_impl
        status = get_status_impl()
        resources = _get_system_resource_info()

        return {
            "status": "success",
            "basic": {
                "uptime_seconds": uptime_seconds,
                "beijing_time": now.isoformat(),
                "beijing_time_display": now.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": APP_VERSION,
            },
            "resources": resources,
            "app_status": status,
        }
    except Exception as e:
        logger.error("Failed to get system info: %s", e)
        return {"status": "error", "message": str(e)}


def save_timeslot_impl(data):
    """保存时间段配置 - 接收 request 数据，返回可 jsonify 的字典"""
    try:
        detector = data.get('detector')
        enabled = data.get('enabled', False)
        start_time = data.get('start', '08:00')
        end_time = data.get('end', '18:00')
        if detector not in ['fall', 'ventilator', 'fight', 'crowd', 'helmet', 'window_door']:
            return {'status': 'error', 'message': '未知的检测器'}
        success = timeslot_config.set(detector, enabled, start_time, end_time)
        if success:
            logger.info("⏰ %s 时间段配置已保存: %s-%s (enabled: %s)", detector, start_time, end_time, enabled)
            return {'status': 'success', 'message': '配置已保存'}
        return {'status': 'error', 'message': '保存失败'}
    except Exception as e:
        logger.error("Failed to save timeslot: %s", e)
        return {'status': 'error', 'message': str(e)}


def get_all_timeslots_impl():
    """获取所有时间段配置 - 返回可 jsonify 的字典"""
    try:
        return {'status': 'success', 'configs': timeslot_config.config}
    except Exception as e:
        logger.error("Failed to get timeslot configs: %s", e)
        return {'status': 'error', 'message': str(e)}


def check_timeslot_impl(detector):
    """检查是否在时间段内 - 返回可 jsonify 的字典"""
    global timeslot_scheduler
    try:
        if not detector:
            return {'status': 'error', 'message': '缺少detector参数'}
        config = timeslot_config.get(detector)
        if not config.get('enabled', False):
            return {'status': 'success', 'in_timeslot': False, 'reason': '时间段未启用'}
        if timeslot_scheduler:
            in_timeslot = timeslot_scheduler._is_in_timeslot(config['start'], config['end'])
            return {'status': 'success', 'in_timeslot': in_timeslot, 'start': config['start'], 'end': config['end']}
        return {'status': 'error', 'message': '调度器未运行'}
    except Exception as e:
        logger.error("Failed to check timeslot: %s", e)
        return {'status': 'error', 'message': str(e)}


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


def get_config_alarm_params_impl():
    """获取各模型报警参数（供配置页使用）"""
    try:
        keys = [
            'fall_detection', 'ventilator_detection', 'fight_detection',
            'crowd_detection', 'helmet_detection', 'window_door_detection'
        ]
        params = {k: CONFIG.get(k, {}) for k in keys if isinstance(CONFIG.get(k), dict)}
        return {'status': 'success', 'params': params}
    except Exception as e:
        logger.error("Failed to get alarm params: %s", e)
        return {'status': 'error', 'message': str(e), 'params': {}}


def save_config_alarm_params_impl(data):
    """保存各模型报警参数到配置文件"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        keys = [
            'fall_detection', 'ventilator_detection', 'fight_detection',
            'crowd_detection', 'helmet_detection', 'window_door_detection'
        ]
        for k in keys:
            if isinstance(data.get(k), dict):
                config[k] = {**config.get(k, {}), **data[k]}  # 合并，保留未提交字段
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        CONFIG = load_config()
        logger.info("Alarm params saved to config_bm1684x.json")
        return {'status': 'success', 'message': '配置已保存，部分参数需重启检测器后生效'}
    except Exception as e:
        logger.error("Failed to save alarm params: %s", e)
        return {'status': 'error', 'message': str(e)}


def get_config_stream_impl():
    """获取视频流配置（路数、输入模式、各路径）供 Web 配置页使用"""
    try:
        stream_count = int(CONFIG.get('stream_count', 1))
        if stream_count not in (1, 2, 4, 9, 16):
            stream_count = 1
        return {
            'status': 'success',
            'input_mode': int(CONFIG.get('input_mode', 0)),
            'stream_count': stream_count,
            'rtsp_url': CONFIG.get('rtsp_url', ''),
            'rtsp_urls': list(CONFIG.get('rtsp_urls', [])),
            'input_video_path': CONFIG.get('input_video_path', ''),
            'input_video_paths': list(CONFIG.get('input_video_paths', [])),
            'video_loop': bool(CONFIG.get('video_loop', True)),
        }
    except Exception as e:
        logger.error("Failed to get stream config: %s", e)
        return {'status': 'error', 'message': str(e)}


def save_config_stream_impl(data):
    """保存视频流配置到 config_bm1684x.json；生效需重启系统"""
    global CONFIG
    config_path = Path('config_bm1684x.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        stream_count = int(data.get('stream_count', 1))
        if stream_count not in (1, 2, 4, 9, 16):
            stream_count = 1
        config['stream_count'] = stream_count
        config['input_mode'] = int(data.get('input_mode', 0))
        config['video_loop'] = bool(data.get('video_loop', True))
        if config['input_mode'] == 0:
            urls = data.get('rtsp_urls') or []
            if stream_count == 1:
                config['rtsp_url'] = (urls[0] if urls else data.get('rtsp_url', '')) or config.get('rtsp_url', '')
            else:
                # 补齐到 stream_count
                while len(urls) < stream_count and urls:
                    urls.append(urls[-1])
                config['rtsp_urls'] = (urls or [config.get('rtsp_url', '')])[:stream_count]
        else:
            paths = data.get('input_video_paths') or []
            if stream_count == 1:
                config['input_video_path'] = (paths[0] if paths else data.get('input_video_path', '')) or config.get('input_video_path', '')
            else:
                while len(paths) < stream_count and paths:
                    paths.append(paths[-1])
                config['input_video_paths'] = (paths or [config.get('input_video_path', '')])[:stream_count]
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        CONFIG = load_config()
        logger.info("Stream config saved to config_bm1684x.json")
        return {'status': 'success', 'message': '视频流配置已保存，重启系统后生效'}
    except Exception as e:
        logger.error("Failed to save stream config: %s", e)
        return {'status': 'error', 'message': str(e)}


# 后端路由使用的 handlers 与 Flask 应用（事件驱动微服务 - Web 层）
HANDLERS = {
    'start': start_system_impl,
    'stop': stop_system_impl,
    'toggle_detector': toggle_detector_impl,
    'status': get_status_impl,
    'system_info': get_system_info_impl,
    'timeslot_save': save_timeslot_impl,
    'timeslot_get_all': get_all_timeslots_impl,
    'timeslot_check': check_timeslot_impl,
    'alerts_history': get_alerts_history_impl,
    'alerts_file': get_alerts_file_impl,
    'alerts_cleanup': get_alerts_cleanup_impl,
    'alerts_clear': get_alerts_clear_impl,
    'alerts_delete_batch': delete_alerts_batch_impl,
    'config_alarm_params_get': get_config_alarm_params_impl,
    'config_alarm_params_save': save_config_alarm_params_impl,
    'config_stream_get': get_config_stream_impl,
    'config_stream_save': save_config_stream_impl,
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
    logger.info("⏰ Timeslot scheduler enabled - detectors will auto start/stop")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if scheduler:
            scheduler.stop()


if __name__ == '__main__':
    main()