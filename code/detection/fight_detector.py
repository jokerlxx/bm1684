"""
第二层:打架检测服务 (Fight Detector Service) - 简化版
改进内容:
1. 参考 realtimevideodetectorfight3.py 的简单状态机逻辑
2. 收集30帧,如果24帧以上检测到打架则触发警报
3. 报警持续2秒
4. 冷却期机制避免重复报警
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime
import logging
import signal
import sys
from collections import deque

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [FightDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 打架检测服务 ====================
class FightDetectorService:
    """打架检测服务 - 简化版 (使用状态机)"""
    
    def __init__(self, model_path, frame_queue, result_queue, control_queue, config):
        """
        Args:
            model_path: 打架检测模型路径
            frame_queue: 输入帧队列
            result_queue: 输出结果队列
            control_queue: 控制命令队列
            config: 配置字典
        """
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config
        
        # 模型
        self.detector = None
        
        # 从配置获取参数
        fight_config = config.get('fight_detection', {})
        self.conf_threshold = fight_config.get('conf_threshold', 0.5)
        self.cooldown_duration = fight_config.get('cooldown_duration', 180)  # 3分钟冷却期
        
        # 状态机参数（参考 realtimevideodetectorfight3.py）
        self.COLLECT_FRAMES = 30  # 收集30帧
        self.FIGHT_THRESHOLD = 12  # 24帧以上检测到打架则触发 (80%)
        self.ALERT_DURATION = 2.0  # 报警持续2秒
        
        # 状态机状态
        self.state = "NORMAL"  # NORMAL, COLLECTING, ALERTING
        self.fight_buffer = []  # 存储每帧是否检测到打架
        self.alert_start_time = 0
        
        # 冷却期管理
        self.alarm_number = 0  # 报警计数
        self.cooldown_start_time = 0
        
        # 当前检测信息
        self.current_detections = []
        self.last_fight_bbox = None  # 最后一次检测到的打架区域
        
        # 运行状态
        self.running = False
        self.enabled = False
        self.enabled_streams = None
        self.frame_count = 0
        self.total_alerts = 0
        
        logger.info(f"FightDetectorService initialized (Simple State Machine):")
        logger.info(f"  - Collect frames: {self.COLLECT_FRAMES}")
        logger.info(f"  - Fight threshold: {self.FIGHT_THRESHOLD}/{self.COLLECT_FRAMES} ({self.FIGHT_THRESHOLD/self.COLLECT_FRAMES*100:.0f}%)")
        logger.info(f"  - Alert duration: {self.ALERT_DURATION}s")
        logger.info(f"  - Cooldown: {self.cooldown_duration}s")
    
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
        """启动检测服务"""
        logger.info("Starting fight detector service...")
        self.setup_signal_handlers()
        
        # 初始化模型
        try:
            self.detector = create_yolo_detector(
                self.model_path,
                conf_threshold=self.conf_threshold,
                device_id=self.config.get('bm1684x', {}).get('device_id', 0)
            )
            logger.info("✅ Fight detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_fight(self, frame):
        """
        检测打架
        
        Args:
            frame: 输入帧
        
        Returns:
            detections: 检测结果列表
        """
        # Fight 模型类别：0=fight，仅对 class_id=0 报警
        dets = run_yolo_inference(
            self.detector,
            frame,
            conf_threshold=self.conf_threshold,
            allowed_classes=[0],
        )
        results = []
        for d in dets:
            results.append(
                {
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "class_id": d["class_id"],
                    "class_name": "fight",
                }
            )
        return results
    
    def run(self):
        """主循环"""
        logger.info("Fight detector service running...")
        
        while self.running:
            # 检查控制命令
            self._check_control_commands()
            
            # 获取帧数据
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
            except:
                continue
            
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            stream_id = frame_data.get('stream_id', 0)
            
            # 推理采样：每 N 帧推理一次，降低 BM1684x 负载
            subsample = self.config.get('detection_inference_subsample', 1)
            if subsample > 1 and (frame_number % subsample != 0):
                continue
            
            self.frame_count += 1
            
            # 如果检测被禁用,发送空结果
            if not self.enabled:
                result = {
                    'detector_type': 'fight',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],
                    'state': self.state,
                    'alarm_number': self.alarm_number,
                    'total_alerts': self.total_alerts,
                    'enabled': False
                }
                
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put(result, block=False)
                except:
                    pass
                continue
            
            if self.enabled_streams is not None and stream_id not in self.enabled_streams:
                continue
            
            # 执行检测
            try:
                detections = self.detect_fight(frame)
                self.current_detections = detections
                
                # 判断当前帧是否检测到打架
                has_fight = len(detections) > 0
                
                # 如果检测到打架,保存最后一个检测框
                if has_fight:
                    self.last_fight_bbox = detections[0]['bbox']
                
                # 状态机处理
                fight_alerts = []
                
                if self.state == "NORMAL":
                    if has_fight:
                        # 开始收集
                        self.state = "COLLECTING"
                        self.fight_buffer = [has_fight]
                        logger.info(f"🔍 State: COLLECTING (检测到打架,开始收集帧)")
                
                elif self.state == "COLLECTING":
                    self.fight_buffer.append(has_fight)
                    
                    if len(self.fight_buffer) >= self.COLLECT_FRAMES:
                        # 收集完成,判断是否触发报警
                        fight_count = sum(self.fight_buffer)
                        
                        if fight_count >= self.FIGHT_THRESHOLD:
                            # 触发报警
                            self.state = "ALERTING"
                            self.alert_start_time = time.time()
                            
                            # 报警计数器逻辑
                            if self.alarm_number == 0:
                                # 首次报警
                                self.alarm_number = 1
                                self.cooldown_start_time = time.time()
                                self.total_alerts += 1
                                
                                logger.warning(f"🚨 报警触发! (帧号: {frame_number}, 打架帧数: {fight_count}/{self.COLLECT_FRAMES})")
                                
                                # 准备警报信息（此时创建警报，bbox会在ALERTING状态持续更新）
                                fight_alerts.append({
                                    'alert_id': self.total_alerts,
                                    'bbox': self.last_fight_bbox if self.last_fight_bbox else [0, 0, 100, 100],
                                    'fight_count': fight_count,
                                    'collect_frames': self.COLLECT_FRAMES,
                                    'fight_rate': fight_count / self.COLLECT_FRAMES,
                                    'confidence': detections[0]['confidence'] if detections else 0.0,
                                    'alert_time': datetime.now()
                                })
                            else:
                                # 冷却期内重复报警
                                self.alarm_number += 1
                                logger.info(f"⚠️ 冷却期内重复报警 (报警次数: {self.alarm_number})")
                        else:
                            # 未达标,回到正常状态
                            self.state = "NORMAL"
                            self.fight_buffer = []
                            logger.info(f"✅ 收集结束,未达阈值 (打架帧数: {fight_count}/{self.COLLECT_FRAMES})")
                
                elif self.state == "ALERTING":
                    # ✅ 关键修复：在ALERTING状态期间，持续输出警报信息（使用最新的检测框）
                    # 无论当前帧是否检测到打架，都要持续输出，确保显示层能实时更新bbox
                    fight_alerts.append({
                        'alert_id': self.total_alerts,
                        'bbox': self.last_fight_bbox if self.last_fight_bbox else [0, 0, 100, 100],
                        'fight_count': sum(self.fight_buffer) if self.fight_buffer else 0,
                        'collect_frames': self.COLLECT_FRAMES,
                        'fight_rate': sum(self.fight_buffer) / self.COLLECT_FRAMES if self.fight_buffer else 0,
                        'confidence': detections[0]['confidence'] if detections else 0.0,
                        'alert_time': datetime.now(),
                        'is_ongoing': True,  # 标记为持续报警
                        'has_current_detection': has_fight  # 标记当前帧是否检测到打架
                    })
                    
                    # 报警状态持续2秒
                    if time.time() - self.alert_start_time >= self.ALERT_DURATION:
                        self.state = "NORMAL"
                        self.fight_buffer = []
                        logger.info(f"🔔 报警结束,回到正常状态")
                
                # 检查冷却期是否结束
                if self.alarm_number > 0 and time.time() - self.cooldown_start_time >= self.cooldown_duration:
                    logger.info(f"❄️ 冷却期结束 (持续了 {self.cooldown_duration}s)")
                    self.alarm_number = 0
                
                # 准备输出结果
                result = {
                    'detector_type': 'fight',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': fight_alerts,  # 警报列表
                    'current_detections': detections,  # 当前帧检测结果
                    'state': self.state,  # 当前状态
                    'fight_buffer_size': len(self.fight_buffer),  # 缓冲区大小
                    'fight_count': sum(self.fight_buffer) if self.fight_buffer else 0,  # 检测到打架的帧数
                    'alarm_number': self.alarm_number,  # 当前报警次数
                    'total_alerts': self.total_alerts,  # 总报警次数
                    'enabled': True
                }
                
                # 输出结果
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put(result, block=False)
                except:
                    pass
                
                # 定期打印状态
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames, "
                              f"State: {self.state}, "
                              f"Buffer: {len(self.fight_buffer)}/{self.COLLECT_FRAMES}, "
                              f"Total alerts: {self.total_alerts}, "
                              f"Alarm number: {self.alarm_number}, "
                              f"Enabled: {self.enabled}")
                    
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
        
        logger.info("Fight detector service stopped")
    
    def _check_control_commands(self):
        """检查控制命令"""
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                logger.info(f"Received command: {cmd}")
                if isinstance(cmd, dict):
                    if cmd.get('cmd') == 'set_streams':
                        self.enabled_streams = set(cmd.get('stream_ids') or [])
                    continue
                if cmd == 'stop':
                    self.stop()
                elif cmd == 'enable':
                    self.enabled = True
                    logger.info("✅ Fight detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Fight detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        """停止检测服务"""
        logger.info("Stopping fight detector service...")
        self.running = False


def run_fight_detector(model_path, frame_queue, result_queue, control_queue, config):
    """进程入口函数"""
    service = FightDetectorService(model_path, frame_queue, result_queue, control_queue, config)
    service.start()


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Fight Detector Service (Simple State Machine)')
    parser.add_argument('--model', type=str, required=True, help='Fight detection model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_fight_detector(args.model, frame_queue, result_queue, control_queue, config)