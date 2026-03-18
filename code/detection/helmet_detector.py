"""
第二层：安全帽检测服务 (Helmet Detector Service)
改进内容：
1. 基于IOU的目标跟踪
2. 持续3秒不戴安全帽触发报警
3. 只有第一个人触发视频录制（冷却期逻辑）
4. 但所有不戴安全帽超过3秒的人都在画面显示
5. 只检测两类：戴安全帽(class 0)和不戴安全帽(class 1)
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime
import logging
import signal
import sys
from collections import defaultdict

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [HelmetDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== IOU跟踪器 ====================
class SimpleIOUTracker:
    """基于IOU的简单跟踪器"""
    
    def __init__(self, max_age=10, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}
    
    def update(self, detections):
        current_ids = {}
        matched_detections = set()
        
        # 增加所有track的age
        for track_id, track in list(self.tracks.items()):
            track['age'] += 1
            
            # 删除过期track
            if track['age'] > self.max_age:
                del self.tracks[track_id]
                continue
            
            # 尝试匹配现有检测
            best_iou = 0
            best_detection_idx = -1
            
            for i, (det_bbox, det_conf, det_class) in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self.calculate_iou(track['bbox'], det_bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = i
            
            # 更新匹配的track
            if best_detection_idx != -1:
                det_bbox, det_conf, det_class = detections[best_detection_idx]
                self.tracks[track_id] = {
                    'bbox': det_bbox,
                    'age': 0,
                    'class': det_class,
                    'confidence': det_conf
                }
                current_ids[track_id] = self.tracks[track_id]
                matched_detections.add(best_detection_idx)
            else:
                # 未匹配的track保留
                current_ids[track_id] = track
        
        # 为未匹配的检测创建新track
        for i, (det_bbox, det_conf, det_class) in enumerate(detections):
            if i not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': det_bbox,
                    'age': 0,
                    'class': det_class,
                    'confidence': det_conf
                }
                current_ids[track_id] = self.tracks[track_id]
        
        return current_ids
    
    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0


# ==================== 安全帽检测服务 ====================
class HelmetDetectorService:
    """安全帽检测服务"""
    
    def __init__(self, model_path, frame_queue, result_queue, control_queue, config):
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config
        
        self.detector = None
        
        helmet_config = config.get('helmet_detection', {})
        self.conf_threshold = helmet_config.get('conf_threshold', 0.3)
        self.iou_threshold = helmet_config.get('iou_threshold', 0.3)
        self.max_age = helmet_config.get('max_age', 10)
        self.alert_duration = helmet_config.get('alert_duration', 1.5)
        self.cooldown_duration = helmet_config.get('cooldown_duration', 180)
        
        self.tracker = SimpleIOUTracker(max_age=self.max_age, iou_threshold=self.iou_threshold)
        
        self.no_helmet_timers = defaultdict(float)
        self.alerted_ids = set()  # 已触发录制的ID
        self.start_times = defaultdict(float)
        self.last_seen = defaultdict(float)
        
        self.alarm_number = 0
        self.total_alerts = 0
        self.cooldown_start_time = 0
        
        self.running = False
        self.enabled = False
        self.enabled_streams = None  # None=全部通道，set=仅处理这些 stream_id（0-based）
        self.frame_count = 0
        
        self.class_names = {
            0: 'helmet',
            1: 'no_helmet'
        }
        
        logger.info(f"HelmetDetectorService initialized:")
        logger.info(f"  - Conf threshold: {self.conf_threshold}")
        logger.info(f"  - IOU threshold: {self.iou_threshold}")
        logger.info(f"  - Alert duration: {self.alert_duration}s")
        logger.info(f"  - Cooldown: {self.cooldown_duration}s")
    
    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        logger.info("Starting helmet detector service...")
        self.setup_signal_handlers()
        
        try:
            self.detector = create_yolo_detector(
                self.model_path,
                conf_threshold=self.conf_threshold,
                device_id=self.config.get('bm1684x', {}).get('device_id', 0)
            )
            logger.info("✅ Helmet detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_helmet(self, frame):
        """
        通用 YOLO 推理包装：
        - 类别: 0=helmet(已佩戴), 1=no_helmet(未佩戴)
        - 这里只返回 0/1 两类框，具体报警逻辑在 run() 中只对 class_id==1 报警
        """
        dets = run_yolo_inference(
            self.detector,
            frame,
            conf_threshold=self.conf_threshold,
            allowed_classes=[0, 1],
        )
        # 转换为旧代码期望的 (bbox(x,y,w,h), conf, class_id)
        results = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            results.append(([x1, y1, w, h], d["confidence"], d["class_id"]))
        return results
    
    def run(self):
        logger.info("Helmet detector service running...")
        
        while self.running:
            self._check_control_commands()
            
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
            
            if not self.enabled:
                result = {
                    'detector_type': 'helmet',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],  # 用于触发录制
                    'display_alerts': [],  # 用于画面显示
                    'tracks': {},
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
            
            try:
                detections = self.detect_helmet(frame)
                current_time = time.time()
                tracked_objects = self.tracker.update(detections)
                
                # 🎯 分离两个列表
                helmet_alerts = []  # 用于触发视频录制（只有第一个）
                display_alerts = []  # 用于画面显示（所有超过3秒的）
                
                for track_id, track_info in tracked_objects.items():
                    class_id = track_info['class']
                    bbox = track_info['bbox']
                    confidence = track_info['confidence']
                    
                    self.last_seen[track_id] = current_time
                    
                    if class_id == 1:  # 不戴安全帽
                        if track_id not in self.start_times:
                            self.start_times[track_id] = current_time
                            self.no_helmet_timers[track_id] = 0.0
                        
                        elapsed = current_time - self.start_times[track_id]
                        self.no_helmet_timers[track_id] = elapsed
                        
                        # 🎯 如果超过3秒，添加到显示列表
                        if elapsed >= self.alert_duration:
                            x, y, w, h = bbox
                            display_info = {
                                'track_id': track_id,
                                'bbox': [x, y, w, h],
                                'confidence': confidence,
                                'duration': elapsed,
                                'alert_time': datetime.now(),
                                'is_recording': track_id not in self.alerted_ids  # 标记是否触发录制
                            }
                            display_alerts.append(display_info)
                            
                            # 🎯 检查是否需要触发视频录制（只有第一个）
                            if track_id not in self.alerted_ids:
                                # 报警计数器逻辑（冷却期管理）
                                if self.alarm_number == 0:
                                    # 首次报警 - 触发录制
                                    self.alerted_ids.add(track_id)
                                    self.alarm_number = 1
                                    self.cooldown_start_time = current_time
                                    self.total_alerts += 1
                                    
                                    logger.warning(f"🚨 安全帽报警触发! Track ID: {track_id}, 持续时间: {elapsed:.2f}s [触发录制]")
                                    
                                    # 添加到录制列表
                                    helmet_alerts.append({
                                        'alert_id': self.total_alerts,
                                        'track_id': track_id,
                                        'bbox': [x, y, w, h],
                                        'confidence': confidence,
                                        'duration': elapsed,
                                        'alert_time': datetime.now()
                                    })
                                else:
                                    # 冷却期内 - 不触发录制
                                    self.alarm_number += 1
                                    logger.info(f"⚠️ 冷却期内检测到不戴安全帽: Track ID: {track_id}, "
                                              f"持续时间: {elapsed:.2f}s [仅显示，不录制] (报警次数: {self.alarm_number})")
                    
                    else:
                        # 戴安全帽，重置计时器
                        if track_id in self.no_helmet_timers:
                            del self.no_helmet_timers[track_id]
                        if track_id in self.start_times:
                            del self.start_times[track_id]
                
                # 检查冷却期
                if self.alarm_number > 0 and current_time - self.cooldown_start_time >= self.cooldown_duration:
                    logger.info(f"❄️ 冷却期结束 (持续了 {self.cooldown_duration}s)")
                    self.alarm_number = 0
                
                self.cleanup_old_tracks(current_time)
                
                # 🎯 调试日志
                if display_alerts:
                    recording_ids = [a['track_id'] for a in display_alerts if a['is_recording']]
                    display_only_ids = [a['track_id'] for a in display_alerts if not a['is_recording']]
                    logger.info(f"📊 [HELMET] 显示 {len(display_alerts)} 个警报: "
                              f"录制={recording_ids}, 仅显示={display_only_ids}")
                
                # 准备输出结果
                result = {
                    'detector_type': 'helmet',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': helmet_alerts,  # 🎯 用于触发录制（只有第一个）
                    'display_alerts': display_alerts,  # 🎯 用于画面显示（所有超过3秒的）
                    'tracks': tracked_objects,
                    'no_helmet_timers': dict(self.no_helmet_timers),
                    'alarm_number': self.alarm_number,
                    'total_alerts': self.total_alerts,
                    'enabled': True
                }
                
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put(result, block=False)
                except:
                    pass
                
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames, "
                              f"Tracking: {len(tracked_objects)}, "
                              f"No helmet: {len(self.no_helmet_timers)}, "
                              f"Total alerts: {self.total_alerts}, "
                              f"Alarm number: {self.alarm_number}, "
                              f"Enabled: {self.enabled}")
                    
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
        
        logger.info("Helmet detector service stopped")
    
    def cleanup_old_tracks(self, current_time):
        expired_ids = []
        for track_id, last_seen_time in list(self.last_seen.items()):
            if current_time - last_seen_time > 0.5:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            for dict_key in [self.no_helmet_timers, self.start_times, self.last_seen]:
                if track_id in dict_key:
                    del dict_key[track_id]
            
            if track_id in self.alerted_ids:
                self.alerted_ids.remove(track_id)
    
    def _check_control_commands(self):
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                logger.info(f"Received command: {cmd}")
                if isinstance(cmd, dict):
                    if cmd.get('cmd') == 'set_streams':
                        self.enabled_streams = set(cmd.get('stream_ids') or [])
                        logger.info("✅ Helmet enabled_streams: %s", self.enabled_streams)
                    continue
                if cmd == 'stop':
                    self.stop()
                elif cmd == 'enable':
                    self.enabled = True
                    logger.info("✅ Helmet detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Helmet detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        logger.info("Stopping helmet detector service...")
        self.running = False


def run_helmet_detector(model_path, frame_queue, result_queue, control_queue, config):
    service = HelmetDetectorService(model_path, frame_queue, result_queue, control_queue, config)
    service.start()


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Helmet Detector Service')
    parser.add_argument('--model', type=str, required=True, help='Helmet detection model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_helmet_detector(args.model, frame_queue, result_queue, control_queue, config)