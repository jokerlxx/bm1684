"""
第二层：窗户门检测服务 (Window Door Detector Service)
功能：
1. 🎯 只对"窗户开"和"门开"两个类别进行跟踪报警（忽略窗关和门关）
2. 基于DetectionConfirmator逻辑进行状态确认
3. 观察3秒（90帧@30fps），检测率达到80%触发报警
4. 保存12秒报警视频
5. 支持恢复监控机制
6. 🎯 分离显示和录制逻辑：所有打开的窗户/门都显示，只有第一个触发录制
7. 报警类别：0=窗户开, 3=门开（窗关和门关不报警）
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime
import logging
import signal
import sys
from collections import deque, defaultdict

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [WindowDoorDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 窗户门配置 ====================
WINDOW_DOOR_CONFIG = {
    "window": {
        0: ("window_open", "窗户开", (0, 0, 255)),
        1: ("window_close", "窗户关", (0, 255, 0)),
    },
    "door": {
        2: ("door_close", "门关", (0, 255, 0)),
        3: ("door_open", "门开", (0, 0, 255)),
    }
}

# 🎯 需要报警的类别（只对窗开和门开报警）
ALERT_CLASSES = [0, 3]  # 0=窗户开, 3=门开


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
        
        for track_id, track in list(self.tracks.items()):
            track['age'] += 1
            
            if track['age'] > self.max_age:
                del self.tracks[track_id]
                continue
            
            best_iou = 0
            best_detection_idx = -1
            
            for i, (det_bbox, det_conf, det_class) in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self._calculate_iou(track['bbox'], det_bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = i
            
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
                current_ids[track_id] = track
        
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
        
        tracked_objects = []
        for track_id, track_info in current_ids.items():
            bbox = track_info['bbox']
            tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        
        return np.array(tracked_objects) if tracked_objects else np.empty((0, 5))
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IOU"""
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


# ==================== 窗户门检测确认器 ====================
class WindowDoorDetectionConfirmator:
    """窗户门检测确认器 - 避免误报"""
    
    def __init__(self, track_id, target_type, observation_frames=60, 
                 detection_threshold=0.6, fps=30):
        self.track_id = track_id
        self.target_type = target_type
        self.observation_frames = observation_frames
        self.detection_threshold = detection_threshold
        self.fps = fps
        
        self.observing = False
        self.detection_window = deque(maxlen=observation_frames)
        self.has_alerted = False
        
        self.in_post_alert_monitoring = False
        self.recovery_window_frames = int(5.0 * fps)
        self.recovery_window = deque(maxlen=self.recovery_window_frames)
        
        self.last_seen_frame = 0
    
    def update(self, frame_number, detected):
        self.last_seen_frame = frame_number
        
        if self.has_alerted and not self.in_post_alert_monitoring:
            self.in_post_alert_monitoring = True
            self.recovery_window.clear()
        
        if self.in_post_alert_monitoring:
            normal_detected = not detected
            self.recovery_window.append(normal_detected)
            
            if len(self.recovery_window) >= self.recovery_window_frames:
                normal_frames = sum(self.recovery_window)
                normal_percentage = normal_frames / len(self.recovery_window)
                
                if normal_percentage >= 0.7:
                    self.reset()
                    return False, f"{self.target_type}已恢复正常"
            
            return False, f"{self.target_type}恢复监控中"
        
        if not self.observing:
            if detected:
                self.observing = True
                self.detection_window.clear()
                self.detection_window.append(True)
                return False, f"检测到{self.target_type}，开始观察"
            else:
                return False, f"未检测到{self.target_type}"
        else:
            self.detection_window.append(detected)
            
            frames_observed = len(self.detection_window)
            detection_frames = sum(self.detection_window)
            current_detection_percentage = detection_frames / frames_observed if frames_observed > 0 else 0.0
            
            if frames_observed >= self.observation_frames:
                if current_detection_percentage >= self.detection_threshold:
                    self.has_alerted = True
                    self.observing = False
                    return True, f"{self.target_type}确认！(检测率: {current_detection_percentage:.1%})"
                else:
                    self.observing = False
                    return False, f"{self.target_type}阈值未达到(检测率: {current_detection_percentage:.1%})"
            else:
                return False, f"观察中({frames_observed}/{self.observation_frames}帧, 检测率: {current_detection_percentage:.1%})"
    
    def reset(self):
        """重置所有状态"""
        self.observing = False
        self.detection_window.clear()
        self.has_alerted = False
        self.in_post_alert_monitoring = False
        self.recovery_window.clear()
    
    def get_status(self):
        """获取当前状态信息"""
        return {
            'track_id': self.track_id,
            'target_type': self.target_type,
            'observing': self.observing,
            'has_alerted': self.has_alerted,
            'in_post_alert_monitoring': self.in_post_alert_monitoring,
            'frames_observed': len(self.detection_window),
            'detection_percentage': sum(self.detection_window) / len(self.detection_window) if len(self.detection_window) > 0 else 0.0
        }


# ==================== 窗户门检测服务 ====================
class WindowDoorDetectorService:
    """窗户门检测服务（可复用为仓内/仓外两套 detector_type）"""
    
    def __init__(self, model_path, frame_queue, result_queue, control_queue, config, detector_type="window_door"):
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config
        self.detector_type = detector_type
        
        self.detector = None
        
        window_door_config = config.get('window_door_detection', {})
        self.conf_threshold = window_door_config.get('conf_threshold', 0.5)
        self.iou_threshold = window_door_config.get('iou_threshold', 0.3)
        self.max_age = window_door_config.get('max_age', 10)
        self.observation_frames = window_door_config.get('observation_frames', 60)
        self.detection_threshold = window_door_config.get('detection_threshold', 0.6)
        self.cooldown_duration = window_door_config.get('cooldown_duration', 180)
        
        self.tracker = SimpleIOUTracker(max_age=self.max_age, iou_threshold=self.iou_threshold)
        
        self.confirmators = {}
        self.continuous_alert_targets = {}
        
        # 🎯 冷却期管理
        self.alarm_number = 0
        self.total_alerts = 0
        self.cooldown_start_time = 0
        
        self.running = False
        self.enabled = False
        self.enabled_streams = None
        self.frame_count = 0
        
        logger.info(f"WindowDoorDetectorService initialized:")
        logger.info(f"  - Conf threshold: {self.conf_threshold}")
        logger.info(f"  - IOU threshold: {self.iou_threshold}")
        logger.info(f"  - Observation frames: {self.observation_frames} ({self.observation_frames/30:.1f}s)")
        logger.info(f"  - Detection threshold: {self.detection_threshold*100:.0f}%")
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
        logger.info("Starting window door detector service...")
        self.setup_signal_handlers()
        
        try:
            self.detector = create_yolo_detector(
                self.model_path,
                conf_threshold=self.conf_threshold,
                device_id=self.config.get('bm1684x', {}).get('device_id', 0)
            )
            logger.info("✅ Window door detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_window_door(self, frame):
        """
        检测窗户门。
        模型类别：0=window_open, 1=window_close, 2=door_close, 3=door_open。
        通用推理阶段保留所有类别，后续仅对 ALERT_CLASSES(0,3) 报警。
        """
        dets = run_yolo_inference(
            self.detector,
            frame,
            conf_threshold=self.conf_threshold,
            allowed_classes=None,  # 先保留全部类别，报警时再筛选 0/3
        )
        return [(d["bbox"], d["confidence"], d["class_id"]) for d in dets]
    
    def _get_detection_type_and_config(self, cls_id):
        """根据类别ID获取检测类型和配置"""
        for detection_type, config in WINDOW_DOOR_CONFIG.items():
            if cls_id in config:
                return detection_type, config[cls_id]
        return None, None
    
    def _iou_match(self, box1, box2, threshold=0.5):
        """计算两个边界框的IoU并判断是否匹配"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold
    
    def run(self):
        """主循环"""
        logger.info("Window door detector service running...")
        
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
            current_time = time.time()
            
            if not self.enabled:
                result = {
                    'detector_type': self.detector_type,
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],
                    'display_alerts': [],
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
            
            # 执行检测
            try:
                detections = self.detect_window_door(frame)
                tracked_objects = self.tracker.update(detections)
                
                # 🎯 分离两个列表
                window_door_alerts = []  # 用于触发视频录制（只有第一个）
                display_alerts = []  # 用于画面显示（所有打开的窗户/门）
                current_tracked_ids = []
                
                for i, (bbox, conf, cls_id) in enumerate(detections):
                    # 🎯 只处理需要报警的类别（窗开=0，门开=3）
                    if cls_id not in ALERT_CLASSES:
                        continue
                    
                    detection_type, config_data = self._get_detection_type_and_config(cls_id)
                    
                    if detection_type is None:
                        continue
                    
                    label, display_name, color = config_data
                    
                    # 匹配跟踪ID
                    track_id = None
                    for trk in tracked_objects:
                        trk_box = [trk[0], trk[1], trk[2], trk[3]]
                        if self._iou_match(bbox, trk_box):
                            track_id = int(trk[4])
                            break
                    
                    if track_id is not None:
                        current_tracked_ids.append(track_id)
                    
                    # 创建或获取确认器
                    confirmator_key = f"{detection_type}_{cls_id}_{track_id}"
                    if confirmator_key not in self.confirmators:
                        self.confirmators[confirmator_key] = WindowDoorDetectionConfirmator(
                            track_id=track_id,
                            target_type=display_name,
                            observation_frames=self.observation_frames,
                            detection_threshold=self.detection_threshold,
                            fps=30
                        )
                    
                    confirmator = self.confirmators[confirmator_key]
                    confirmed, status_msg = confirmator.update(frame_number, True)
                    
                    # 🎯 如果确认器已经触发警报，添加到显示列表
                    if confirmator.has_alerted:
                        display_info = {
                            'track_id': track_id,
                            'bbox': bbox,
                            'confidence': conf,
                            'label': label,
                            'display_name': display_name,
                            'color': color,
                            'detection_type': detection_type,
                            'is_recording': confirmed  # 标记是否刚触发录制
                        }
                        display_alerts.append(display_info)
                    
                    # 🎯 如果刚确认警报，检查是否需要触发视频录制
                    if confirmed:
                        # 冷却期逻辑
                        if self.alarm_number == 0:
                            # 首次报警 - 触发录制
                            self.alarm_number = 1
                            self.cooldown_start_time = current_time
                            self.total_alerts += 1
                            
                            logger.warning(f"🚨 窗户门报警触发! Track ID: {track_id}, 类型: {display_name} [触发录制]")
                            
                            window_door_alerts.append({
                                'alert_id': self.total_alerts,
                                'track_id': track_id,
                                'bbox': bbox,
                                'confidence': conf,
                                'label': label,
                                'display_name': display_name,
                                'color': color,
                                'detection_type': detection_type,
                                'alert_time': datetime.now()
                            })
                        else:
                            # 冷却期内 - 不触发录制
                            self.alarm_number += 1
                            logger.info(f"⚠️ 冷却期内检测到{display_name}: Track ID: {track_id} "
                                      f"[仅显示，不录制] (报警次数: {self.alarm_number})")
                        
                        # 添加到持续警报目标
                        self.continuous_alert_targets[confirmator_key] = {
                            'bbox': bbox,
                            'display_name': display_name,
                            'conf': conf,
                            'color': color,
                            'detection_type': detection_type,
                            'track_id': track_id
                        }
                
                # 检查冷却期
                if self.alarm_number > 0 and current_time - self.cooldown_start_time >= self.cooldown_duration:
                    logger.info(f"❄️ 冷却期结束 (持续了 {self.cooldown_duration}s)")
                    self.alarm_number = 0
                
                # 清理不活跃的确认器
                self._cleanup_inactive_confirmators(frame_number, current_tracked_ids)
                
                # 🎯 调试日志
                if display_alerts:
                    recording_ids = [a['track_id'] for a in display_alerts if a['is_recording']]
                    display_only_ids = [a['track_id'] for a in display_alerts if not a['is_recording']]
                    logger.info(f"📊 [WINDOW/DOOR] 显示 {len(display_alerts)} 个警报: "
                              f"录制={recording_ids}, 仅显示={display_only_ids}")
                
                result = {
                    'detector_type': self.detector_type,
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': window_door_alerts,  # 🎯 用于触发录制（只有第一个）
                    'display_alerts': display_alerts,  # 🎯 用于画面显示（所有打开的）
                    'tracks': {
                        track_id: {
                            'bbox': self.tracker.tracks[track_id]['bbox'],
                            'class': self.tracker.tracks[track_id]['class'],
                            'confidence': self.tracker.tracks[track_id]['confidence']
                        } for track_id in self.tracker.tracks.keys()
                    },
                    'continuous_alerts': self.continuous_alert_targets,
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
                              f"Confirmators: {len(self.confirmators)}, "
                              f"Total alerts: {self.total_alerts}, "
                              f"Alarm number: {self.alarm_number}, "
                              f"Enabled: {self.enabled}")
                    
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
        
        logger.info("Window door detector service stopped")
    
    def _cleanup_inactive_confirmators(self, frame_number, current_tracked_ids, inactive_threshold=60):
        """清理不活跃的确认器"""
        to_remove = []
        
        for confirmator_key, confirmator in self.confirmators.items():
            if frame_number - confirmator.last_seen_frame > inactive_threshold:
                to_remove.append(confirmator_key)
        
        for key in to_remove:
            logger.info(f"清理不活跃目标: {key}")
            del self.confirmators[key]
            if key in self.continuous_alert_targets:
                del self.continuous_alert_targets[key]
    
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
                    logger.info("✅ Window door detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Window door detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        """停止检测服务"""
        logger.info("Stopping window door detector service...")
        self.running = False


def run_window_door_inside_detector(model_path, frame_queue, result_queue, control_queue, config):
    """进程入口函数：门窗（仓内）"""
    service = WindowDoorDetectorService(
        model_path, frame_queue, result_queue, control_queue, config, detector_type="window_door_inside"
    )
    service.start()


def run_window_door_outside_detector(model_path, frame_queue, result_queue, control_queue, config):
    """进程入口函数：门窗（仓外）"""
    service = WindowDoorDetectorService(
        model_path, frame_queue, result_queue, control_queue, config, detector_type="window_door_outside"
    )
    service.start()


# 兼容旧名称（如外部脚本仍调用），默认映射到仓内
def run_window_door_detector(model_path, frame_queue, result_queue, control_queue, config):
    return run_window_door_inside_detector(model_path, frame_queue, result_queue, control_queue, config)


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Window Door Detector Service')
    parser.add_argument('--model', type=str, required=True, help='Window door detection model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_window_door_detector(args.model, frame_queue, result_queue, control_queue, config)