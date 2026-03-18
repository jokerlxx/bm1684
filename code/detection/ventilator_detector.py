"""
第二层：呼吸机检测服务 (Ventilator Detector Service) - 改进版
改进内容:
1. ✅ 使用安全帽检测（头部框）代替人员检测
2. ✅ 正面场景：面罩 + 头部 IoU匹配
3. ✅ 背面场景：氧气瓶 + 头部 距离匹配
4. ✅ 综合判定：面罩 OR 氧气瓶 → 佩戴成功
5. ✅ Fallback机制：无安全帽时使用人员检测
6. ✅ 保留原有10秒时间窗口逻辑
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime, timedelta
import logging
import signal
import sys
from collections import deque
from scipy.optimize import linear_sum_assignment

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [VentilatorDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 🕐 基于时间的卡尔曼跟踪器 ====================
class KalmanBoxTrackerTimeBased:
    """使用卡尔曼滤波器跟踪单个边界框 - 基于时间窗口"""
    count = 0
    
    def __init__(self, bbox, observation_duration=10.0):
        """
        Args:
            bbox: 初始边界框
            observation_duration: 观察时长（秒）
        """
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.kf.statePost = np.array([[x1], [y1], [w], [h], [0], [0], [0]], dtype=np.float32)
        
        self.time_since_update = 0
        self.id = KalmanBoxTrackerTimeBased.count
        KalmanBoxTrackerTimeBased.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # ✅ 🕐 呼吸机佩戴状态跟踪 - 使用时间窗口
        self.observation_duration = observation_duration  # 秒
        self.mask_observations = deque()  # [(timestamp, has_equipment), ...]
        
        logger.info(f"Tracker {self.id}: 🕐 观察时长={observation_duration}秒")
        
        self.pass_threshold = 0.2
        self.fail_threshold = 0.8
        
        self.check_completed = False
        self.check_passed = False
        self.alarm_triggered = False
        self.alarm_time = None
        
        # 是否已发送报警（避免重复发送）
        self.alert_sent = False
        
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.kf.correct(np.array([[x1], [y1], [w], [h]], dtype=np.float32))
        
    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        state = self.kf.statePost
        x1, y1, w, h = state[0][0], state[1][0], state[2][0], state[3][0]
        return [x1, y1, x1 + w, y1 + h]
    
    def get_state(self):
        state = self.kf.statePost
        x1, y1, w, h = state[0][0], state[1][0], state[2][0], state[3][0]
        return [x1, y1, x1 + w, y1 + h]
    
    def _cleanup_old_observations(self, current_time):
        """清理超过时间窗口的旧观察"""
        cutoff_time = current_time - timedelta(seconds=self.observation_duration)
        
        # 移除超过时间窗口的观察
        while self.mask_observations and self.mask_observations[0][0] < cutoff_time:
            self.mask_observations.popleft()
    
    def update_mask_status(self, has_equipment, timestamp):
        """
        更新设备佩戴状态（基于时间窗口）
        
        Args:
            has_equipment: 是否佩戴设备（面罩或氧气瓶）
            timestamp: 时间戳
        """
        # 清理旧观察
        self._cleanup_old_observations(timestamp)
        
        # 添加新观察
        self.mask_observations.append((timestamp, has_equipment))
        
        # 计算时间窗口内的佩戴率
        if len(self.mask_observations) > 0:
            pass_count = sum(1 for _, equipment in self.mask_observations if equipment)
            total_count = len(self.mask_observations)
            pass_rate = pass_count / total_count
            
            # 计算观察时长
            if len(self.mask_observations) > 1:
                earliest_time = self.mask_observations[0][0]
                observation_span = (timestamp - earliest_time).total_seconds()
            else:
                observation_span = 0.0
            
            # 🎯 正确逻辑：需要接近目标观察时长且百分比达标
            min_duration = self.observation_duration - 1.0  # 10.0 - 1.0 = 9.0秒
            min_frames = int(min_duration * 25)  # 约225帧（假设25fps）
            
            # 🐛 修复: 如果有足够长的观察时间且还未完成检查，进行判断
            # 原逻辑错误: elif条件与if条件重叠，导致佩戴率在阈值之间时检查永远不完成
            # 修复: 使用else确保任何情况下检查都会完成
            if (observation_span >= min_duration or total_count >= min_frames) and not self.check_completed:
                if pass_rate >= self.pass_threshold:
                    # 佩戴率 >= 20% → 通过检查
                    self.check_passed = True
                    self.check_completed = True
                    logger.info(f"Tracker {self.id}: ✅ 通过检查 "
                              f"(佩戴率: {pass_rate:.1%}, "
                              f"观察窗口: {observation_span:.1f}秒 (要求>={min_duration:.1f}秒), "
                              f"帧数: {total_count})")
                else:
                    # 佩戴率 < 20% → 失败检查并触发警报
                    self.check_passed = False
                    self.check_completed = True
                    
                    if not self.alarm_triggered:
                        self.alarm_triggered = True
                        self.alarm_time = datetime.now()
                        logger.warning(f"Tracker {self.id}: 🚨 触发警报 "
                                     f"(佩戴率: {pass_rate:.1%} < {self.pass_threshold:.1%}, "
                                     f"观察窗口: {observation_span:.1f}秒 (要求>={min_duration:.1f}秒), "
                                     f"帧数: {total_count})")
        
        return self.check_passed, self.alarm_triggered
    
    @property
    def mask_history(self):
        """兼容旧接口 - 返回佩戴状态列表"""
        return [has_equipment for _, has_equipment in self.mask_observations]
    
    def get_stats(self):
        """获取统计信息"""
        if self.mask_observations:
            earliest_time = self.mask_observations[0][0]
            latest_time = self.mask_observations[-1][0]
            observation_span = (latest_time - earliest_time).total_seconds()
            pass_count = sum(1 for _, equipment in self.mask_observations if equipment)
            total_count = len(self.mask_observations)
            pass_rate = pass_count / total_count
        else:
            observation_span = 0.0
            pass_rate = 0.0
            total_count = 0
        
        return {
            'tracker_id': self.id,
            'mask_wearing_rate': pass_rate,
            'observation_span': observation_span,
            'observation_count': total_count,
            'check_completed': self.check_completed,
            'alarm_triggered': self.alarm_triggered
        }


# ==================== 辅助函数 ====================
def iou(bb_test, bb_gt):
    """计算IoU"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """匈牙利算法匹配"""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ==================== 呼吸机检测服务 ====================
class VentilatorDetectorService:
    """呼吸机检测服务 - 改进版（使用安全帽检测）"""
    
    def __init__(self, equipment_model_path, helmet_model_path,
                 frame_queue, result_queue, control_queue, config=None):
        """
        Args:
            equipment_model_path: 设备模型路径（面罩+氧气瓶）
            helmet_model_path: 安全帽模型路径（用于头部检测）
            frame_queue: 输入帧队列
            result_queue: 输出结果队列
            control_queue: 控制命令队列
            config: 配置字典
        """
        self.equipment_model_path = equipment_model_path
        self.helmet_model_path = helmet_model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config or {}
        
        self.equipment_model = None
        self.helmet_model = None
        self.trackers = []
        
        # 从配置获取参数
        ventilator_config = self.config.get('ventilator_detection', {})
        self.equipment_conf = ventilator_config.get('equipment_conf', 0.4)
        self.observation_duration = ventilator_config.get('observation_duration', 10.0)
        self.cooldown_duration = ventilator_config.get('cooldown_duration', 180)
        
        # 🆕 新增参数
        self.mask_iou_threshold = ventilator_config.get('mask_iou_threshold', 0.15)
        self.tank_distance_coefficient = ventilator_config.get('tank_distance_coefficient', 2.0)
        self.tank_x_offset_coefficient = ventilator_config.get('tank_x_offset_coefficient', 1.0)
        
        # 冷却期管理
        self.alarm_number = 0
        self.cooldown_start_time = 0
        
        # 统计
        self.total_alerts = 0
        
        # 运行状态
        self.running = False
        self.enabled = False
        self.enabled_streams = None
        self.frame_count = 0
        
        # 统计计数器（只保留安全帽检测计数）
        self.helmet_detection_count = 0
        
        logger.info(f"VentilatorDetectorService initialized (Helmet-Only Version):")
        logger.info(f"  - 🕐 观察时长: {self.observation_duration}秒")
        logger.info(f"  - 冷却时长: {self.cooldown_duration}秒")
        logger.info(f"  - 🎯 仅使用安全帽检测（已删除人员检测fallback）")
        logger.info(f"  - 🆕 面罩IoU阈值: {self.mask_iou_threshold}")
        logger.info(f"  - 🆕 氧气瓶距离系数: {self.tank_distance_coefficient}")
    
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
        logger.info("Starting ventilator detector service (improved version)...")
        self.setup_signal_handlers()
        
        # 初始化模型
        try:
            device_id = self.config.get('bm1684x', {}).get('device_id', 0)
            
            # 加载设备检测模型
            self.equipment_model = create_yolo_detector(
                self.equipment_model_path,
                conf_threshold=self.equipment_conf,
                device_id=device_id
            )
            logger.info("✅ Equipment detection model loaded")
            
            # 🆕 加载安全帽检测模型
            self.helmet_model = create_yolo_detector(
                self.helmet_model_path,
                conf_threshold=0.3,  # 安全帽检测阈值
                device_id=device_id
            )
            logger.info("✅ Helmet detection model loaded (for head detection)")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_equipment(self, frame):
        """检测呼吸设备（面罩和氧气瓶）"""
        results = self.equipment_model(frame)
        
        masks = []
        tanks = []
        
        if hasattr(results, '__iter__') and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_data = result.boxes.data
                
                if hasattr(boxes_data, 'cpu'):
                    boxes_data = boxes_data.cpu().numpy()
                elif hasattr(boxes_data, 'numpy'):
                    boxes_data = boxes_data.numpy()
                
                fh, fw = frame.shape[:2] if frame is not None else (0, 0)
                for detection in boxes_data:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, confidence, class_id = detection[:6]
                        class_id = int(class_id)
                        confidence = float(confidence)
                        
                        if confidence >= self.equipment_conf:
                            # 防御：模型偶发输出 NaN/Inf，直接 int() 会崩
                            vals = [x1, y1, x2, y2]
                            if not all(np.isfinite(v) for v in vals):
                                continue
                            x1i, y1i, x2i, y2i = [int(v) for v in vals]
                            # 裁剪到画面范围
                            if fw > 0 and fh > 0:
                                x1i = max(0, min(fw - 1, x1i))
                                x2i = max(0, min(fw - 1, x2i))
                                y1i = max(0, min(fh - 1, y1i))
                                y2i = max(0, min(fh - 1, y2i))
                            if x2i <= x1i or y2i <= y1i:
                                continue
                            bbox = [x1i, y1i, x2i, y2i]
                            
                            if class_id == 1:  # 面罩
                                masks.append({
                                    'bbox': bbox,
                                    'confidence': confidence
                                })
                            elif class_id == 0:  # 氧气瓶
                                tanks.append({
                                    'bbox': bbox,
                                    'confidence': confidence
                                })
        
        return masks, tanks
    
    def detect_helmets(self, frame):
        """检测安全帽（头部框）"""
        if self.helmet_model is None:
            return []
        
        results = self.helmet_model(frame)
        
        helmet_boxes = []
        
        if hasattr(results, '__iter__') and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_data = result.boxes.data
                
                if hasattr(boxes_data, 'cpu'):
                    boxes_data = boxes_data.cpu().numpy()
                elif hasattr(boxes_data, 'numpy'):
                    boxes_data = boxes_data.numpy()
                
                fh, fw = frame.shape[:2] if frame is not None else (0, 0)
                for detection in boxes_data:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, confidence, class_id = detection[:6]
                        class_id = int(class_id)
                        confidence = float(confidence)
                        
                        # 只要检测到头部（不管是否戴安全帽，class 0或1都要）
                        # 因为我们需要的是头部位置，不是判断是否戴安全帽
                        if confidence >= 0.3:
                            vals = [x1, y1, x2, y2]
                            if not all(np.isfinite(v) for v in vals):
                                continue
                            x1i, y1i, x2i, y2i = [int(v) for v in vals]
                            if fw > 0 and fh > 0:
                                x1i = max(0, min(fw - 1, x1i))
                                x2i = max(0, min(fw - 1, x2i))
                                y1i = max(0, min(fh - 1, y1i))
                                y2i = max(0, min(fh - 1, y2i))
                            if x2i <= x1i or y2i <= y1i:
                                continue
                            helmet_boxes.append([x1i, y1i, x2i, y2i])
        
        return helmet_boxes
    
    def _calculate_iou(self, box1, box2):
        """🆕 计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def check_equipment_wearing(self, head_bbox, masks, tanks):
        """
        🆕 检查是否佩戴呼吸设备（基于头部框）
        
        逻辑：
        - 正面场景：面罩 + 头部 IoU匹配
        - 背面场景：氧气瓶 + 头部 距离匹配
        - 综合判定：面罩 OR 氧气瓶 → 佩戴成功
        
        Args:
            head_bbox: 头部边界框 [x1, y1, x2, y2]
            masks: 面罩列表 [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
            tanks: 氧气瓶列表 [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        
        Returns:
            has_equipment: 是否佩戴设备（面罩或氧气瓶）
            match_info: 匹配详情（用于调试）
        """
        hx1, hy1, hx2, hy2 = head_bbox
        helmet_center_x = (hx1 + hx2) / 2
        helmet_center_y = (hy1 + hy2) / 2
        helmet_width = hx2 - hx1
        helmet_height = hy2 - hy1
        
        # ========== 方法1: 检查面罩（正面） ==========
        has_mask = False
        best_mask_iou = 0
        matched_mask = None
        
        for mask in masks:
            mx1, my1, mx2, my2 = mask['bbox']
            
            # 计算IoU
            iou_value = self._calculate_iou(head_bbox, mask['bbox'])
            
            if iou_value > best_mask_iou:
                best_mask_iou = iou_value
                matched_mask = mask
            
            # 阈值判断
            if iou_value > self.mask_iou_threshold:
                has_mask = True
                # logger.debug(f"✅ 面罩匹配成功: IoU={iou_value:.3f}")
                break
        
        # ========== 方法2: 检查氧气瓶（背面） ==========
        has_tank = False
        best_tank_distance = float('inf')
        matched_tank = None
        
        for tank in tanks:
            tx1, ty1, tx2, ty2 = tank['bbox']
            tank_center_x = (tx1 + tx2) / 2
            tank_center_y = (ty1 + ty2) / 2
            
            # 计算欧氏距离
            distance = np.sqrt(
                (tank_center_x - helmet_center_x)**2 + 
                (tank_center_y - helmet_center_y)**2
            )
            
            # 动态阈值：基于头部高度
            max_distance = self.tank_distance_coefficient * helmet_height
            
            # X方向偏差约束
            x_offset = abs(tank_center_x - helmet_center_x)
            max_x_offset = self.tank_x_offset_coefficient * helmet_width
            
            # 位置约束：氧气瓶必须在头部下方（背面场景）
            is_below = tank_center_y > helmet_center_y
            
            if distance < best_tank_distance:
                best_tank_distance = distance
                matched_tank = tank
            
            # 综合判断
            if (distance < max_distance and 
                x_offset < max_x_offset and 
                is_below):
                has_tank = True
                # logger.debug(f"✅ 氧气瓶匹配成功: distance={distance:.1f}px "
                #            f"(max={max_distance:.1f}px), "
                #            f"x_offset={x_offset:.1f}px (max={max_x_offset:.1f}px)")
                break
        
        # ========== 综合判定 ==========
        has_equipment = has_mask or has_tank
        
        # 构建调试信息
        match_info = {
            'has_mask': has_mask,
            'has_tank': has_tank,
            'has_equipment': has_equipment,
            'mask_iou': best_mask_iou,
            'tank_distance': best_tank_distance if best_tank_distance != float('inf') else None,
            'head_bbox': head_bbox,
            'matched_mask': matched_mask,
            'matched_tank': matched_tank
        }
        
        return has_equipment, match_info
    
    def update_trackers(self, head_boxes, masks, tanks, timestamp):
        """
        更新跟踪器（基于安全帽检测）
        
        Args:
            head_boxes: 头部边界框列表（来自安全帽检测）
            masks: 面罩列表
            tanks: 氧气瓶列表
            timestamp: 时间戳
        """
        # 预测现有跟踪器位置
        if len(self.trackers) > 0:
            trk_boxes = np.array([t.predict() for t in self.trackers])
        else:
            trk_boxes = np.empty((0, 4))
        
        # 准备检测框
        if len(head_boxes) > 0:
            det_boxes = np.array(head_boxes)
        else:
            det_boxes = np.empty((0, 4))
        
        # 匈牙利算法匹配
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, trk_boxes, iou_threshold=0.3
        )
        
        # ========== 更新匹配的跟踪器 ==========
        for m in matched:
            det_idx = m[0]
            trk_idx = m[1]
            
            head_bbox = head_boxes[det_idx]
            self.trackers[trk_idx].update(head_bbox)
            
            # 🆕 使用新的设备检查方法
            has_equipment, match_info = self.check_equipment_wearing(
                head_bbox, masks, tanks
            )
            
            # 更新跟踪器状态
            self.trackers[trk_idx].update_mask_status(has_equipment, timestamp)
            
            # 详细日志（每30帧打印一次）
            if self.frame_count % 30 == 0:
                logger.debug(f"Tracker {self.trackers[trk_idx].id}: "
                           f"has_mask={match_info['has_mask']}, "
                           f"has_tank={match_info['has_tank']}, "
                           f"has_equipment={has_equipment}, "
                           f"mask_iou={match_info['mask_iou']:.3f}")
        
        # ========== 创建新跟踪器 ==========
        for i in unmatched_dets:
            head_bbox = head_boxes[i]
            
            trk = KalmanBoxTrackerTimeBased(
                head_bbox,
                observation_duration=self.observation_duration
            )
            
            # 🆕 检查设备佩戴状态
            has_equipment, match_info = self.check_equipment_wearing(
                head_bbox, masks, tanks
            )
            trk.update_mask_status(has_equipment, timestamp)
            
            self.trackers.append(trk)
            
            logger.info(f"🆕 新跟踪器 {trk.id}: "
                      f"has_equipment={has_equipment}, "
                      f"mask={match_info['has_mask']}, "
                      f"tank={match_info['has_tank']}")
        
        # ========== 删除旧跟踪器 ==========
        self.trackers = [t for t in self.trackers if t.time_since_update < 10]
    
    def run(self):
        """主循环"""
        logger.info("Ventilator detector service running (improved version)...")
        
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
                    'detector_type': 'ventilator',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],
                    'display_alerts': [],
                    'masks': [],
                    'tanks': [],
                    'persons': [],
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
                # 1. 检测设备（面罩+氧气瓶）
                masks, tanks = self.detect_equipment(frame)
                
                # 2. 🆕 检测头部（仅使用安全帽检测）
                head_boxes = self.detect_helmets(frame)
                
                # 统计检测结果
                if len(head_boxes) > 0:
                    self.helmet_detection_count += 1
                
                # 4. 更新跟踪器（传入masks和tanks）
                self.update_trackers(head_boxes, masks, tanks, timestamp)
                
                # 分离两个列表
                alerts = []
                display_alerts = []
                
                for trk in self.trackers:
                    # 如果检查完成且确认未佩戴
                    if trk.alarm_triggered and trk.check_completed:
                        stats = trk.get_stats()
                        
                        # 添加到显示列表
                        display_info = {
                            'tracker_id': trk.id,
                            'bbox': trk.get_state(),
                            'has_mask': False,
                            'mask_wearing_rate': stats['mask_wearing_rate'],
                            'observation_span': stats['observation_span'],
                            'observation_count': stats['observation_count'],
                            'is_recording': not trk.alert_sent
                        }
                        display_alerts.append(display_info)
                        
                        # 检查是否需要触发视频录制（只有第一个）
                        if not trk.alert_sent:
                            # 冷却期逻辑
                            if self.alarm_number == 0:
                                # 首次报警 - 触发录制
                                self.alarm_number = 1
                                self.cooldown_start_time = current_time
                                self.total_alerts += 1
                                
                                alerts.append({
                                    'tracker_id': trk.id,
                                    'bbox': trk.get_state(),
                                    'has_mask': False,
                                    'mask_wearing_rate': stats['mask_wearing_rate'],
                                    'observation_span': stats['observation_span'],
                                    'observation_count': stats['observation_count'],
                                    'alert_id': self.total_alerts,
                                    'alarm_time': trk.alarm_time
                                })
                                
                                trk.alert_sent = True
                                
                                logger.warning(f"🚨 Ventilator alert #{self.total_alerts}: "
                                            f"Person {trk.id} not wearing ventilator properly "
                                            f"(佩戴率: {stats['mask_wearing_rate']:.1%}, "
                                            f"观察窗口: {stats['observation_span']:.1f}秒, "
                                            f"帧数: {stats['observation_count']}) [触发录制]")
                            else:
                                # 冷却期内 - 不触发录制
                                self.alarm_number += 1
                                trk.alert_sent = True
                                
                                logger.info(f"⚠️ 冷却期内检测到未佩戴呼吸机: Tracker ID: {trk.id} "
                                          f"[仅显示，不录制] (报警次数: {self.alarm_number})")
                
                # 检查冷却期
                if self.alarm_number > 0 and current_time - self.cooldown_start_time >= self.cooldown_duration:
                    logger.info(f"❄️ 冷却期结束 (持续了 {self.cooldown_duration}s)")
                    self.alarm_number = 0
                
                # 准备所有跟踪器信息
                all_persons = []
                for trk in self.trackers:
                    stats = trk.get_stats()
                    all_persons.append({
                        'tracker_id': trk.id,
                        'bbox': trk.get_state(),
                        'mask_wearing_rate': stats['mask_wearing_rate'],
                        'observation_span': stats['observation_span'],
                        'observation_count': stats['observation_count'],
                        'check_completed': trk.check_completed,
                        'check_passed': trk.check_passed,
                        'alarm_triggered': trk.alarm_triggered
                    })
                
                result = {
                    'detector_type': 'ventilator',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': alerts,
                    'display_alerts': display_alerts,
                    'masks': masks,
                    'tanks': tanks,
                    'persons': all_persons,
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
                            f"{len(masks)} masks, {len(tanks)} tanks, "
                            f"{len(head_boxes)} helmet detections, "
                            f"{len(self.trackers)} trackers, "
                            f"Total alerts: {self.total_alerts}, "
                            f"Enabled: {self.enabled}")
                    
            except Exception as e:
                logger.error(f"Detection error: {e}", exc_info=True)
        
        logger.info("Ventilator detector service stopped")
    
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
                    logger.info("✅ Ventilator detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Ventilator detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        """停止检测服务"""
        logger.info("Stopping ventilator detector service...")
        self.running = False


def run_ventilator_detector(equipment_model_path, helmet_model_path,
                            frame_queue, result_queue, control_queue, config=None):
    """进程入口函数"""
    service = VentilatorDetectorService(
        equipment_model_path,
        helmet_model_path,
        frame_queue,
        result_queue,
        control_queue,
        config
    )
    service.start()


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Ventilator Detector Service (Helmet-Only)')
    parser.add_argument('--equipment', type=str, required=True, help='Equipment model path')
    parser.add_argument('--helmet', type=str, required=True, help='Helmet model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_ventilator_detector(args.equipment, args.helmet,
                           frame_queue, result_queue, control_queue, config)