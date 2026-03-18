"""
第二层：跌倒检测服务 (Fall Detector Service) - 基于时间窗口版本
改进内容：
1. 使用5秒时间窗口而不是固定帧数
2. 百分比基于时间窗口内实际处理的帧数计算
3. 不受帧率波动影响
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
from filterpy.kalman import KalmanFilter

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector, run_yolo_inference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [FallDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== SORT跟踪器（使用filterpy）====================
class KalmanBoxTracker:
    """使用卡尔曼滤波器跟踪单个边界框（filterpy版本）"""
    
    count = 0
    
    def __init__(self, bbox):
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # 状态转移矩阵
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # 观测矩阵
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # 测量噪声协方差
        self.kf.R[2:,2:] *= 10.
        # 初始状态协方差
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        # 过程噪声协方差
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # 初始化状态
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """用观测到的边界框更新状态"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
    
    def predict(self):
        """预测下一个状态"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """返回当前边界框估计"""
        return self.convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def convert_bbox_to_z(bbox):
        """将边界框 [x1,y1,x2,y2] 转换为观测向量 [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h  # 面积
        r = w / float(h)  # 宽高比
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def convert_x_to_bbox(x):
        """将状态向量 [x,y,s,r] 转换回边界框 [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w/2., 
            x[1] - h/2., 
            x[0] + w/2., 
            x[1] + h/2.
        ]).reshape((1, 4))


class Sort:
    """SORT跟踪器"""
    
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets):
        """更新跟踪器"""
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
        
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """批量计算IoU"""
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / (
            (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) 
            - wh
        )
        return o
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """使用匈牙利算法匹配检测和跟踪器"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = self.iou_batch(detections, trackers)
        
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
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
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


# ==================== 🕐 基于时间的人员跟踪器 ====================
class PersonTrackerTimeBased:
    """基于时间窗口的人员跟踪器"""
    
    def __init__(self, tracker_id, observation_duration=5.0, fall_threshold=0.8):
        """
        Args:
            tracker_id: 跟踪器ID
            observation_duration: 观察时长（秒）
            fall_threshold: 跌倒阈值（百分比）
        """
        self.tracker_id = tracker_id
        self.observation_duration = observation_duration  # 秒
        self.fall_threshold = fall_threshold
        
        # 🕐 使用时间戳记录每一帧的跌倒状态
        self.fall_observations = deque()  # [(timestamp, is_fallen), ...]
        
        # 状态
        self.observing_fall = False
        self.has_alerted = False
        self.in_post_alert_monitoring = False
        self.alert_time = None
        
        # 当前统计
        self.current_fall_percentage = 0.0
        
        # 最佳警报帧
        self.alert_frames = deque(maxlen=100)
        
        logger.info(f"👤 PersonTracker {tracker_id} 初始化: "
                   f"观察时长={observation_duration}秒, "
                   f"跌倒阈值={fall_threshold:.0%}")
    
    def _cleanup_old_observations(self, current_time):
        """清理超过时间窗口的旧观察"""
        cutoff_time = current_time - timedelta(seconds=self.observation_duration)
        
        # 移除超过时间窗口的观察
        while self.fall_observations and self.fall_observations[0][0] < cutoff_time:
            self.fall_observations.popleft()
    
    def update(self, is_fallen, confidence, frame, frame_number, timestamp):
        """
        更新跟踪状态（基于时间窗口）
        
        Args:
            is_fallen: 当前帧是否检测到跌倒
            confidence: 置信度
            frame: 当前帧
            frame_number: 帧号
            timestamp: 时间戳
        
        Returns:
            should_alert: 是否应该触发警报
        """
        # 清理旧观察
        self._cleanup_old_observations(timestamp)
        
        # 添加新观察
        self.fall_observations.append((timestamp, is_fallen))
        
        # 计算时间窗口内的跌倒百分比
        if len(self.fall_observations) > 0:
            fall_count = sum(1 for _, fallen in self.fall_observations if fallen)
            total_count = len(self.fall_observations)
            self.current_fall_percentage = fall_count / total_count
        else:
            self.current_fall_percentage = 0.0
        
        # 保存可能的警报帧
        if is_fallen:
            self.alert_frames.append({
                'frame': frame.copy(),
                'confidence': confidence,
                'frame_number': frame_number,
                'timestamp': timestamp
            })
        
        # 状态机逻辑
        should_alert = False
        
        if not self.has_alerted and not self.in_post_alert_monitoring:
            # 计算观察时长
            if len(self.fall_observations) > 1:
                earliest_time = self.fall_observations[0][0]
                observation_span = (timestamp - earliest_time).total_seconds()
            else:
                observation_span = 0.0
            
            # 🎯 正确逻辑：需要接近目标观察时长且百分比达标
            # 由于清理机制会删除超过5秒的数据，所以窗口最多接近5秒
            # 要求至少4.5秒的观察数据（接近5秒，留0.5秒buffer）
            min_duration = self.observation_duration - 0.5  # 5.0 - 0.5 = 4.5秒
            min_frames = int(min_duration * 25)  # 约112帧（假设25fps）
            
            # 需要有足够长的观察时间
            if observation_span >= min_duration or total_count >= min_frames:
                # 检查是否达到跌倒阈值
                if self.current_fall_percentage >= self.fall_threshold:
                    self.observing_fall = True
                    should_alert = True
                    self.observing_fall = False
                    logger.info(f"🚨 Track {self.tracker_id}: 触发警报! "
                              f"跌倒率={self.current_fall_percentage:.1%}, "
                              f"观察窗口={observation_span:.1f}秒 (要求>={min_duration:.1f}秒), "
                              f"帧数={total_count}")
                else:
                    self.observing_fall = False
            else:
                # 还在积累数据中
                self.observing_fall = False
        
        return should_alert
    
    def mark_alerted(self, timestamp):
        """标记已警报"""
        self.has_alerted = True
        self.alert_time = timestamp
        self.in_post_alert_monitoring = True
    
    def select_best_alert_frame(self):
        """选择最佳警报帧"""
        if not self.alert_frames:
            return None, 0.0, 0
        
        # 选择置信度最高的帧
        best_frame_data = max(self.alert_frames, key=lambda x: x['confidence'])
        return (
            best_frame_data['frame'],
            best_frame_data['confidence'],
            best_frame_data['frame_number']
        )
    
    def get_stats(self):
        """获取统计信息"""
        if self.fall_observations:
            earliest_time = self.fall_observations[0][0]
            latest_time = self.fall_observations[-1][0]
            observation_span = (latest_time - earliest_time).total_seconds()
        else:
            observation_span = 0.0
        
        return {
            'tracker_id': self.tracker_id,
            'fall_percentage': self.current_fall_percentage,
            'observation_span': observation_span,
            'observation_count': len(self.fall_observations),
            'observing': self.observing_fall,
            'has_alerted': self.has_alerted
        }


# ==================== 跌倒检测服务 ====================
class FallDetectorService:
    """跌倒检测服务 - 基于时间窗口版本"""
    
    def __init__(self, model_path, frame_queue, result_queue, control_queue, config):
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config
        
        # 模型
        self.detector = None
        
        # 从配置获取参数
        fall_config = config.get('fall_detection', {})
        self.conf_threshold = fall_config.get('conf_threshold', 0.5)
        self.observation_duration = fall_config.get('observation_duration', 5.0)  # 🕐 5秒
        self.fall_threshold = fall_config.get('fall_threshold', 0.8)  # 80%
        self.cooldown_duration = fall_config.get('cooldown_duration', 180)
        self.large_bbox_conf = fall_config.get('large_bbox_conf', 0.65)
        self.small_bbox_conf = fall_config.get('small_bbox_conf', 0.30)
        
        # SORT跟踪器
        self.tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
        
        # 人员跟踪器（使用时间窗口）
        self.person_trackers = {}
        
        # 冷却期管理
        self.alarm_number = 0
        self.cooldown_start_time = 0
        
        # 统计
        self.total_alerts = 0
        self.adaptive_stats = {'large_count': 0, 'small_count': 0}
        
        # 运行状态
        self.running = False
        self.enabled = False
        self.enabled_streams = None  # None=全部通道，set=仅处理这些 stream_id（0-based）
        self.frame_count = 0
        
        logger.info(f"FallDetectorService initialized (Time-Based):")
        logger.info(f"  - 🕐 观察时长: {self.observation_duration}秒")
        logger.info(f"  - 跌倒阈值: {self.fall_threshold:.0%}")
        logger.info(f"  - 冷却时长: {self.cooldown_duration}秒")
    
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
        logger.info("Starting fall detector service...")
        self.setup_signal_handlers()
        
        # 初始化模型
        try:
            self.detector = create_yolo_detector(
                self.model_path,
                conf_threshold=self.conf_threshold,
                device_id=self.config.get('bm1684x', {}).get('device_id', 0)
            )
            logger.info("✅ Fall detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_fall_with_tracking(self, frame):
        """
        检测跌倒并进行跟踪。
        模型类别：0=fall(跌倒), 1=normal(正常)，仅对 class_id=0 作为跌倒候选。
        """
        raw_dets = run_yolo_inference(
            self.detector,
            frame,
            conf_threshold=min(self.large_bbox_conf, self.small_bbox_conf),
            allowed_classes=[0, 1],
        )

        fall_detections = []
        for d in raw_dets:
            x1, y1, x2, y2 = d["bbox"]
            confidence = d["confidence"]
            class_id = d["class_id"]

            # 仅 class_id=0 视作跌倒类别；normal(1) 仅用于“无跌倒”的情况
            if class_id != 0:
                continue

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area > 50000:
                threshold = self.large_bbox_conf
                self.adaptive_stats["large_count"] += 1
            else:
                threshold = self.small_bbox_conf
                self.adaptive_stats["small_count"] += 1

            if confidence >= threshold:
                fall_detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(confidence),
                        "class_id": class_id,
                    }
                )

        # SORT跟踪（跟踪的是所有跌倒候选框）
        if len(fall_detections) > 0:
            dets = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']] 
                           for d in fall_detections])
            tracked_objects = self.tracker.update(dets)
        else:
            tracked_objects = self.tracker.update(np.empty((0, 5)))
        
        return tracked_objects, fall_detections
    
    def match_falls_to_tracks(self, tracked_objects, fall_detections):
        """匹配跌倒检测到跟踪对象"""
        track_fall_map = {}
        
        for track in tracked_objects:
            track_bbox = track[:4]
            track_id = int(track[4])
            
            best_iou = 0
            best_detection = None
            
            for detection in fall_detections:
                det_bbox = detection['bbox']
                iou_value = self._calculate_iou(track_bbox, det_bbox)
                
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_detection = detection
            
            if best_iou > 0.3 and best_detection:
                track_fall_map[track_id] = {
                    'bbox': best_detection['bbox'],
                    'confidence': best_detection['confidence'],
                    'iou': best_iou
                }
        
        return track_fall_map
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = inter_area / (bbox1_area + bbox2_area - inter_area)
        return iou
    
    def update_person_trackers(self, tracked_objects, track_fall_map, frame, frame_number, timestamp):
        """更新人员跟踪器（基于时间）"""
        alerts_to_trigger = []
        
        # 清理不再存在的跟踪器
        active_track_ids = set(int(track[4]) for track in tracked_objects)
        to_remove = [tid for tid in self.person_trackers.keys() if tid not in active_track_ids]
        for tid in to_remove:
            del self.person_trackers[tid]
        
        # 更新跟踪器
        for track in tracked_objects:
            track_id = int(track[4])
            
            # 创建新跟踪器（如果需要）
            if track_id not in self.person_trackers:
                self.person_trackers[track_id] = PersonTrackerTimeBased(
                    track_id,
                    observation_duration=self.observation_duration,
                    fall_threshold=self.fall_threshold
                )
            
            # 更新状态
            person_tracker = self.person_trackers[track_id]
            is_fallen = track_id in track_fall_map
            confidence = track_fall_map[track_id]['confidence'] if is_fallen else 0.0
            
            should_alert = person_tracker.update(
                is_fallen, confidence, frame, frame_number, timestamp
            )
            
            if should_alert:
                alerts_to_trigger.append(track_id)
        
        return alerts_to_trigger
    
    def run(self):
        """主循环"""
        logger.info("Fall detector service running...")
        
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
            
            # 推理采样：每 N 帧推理一次，降低 BM1684x 负载；展示层用最近结果
            subsample = self.config.get('detection_inference_subsample', 1)
            if subsample > 1 and (frame_number % subsample != 0):
                continue
            
            self.frame_count += 1
            current_time = time.time()
            
            if not self.enabled:
                result = {
                    'detector_type': 'fall',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],
                    'display_alerts': [],
                    'all_tracks': [],
                    'num_persons': 0,
                    'num_falls': 0,
                    'total_alerts': self.total_alerts,
                    'adaptive_stats': self.adaptive_stats.copy(),
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
            
            # 执行检测和跟踪
            try:
                tracked_objects, fall_detections = self.detect_fall_with_tracking(frame)
                track_fall_map = self.match_falls_to_tracks(tracked_objects, fall_detections)
                alerts_to_trigger = self.update_person_trackers(
                    tracked_objects, track_fall_map, frame, frame_number, timestamp
                )
                
                # 分离两个列表
                fall_alerts = []
                display_alerts = []
                
                # 收集所有达到警报条件的人员
                for track_id, person_tracker in self.person_trackers.items():
                    if person_tracker.observing_fall or person_tracker.has_alerted or person_tracker.in_post_alert_monitoring:
                        bbox = None
                        for track in tracked_objects:
                            if int(track[4]) == track_id:
                                bbox = track[:4].tolist()
                                break
                        
                        if bbox is None:
                            continue
                        
                        # 获取统计信息
                        stats = person_tracker.get_stats()
                        
                        # 添加到显示列表
                        display_info = {
                            'tracker_id': track_id,
                            'bbox': bbox,
                            'is_fallen': True,
                            'fall_percentage': person_tracker.current_fall_percentage,
                            'observation_span': stats['observation_span'],
                            'observation_count': stats['observation_count'],
                            'confidence': track_fall_map[track_id]['confidence'] if track_id in track_fall_map else 0.0,
                            'is_recording': track_id in alerts_to_trigger
                        }
                        display_alerts.append(display_info)
                
                # 处理触发警报的人员（只有第一个）
                for track_id in alerts_to_trigger:
                    person_tracker = self.person_trackers[track_id]
                    best_frame, best_confidence, best_frame_number = person_tracker.select_best_alert_frame()
                    
                    bbox = None
                    for track in tracked_objects:
                        if int(track[4]) == track_id:
                            bbox = track[:4].tolist()
                            break
                    
                    if bbox is None:
                        continue
                    
                    # 冷却期逻辑（只录制第一个）
                    if self.alarm_number == 0:
                        # 首次报警 - 触发录制
                        person_tracker.mark_alerted(timestamp)
                        self.alarm_number = 1
                        self.cooldown_start_time = current_time
                        self.total_alerts += 1
                        
                        stats = person_tracker.get_stats()
                        
                        fall_alerts.append({
                            'tracker_id': track_id,
                            'bbox': bbox,
                            'is_fallen': True,
                            'fall_percentage': person_tracker.current_fall_percentage,
                            'observation_span': stats['observation_span'],
                            'observation_count': stats['observation_count'],
                            'confidence': best_confidence,
                            'alert_id': self.total_alerts
                        })
                        
                        logger.warning(f"🚨 警报 #{self.total_alerts} [人员ID#{track_id}]: "
                                     f"跌倒率{person_tracker.current_fall_percentage:.1%}, "
                                     f"观察窗口={stats['observation_span']:.1f}秒, "
                                     f"帧数={stats['observation_count']}, "
                                     f"置信度{best_confidence:.3f} [触发录制]")
                    else:
                        # 冷却期内 - 不触发录制
                        person_tracker.mark_alerted(timestamp)
                        self.alarm_number += 1
                        
                        logger.info(f"⚠️ 冷却期内检测到跌倒: Track ID: {track_id}, "
                                  f"跌倒率{person_tracker.current_fall_percentage:.1%} "
                                  f"[仅显示，不录制] (报警次数: {self.alarm_number})")
                
                # 检查冷却期
                if self.alarm_number > 0 and current_time - self.cooldown_start_time >= self.cooldown_duration:
                    logger.info(f"❄️ 冷却期结束 (持续了 {self.cooldown_duration}s)")
                    self.alarm_number = 0
                
                # 准备所有人员的跟踪信息
                all_tracks = []
                for track in tracked_objects:
                    track_id = int(track[4])
                    person_tracker = self.person_trackers.get(track_id)
                    
                    track_info = {
                        'tracker_id': track_id,
                        'bbox': track[:4].tolist(),
                        'is_fallen': track_id in track_fall_map,
                        'confidence': track_fall_map[track_id]['confidence'] if track_id in track_fall_map else 0.0,
                    }
                    
                    if person_tracker:
                        track_info['has_alerted'] = person_tracker.has_alerted
                        track_info['observing'] = person_tracker.observing_fall
                        track_info['monitoring'] = person_tracker.in_post_alert_monitoring
                        track_info['fall_percentage'] = person_tracker.current_fall_percentage
                        stats = person_tracker.get_stats()
                        track_info['observation_span'] = stats['observation_span']
                        track_info['observation_count'] = stats['observation_count']
                    
                    all_tracks.append(track_info)
                
                result = {
                    'detector_type': 'fall',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': fall_alerts,
                    'display_alerts': display_alerts,
                    'all_tracks': all_tracks,
                    'num_persons': len(tracked_objects),
                    'num_falls': len(fall_alerts),
                    'total_alerts': self.total_alerts,
                    'adaptive_stats': self.adaptive_stats.copy(),
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
                              f"Tracking {len(tracked_objects)} persons, "
                              f"{len(fall_alerts)} new alerts, "
                              f"Total alerts: {self.total_alerts}, "
                              f"Enabled: {self.enabled}")
                    
            except Exception as e:
                logger.error(f"Detection error: {e}", exc_info=True)
        
        logger.info("Fall detector service stopped")
    
    def _check_control_commands(self):
        """检查控制命令"""
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                logger.info(f"Received command: {cmd}")
                if isinstance(cmd, dict):
                    if cmd.get('cmd') == 'set_streams':
                        self.enabled_streams = set(cmd.get('stream_ids') or [])
                        logger.info("✅ Fall enabled_streams: %s", self.enabled_streams)
                    continue
                if cmd == 'stop':
                    self.stop()
                elif cmd == 'enable':
                    self.enabled = True
                    logger.info("✅ Fall detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Fall detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        """停止检测服务"""
        logger.info("Stopping fall detector service...")
        self.running = False


def run_fall_detector(model_path, frame_queue, result_queue, control_queue, config):
    """进程入口函数"""
    service = FallDetectorService(model_path, frame_queue, result_queue, control_queue, config)
    service.start()


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Fall Detector Service (Time-Based)')
    parser.add_argument('--model', type=str, required=True, help='YOLO model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_fall_detector(args.model, frame_queue, result_queue, control_queue, config)