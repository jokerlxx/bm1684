"""
第二层：人员聚集检测服务 (Crowd Detector Service) - 优化版
主要改进：
1. 增加滑动窗口缓冲机制，避免单帧波动
2. 使用更宽松的DBSCAN参数
3. 增加时序平滑逻辑
4. 改进日志输出，方便调试
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
from sklearn.cluster import DBSCAN

# 使用BM1684X适配器
from core.bm1684x_yolo_adapter import create_yolo_detector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [CrowdDetector] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 人员聚集检测服务 ====================
class CrowdDetectorService:
    """人员聚集检测服务 - 优化版"""
    
    def __init__(self, model_path, frame_queue, result_queue, control_queue, config):
        """
        Args:
            model_path: 人员检测模型路径
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
        crowd_config = config.get('crowd_detection', {})
        self.person_conf = crowd_config.get('person_conf', 0.5)
        
        # 🆕 调整DBSCAN参数，使其更宽松
        self.eps_pixels = crowd_config.get('eps_pixels', 80.0)  # 从50增加到80
        self.min_samples = crowd_config.get('min_samples', 3)
        
        # 🆕 稳定性参数优化
        self.stability_duration = crowd_config.get('stability_duration', 1.0)  # 1秒
        self.cooldown_duration = crowd_config.get('cooldown_duration', 30)  # 30秒
        self.spatial_distance_threshold = crowd_config.get('spatial_distance', 150)
        
        # 帧率
        self.fps = config.get('fps', 25)
        self.stability_required_frames = int(self.fps * self.stability_duration)  # 25帧
        
        # 🆕 滑动窗口缓冲 - 关键改进
        self.detection_buffer_size = 5  # 缓冲最近5帧的检测结果
        self.detection_buffer = deque(maxlen=self.detection_buffer_size)
        self.buffer_threshold = 3  # 5帧中至少3帧检测到聚集才算稳定
        
        # 聚集检测状态
        self.crowd_detection_start_time = None
        self.crowd_confirmed = False
        self.crowd_stability_frames = 0
        self.current_crowd_start_time = None
        
        # 冷却期管理
        self.last_save_time = 0
        self.last_recorded_positions = []
        
        # 运行状态
        self.running = False
        self.enabled = False
        self.enabled_streams = None
        self.frame_count = 0
        self.total_alerts = 0
        
        # 🆕 调试统计
        self.debug_stats = {
            'total_detections': 0,
            'buffer_hits': 0,
            'buffer_misses': 0,
            'confirmed_crowds': 0,
            'interrupted_crowds': 0
        }
        
        logger.info(f"CrowdDetectorService initialized (OPTIMIZED):")
        logger.info(f"  - DBSCAN eps: {self.eps_pixels}px (宽松参数)")
        logger.info(f"  - Min samples: {self.min_samples}")
        logger.info(f"  - Detection buffer: {self.detection_buffer_size} frames")
        logger.info(f"  - Buffer threshold: {self.buffer_threshold}/{self.detection_buffer_size} frames")
        logger.info(f"  - Stability: {self.stability_duration}s ({self.stability_required_frames} frames)")
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
        logger.info("Starting crowd detector service...")
        self.setup_signal_handlers()
        
        # 初始化模型
        try:
            self.detector = create_yolo_detector(
                self.model_path,
                conf_threshold=self.person_conf,
                device_id=self.config.get('bm1684x', {}).get('device_id', 0)
            )
            logger.info("✅ Helmet detection model loaded successfully (for crowd detection)")  # 🆕 改这里
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        self.running = True
        self.run()
    
    def detect_helmets(self, frame):
        """
        检测头部（安全帽）- 用于人员聚集检测
        检测所有头部（class_id=0和1都要）
        
        Args:
            frame: 输入帧
        
        Returns:
            detections: 头部检测结果列表
        """
        try:
            # 调用模型检测
            results = self.detector(frame)
            
            detections = []
            
            # 处理YOLO结果对象
            if hasattr(results, '__iter__') and len(results) > 0:
                result = results[0]  # 取第一个结果
                
                # 检查是否有boxes属性（ultralytics YOLO格式）
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes_data = result.boxes.data  # tensor格式 [x1, y1, x2, y2, conf, cls]
                    
                    # 转换为numpy数组（如果是tensor）
                    if hasattr(boxes_data, 'cpu'):
                        boxes_data = boxes_data.cpu().numpy()
                    elif hasattr(boxes_data, 'numpy'):
                        boxes_data = boxes_data.numpy()
                    
                    for detection in boxes_data:
                        if len(detection) >= 6:
                            x1, y1, x2, y2, confidence, class_id = detection[:6]
                            class_id = int(class_id)
                            confidence = float(confidence)
                            
                            # 🆕 检测所有头部（class_id = 0 或 1 都要，不限制class_id）
                            if confidence >= self.person_conf:
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class_name': 'head'  # 🆕 改为 head
                                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return []
    
    def detect_crowd_gathering(self, detections):
        """
        使用DBSCAN算法进行人群聚集检测
        
        Args:
            detections: 人员检测结果列表
        
        Returns:
            检测结果字典
        """
        if len(detections) < self.min_samples:
            return {
                'is_crowd': False,
                'clusters': {},
                'warnings': [],
                'n_clusters': 0,
                'n_noise': 0
            }
        
        # 提取2D中心点
        centers = []
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        
        if len(centers) < self.min_samples:
            return {
                'is_crowd': False,
                'clusters': {},
                'warnings': [],
                'n_clusters': 0,
                'n_noise': 0
            }
        
        # 执行DBSCAN聚类
        centers_array = np.array(centers)
        dbscan = DBSCAN(eps=self.eps_pixels, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(centers_array)
        
        # 分析聚类结果
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # 构建结果
        clusters = {}
        warnings = []
        is_crowd = n_clusters > 0
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # 噪声点
                continue
            
            cluster_points = centers_array[cluster_labels == cluster_id]
            cluster_detections = [detections[i] for i in range(len(detections)) if cluster_labels[i] == cluster_id]
            
            if len(cluster_points) >= self.min_samples:
                # 计算聚类边界框
                cluster_bboxes = [det['bbox'] for det in cluster_detections]
                min_x = min(bbox[0] for bbox in cluster_bboxes)
                min_y = min(bbox[1] for bbox in cluster_bboxes)
                max_x = max(bbox[2] for bbox in cluster_bboxes)
                max_y = max(bbox[3] for bbox in cluster_bboxes)
                
                clusters[cluster_id] = {
                    'points': cluster_points.tolist(),
                    'detections': cluster_detections,
                    'count': len(cluster_points),
                    'bbox': [min_x, min_y, max_x, max_y]
                }
                
                warnings.append({
                    'cluster_id': int(cluster_id),
                    'count': len(cluster_points),
                    'bbox': [float(min_x), float(min_y), float(max_x), float(max_y)],
                    'center': cluster_points.mean(axis=0).tolist()
                })
        
        result = {
            'is_crowd': is_crowd,
            'clusters': clusters,
            'warnings': warnings,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        return result
    
    def is_stable_crowd_detection(self):
        """
        🆕 检查滑动窗口内的检测是否稳定
        
        Returns:
            bool: 如果buffer中有足够多的帧检测到聚集，返回True
        """
        if len(self.detection_buffer) < self.detection_buffer_size:
            # 缓冲区未满，不判断
            return False
        
        # 统计buffer中检测到聚集的帧数
        crowd_count = sum(1 for is_crowd in self.detection_buffer if is_crowd)
        
        # 判断是否达到阈值
        is_stable = crowd_count >= self.buffer_threshold
        
        # 🆕 调试统计
        if is_stable:
            self.debug_stats['buffer_hits'] += 1
        else:
            self.debug_stats['buffer_misses'] += 1
        
        return is_stable
    
    def is_new_location(self, warnings):
        """检查是否是新位置的聚集"""
        if not warnings:
            return True
        
        # 获取当前聚集的中心位置列表
        current_positions = []
        for warning in warnings:
            if 'center' in warning:
                current_positions.append(np.array(warning['center']))
        
        if not current_positions:
            return True
        
        # 如果没有任何历史记录，认为是新位置
        if not self.last_recorded_positions:
            return True
        
        # 检查当前聚集的位置是否与历史记录太近
        for current_pos in current_positions:
            for recorded_pos in self.last_recorded_positions:
                distance = np.linalg.norm(current_pos - recorded_pos)
                if distance < self.spatial_distance_threshold:
                    # 找到附近的历史记录，认为是同一位置
                    return False
        
        # 所有位置都不在历史记录附近，认为是新位置
        return True
    
    def record_positions(self, warnings):
        """记录聚集位置，用于下次比较"""
        self.last_recorded_positions = []
        for warning in warnings:
            if 'center' in warning:
                self.last_recorded_positions.append(np.array(warning['center']))
        
        # 只保留最近10个位置
        if len(self.last_recorded_positions) > 10:
            self.last_recorded_positions = self.last_recorded_positions[-10:]
    
    def reset_crowd_detection(self):
        """重置聚集检测状态"""
        self.crowd_detection_start_time = None
        self.crowd_confirmed = False
        self.crowd_stability_frames = 0
        self.current_crowd_start_time = None
        
        # 🆕 清空检测缓冲区
        self.detection_buffer.clear()
    
    def run(self):
        """主循环"""
        logger.info("Crowd detector service running...")
        
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
            current_time = time.time()
            
            # 如果检测被禁用,发送空结果
            if not self.enabled:
                result = {
                    'detector_type': 'crowd',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': [],
                    'person_detections': [],
                    'crowd_confirmed': False,
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
                # 1. 检测人员
                person_detections = self.detect_helmets(frame)
                
                # 2. 执行聚集检测
                if len(person_detections) >= self.min_samples:
                    crowd_result = self.detect_crowd_gathering(person_detections)
                else:
                    crowd_result = {
                        'is_crowd': False,
                        'clusters': {},
                        'warnings': [],
                        'n_clusters': 0,
                        'n_noise': 0
                    }
                
                # 🆕 更新检测缓冲区
                is_crowd_detected = crowd_result['is_crowd']
                self.detection_buffer.append(is_crowd_detected)
                self.debug_stats['total_detections'] += 1
                
                # 🆕 使用缓冲区判断稳定性
                is_stable_crowd = self.is_stable_crowd_detection()
                
                # 3. 处理聚集状态和稳定性检测
                crowd_alerts = []  # 用于触发录制
                display_alerts = []  # 🎯 用于画面显示（持续显示）
                
                # 🆕 改进的状态机逻辑
                if is_stable_crowd:
                    # 稳定检测到聚集（buffer中多数帧都检测到）
                    if self.crowd_detection_start_time is None:
                        # 首次稳定检测到聚集
                        self.crowd_detection_start_time = current_time
                        self.crowd_stability_frames = 1
                        self.crowd_confirmed = False
                        logger.info(f"🔍 稳定检测到聚集 (帧号: {frame_number}, buffer: {sum(self.detection_buffer)}/{len(self.detection_buffer)})")
                    else:
                        # 继续稳定检测到聚集
                        self.crowd_stability_frames += 1
                        
                        # 检查是否达到稳定性阈值
                        if not self.crowd_confirmed and self.crowd_stability_frames >= self.stability_required_frames:
                            # 确认聚集
                            self.crowd_confirmed = True
                            self.current_crowd_start_time = current_time
                            self.debug_stats['confirmed_crowds'] += 1
                            
                            logger.warning(f"🚨 聚集确认! 持续检测{self.crowd_stability_frames}帧 ({self.stability_duration}s)")
                            
                            # 检查是否需要触发警报
                            is_new_location = self.is_new_location(crowd_result['warnings'])
                            
                            if is_new_location:
                                # 新位置聚集，立即触发警报
                                logger.warning(f"📍 新位置聚集，触发警报")
                                self.total_alerts += 1
                                
                                # 准备警报信息（用于触发录制）
                                for warning in crowd_result['warnings']:
                                    crowd_alerts.append({
                                        'alert_id': self.total_alerts,
                                        'cluster_id': warning['cluster_id'],
                                        'count': warning['count'],
                                        'bbox': warning['bbox'],
                                        'center': warning['center'],
                                        'is_new_location': True
                                    })
                                
                                self.record_positions(crowd_result['warnings'])
                                self.last_save_time = current_time
                                
                            else:
                                # 同一位置聚集，检查冷却期
                                if current_time - self.last_save_time > self.cooldown_duration:
                                    logger.warning(f"⏰ 同一位置聚集，冷却期已过，触发警报")
                                    self.total_alerts += 1
                                    
                                    # 准备警报信息（用于触发录制）
                                    for warning in crowd_result['warnings']:
                                        crowd_alerts.append({
                                            'alert_id': self.total_alerts,
                                            'cluster_id': warning['cluster_id'],
                                            'count': warning['count'],
                                            'bbox': warning['bbox'],
                                            'center': warning['center'],
                                            'is_new_location': False
                                        })
                                    
                                    self.record_positions(crowd_result['warnings'])
                                    self.last_save_time = current_time
                                else:
                                    elapsed = current_time - self.last_save_time
                                    logger.info(f"⏸️ 同一位置聚集，仍在冷却中 ({elapsed:.1f}s/{self.cooldown_duration}s)")
                        
                        # 🎯 如果聚集已确认，持续添加到display_alerts（无论是否触发录制）
                        if self.crowd_confirmed:
                            for warning in crowd_result['warnings']:
                                display_alerts.append({
                                    'cluster_id': warning['cluster_id'],
                                    'count': warning['count'],
                                    'bbox': warning['bbox'],
                                    'center': warning['center'],
                                    'is_confirmed': True
                                })
                
                else:
                    # 未稳定检测到聚集（buffer中多数帧未检测到）
                    if self.crowd_detection_start_time is not None:
                        # 聚集检测中断
                        if not self.crowd_confirmed:
                            # 未确认的聚集被中断，重置状态
                            self.debug_stats['interrupted_crowds'] += 1
                            logger.info(f"❌ 聚集未确认就中断了 (持续{self.crowd_stability_frames}帧, buffer: {sum(self.detection_buffer)}/{len(self.detection_buffer)})")
                            self.reset_crowd_detection()
                        else:
                            # 已确认的聚集结束
                            crowd_duration = current_time - self.current_crowd_start_time
                            logger.info(f"✅ 聚集结束，持续时间: {crowd_duration:.1f}秒")
                            self.reset_crowd_detection()
                
                # 准备输出结果
                result = {
                    'detector_type': 'crowd',
                    'stream_id': stream_id,
                    'frame': frame.copy(),
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': crowd_alerts,  # 🎯 用于触发录制（只有第一次）
                    'display_alerts': display_alerts,  # 🎯 用于画面显示（持续显示）
                    'person_detections': person_detections,  # 所有人员检测
                    'crowd_result': crowd_result,  # 聚类结果
                    'crowd_confirmed': self.crowd_confirmed,  # 是否确认聚集
                    'crowd_stability_frames': self.crowd_stability_frames,  # 稳定性帧数
                    'detection_buffer': list(self.detection_buffer),  # 🆕 检测缓冲区
                    'is_stable_crowd': is_stable_crowd,  # 🆕 是否稳定检测
                    'total_alerts': self.total_alerts,  # 总警报次数
                    'enabled': True
                }
                
                # 输出结果
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put(result, block=False)
                except:
                    pass
                
                # 🆕 定期打印状态（包含更多调试信息）
                if self.frame_count % 100 == 0:
                    buffer_info = f"{sum(self.detection_buffer)}/{len(self.detection_buffer)}" if self.detection_buffer else "empty"
                    logger.info(f"📊 Processed {self.frame_count} frames")
                    logger.info(f"   ├─ Persons: {len(person_detections)}, Clusters: {crowd_result['n_clusters']}")
                    logger.info(f"   ├─ Buffer: {buffer_info}, Stable: {is_stable_crowd}")
                    logger.info(f"   ├─ Confirmed: {self.crowd_confirmed}, Stability frames: {self.crowd_stability_frames}")
                    logger.info(f"   ├─ Total alerts: {self.total_alerts}")
                    logger.info(f"   └─ Stats - Hits: {self.debug_stats['buffer_hits']}, "
                              f"Misses: {self.debug_stats['buffer_misses']}, "
                              f"Confirmed: {self.debug_stats['confirmed_crowds']}, "
                              f"Interrupted: {self.debug_stats['interrupted_crowds']}")
                    
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
        
        logger.info("Crowd detector service stopped")
    
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
                    logger.info("✅ Crowd detection ENABLED")
                elif cmd == 'disable':
                    self.enabled = False
                    logger.info("❌ Crowd detection DISABLED")
        except Exception as e:
            pass
    
    def stop(self):
        """停止检测服务"""
        logger.info("Stopping crowd detector service...")
        self.running = False


def run_crowd_detector(model_path, frame_queue, result_queue, control_queue, config):
    """进程入口函数"""
    service = CrowdDetectorService(model_path, frame_queue, result_queue, control_queue, config)
    service.start()


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Crowd Detector Service (Optimized)')
    parser.add_argument('--model', type=str, required=True, help='Person detection model path')
    parser.add_argument('--config', type=str, default='config_bm1684x.json', help='Config file path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    control_queue = mp.Queue()
    
    run_crowd_detector(args.model, frame_queue, result_queue, control_queue, config)