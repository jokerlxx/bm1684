"""
BM1684X YOLO 推理适配器 - 增强版
用于在BM1684X盒子上使用bmodel格式的模型进行推理
增强功能：更清晰地显示模型输入尺寸信息
"""

import numpy as np
import cv2
import logging
import os
import time

from app.model_runtime.model_type_registry import ensure_supported_model_type, parse_model_type, resolve_model_type

logger = logging.getLogger(__name__)

try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
    logger.info("Sophon SAIL library imported successfully")
except ImportError:
    SOPHON_AVAILABLE = False
    logger.warning("Sophon SAIL library not available, falling back to dummy mode")


class BM1684X_YOLO:
    """BM1684X YOLO推理器 - 使用bmodel"""
    
    def __init__(self, bmodel_path, device_id=0, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化BM1684X YOLO推理器
        
        Args:
            bmodel_path: bmodel模型文件路径
            device_id: TPU设备ID（默认0）
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
        """
        self.bmodel_path = bmodel_path
        self.device_id = device_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if not SOPHON_AVAILABLE:
            logger.error("Sophon SAIL library is not available!")
            raise ImportError("Sophon SAIL library is required for BM1684X")
        
        # 初始化推理引擎与 I/O 信息
        self.engine = None
        self.graph_name = None
        self.input_name = None
        self.output_names = []
        self.input_shape = None
        self.input_dtype = None
        self.input_scale = None
        self.batch_size = None
        self.input_h = None
        self.input_w = None
        # 设备侧资源（Bmcv + BMImage + Tensor）
        self.handle = None
        self.bmcv = None
        self.img_dtype = None
        # 预处理配置（与 yolov8_bmcv 保持一致风格）
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = None
        # 输出 Tensor（SYSO 模式）
        self.output_tensors = {}
        # 推理耗时统计
        self._timing_count = 0
        self._timing_log_warmup = 5
        self._timing_log_interval = 30
        self._timing_sum_ms = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }
        self._last_timing_ms = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
            "batch_size": 1,
        }
        
        self._load_model()
        
    def _load_model(self):
        """加载bmodel模型"""
        try:
            print("=" * 80)
            logger.info(f"Loading bmodel from: {self.bmodel_path}")
            
            # 创建Engine（使用 SYSO 模式，使输入输出都在设备侧 Tensor 中）
            self.engine = sail.Engine(self.bmodel_path, self.device_id, sail.IOMode.SYSO)
            logger.info(f"Engine created for device {self.device_id} (IOMode.SYSO)")

            # 初始化设备句柄与 Bmcv（必须先于 Tensor 分配；否则 handle 会是 None）
            self.handle = sail.Handle(self.device_id)
            self.bmcv = sail.Bmcv(self.handle)
            
            # 获取模型信息
            self.graph_name = self.engine.get_graph_names()[0]
            logger.info(f"Graph name: {self.graph_name}")
            
            # 获取输入信息
            self.input_name = self.engine.get_input_names(self.graph_name)[0]
            self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
            self.input_dtype = self.engine.get_input_dtype(self.graph_name, self.input_name)
            try:
                self.input_scale = self.engine.get_input_scale(self.graph_name, self.input_name)
            except Exception:
                self.input_scale = 1.0
            
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Input dtype: {self.input_dtype}")
            logger.info(f"Input scale: {self.input_scale}")
            
            # 获取输出信息
            self.output_names = self.engine.get_output_names(self.graph_name)
            logger.info(f"Output names: {self.output_names}")
            
            for output_name in self.output_names:
                output_shape = self.engine.get_output_shape(self.graph_name, output_name)
                logger.info(f"Output {output_name} shape: {output_shape}")
                # SYSO 模式下预先分配输出 Tensor
                output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
                # 关键：输出 Tensor 必须绑定有效 handle，并拥有 device buffer
                # 使用构造签名：Tensor(handle, shape, dtype, own_sys_data, own_dev_data)
                # SYSO 模式下，engine.process 需要输出 Tensor 具备 system memory（否则报：
                # "Not found system memory in output tensor"）
                self.output_tensors[output_name] = sail.Tensor(
                    self.handle, output_shape, output_dtype, True, True
                )
            
            # 提取输入尺寸
            self.batch_size = self.input_shape[0]
            self.input_h = self.input_shape[2]
            self.input_w = self.input_shape[3]
            
            # ===== 增强：醒目地显示输入尺寸 =====
            print("=" * 80)
            print(f"📐 MODEL INPUT SIZE INFORMATION")
            print("=" * 80)
            print(f"  Model Path    : {self.bmodel_path}")
            print(f"  Batch Size    : {self.batch_size}")
            print(f"  Input Width   : {self.input_w} pixels")
            print(f"  Input Height  : {self.input_h} pixels")
            print(f"  Input Size    : {self.input_w}×{self.input_h}")
            print(f"  Input Shape   : {self.input_shape}")
            print(f"  Data Type     : {self.input_dtype}")
            
            # 检查是否为640x640
            if self.input_w == 640 and self.input_h == 640:
                print(f"  ✅ Standard YOLOv8 input size: 640×640")
            else:
                print(f"  ⚠️  Non-standard input size detected!")
            
            print("=" * 80)
            logger.info(f"✅ Model loaded successfully: batch={self.batch_size}, "
                       f"input_size=({self.input_w}×{self.input_h})")
            # 根据输入 dtype 推导 BMImage 数据格式
            self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
            # 预处理 scale 参数，与 yolov8_bmcv 保持一致
            self.ab = [x * self.input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]
            
        except Exception as e:
            logger.error(f"Failed to load bmodel: {e}")
            raise
    
    def get_input_size(self):
        """
        获取模型输入尺寸
        
        Returns:
            tuple: (width, height)
        """
        return (self.input_w, self.input_h)
    
    def print_model_info(self):
        """打印详细的模型信息"""
        print("\n" + "=" * 80)
        print("📊 DETAILED MODEL INFORMATION")
        print("=" * 80)
        print(f"Model Path       : {self.bmodel_path}")
        print(f"Device ID        : {self.device_id}")
        print(f"Graph Name       : {self.graph_name}")
        print(f"Input Name       : {self.input_name}")
        print(f"Input Shape      : {self.input_shape}")
        print(f"Input Size       : {self.input_w}×{self.input_h}")
        print(f"Batch Size       : {self.batch_size}")
        print(f"Input Data Type  : {self.input_dtype}")
        print(f"Output Names     : {self.output_names}")
        print(f"Conf Threshold   : {self.conf_threshold}")
        print(f"IoU Threshold    : {self.iou_threshold}")
        print("=" * 80 + "\n")

    def _record_timing(self, preprocess_ms, inference_ms, postprocess_ms, batch_size=1, force_log=False):
        total_ms = preprocess_ms + inference_ms + postprocess_ms
        self._timing_count += 1
        self._last_timing_ms = {
            "preprocess_ms": float(preprocess_ms),
            "inference_ms": float(inference_ms),
            "postprocess_ms": float(postprocess_ms),
            "total_ms": float(total_ms),
            "batch_size": int(batch_size),
        }

        self._timing_sum_ms["preprocess_ms"] += preprocess_ms
        self._timing_sum_ms["inference_ms"] += inference_ms
        self._timing_sum_ms["postprocess_ms"] += postprocess_ms
        self._timing_sum_ms["total_ms"] += total_ms

        should_log = force_log or self._timing_count <= self._timing_log_warmup
        if not should_log and self._timing_count % self._timing_log_interval == 0:
            should_log = True

        if should_log:
            avg_total_ms = self._timing_sum_ms["total_ms"] / self._timing_count
            logger.info(
                "⏱️ Detection time [%s] #%d: preprocess=%.2fms inference=%.2fms postprocess=%.2fms total=%.2fms avg_total=%.2fms",
                os.path.basename(self.bmodel_path),
                self._timing_count,
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms,
                avg_total_ms,
            )

    def get_last_timing(self):
        """返回最近一次检测耗时（毫秒）。"""
        return dict(self._last_timing_ms)
    
    def preprocess(self, image):
        """
        图像预处理（纯设备侧：Bmcv + BMImage + Tensor）
        
        Args:
            image: 输入图像 (BGR format from OpenCV)
            
        Returns:
            input_tensor: 预处理后的设备侧 Tensor
            ratio: 缩放比例
            pad_w: 宽度padding
            pad_h: 高度padding
        """
        img_h, img_w = image.shape[:2]
        
        # 1. numpy BGR -> 设备侧 BGR BMImage
        bmimg_bgr = sail.BMImage(
            self.handle, img_h, img_w,
            sail.Format.FORMAT_BGR_PACKED,
            sail.DATA_TYPE_EXT_1N_BYTE
        )
        self.bmcv.mat_to_bm_image(image, bmimg_bgr)

        # 2. BGR -> RGB_PLANAR
        bmimg_rgb = sail.BMImage(
            self.handle, img_h, img_w,
            sail.Format.FORMAT_RGB_PLANAR,
            sail.DATA_TYPE_EXT_1N_BYTE
        )
        self.bmcv.convert_format(bmimg_bgr, bmimg_rgb)

        # 3. resize + padding（letterbox）
        src_w = bmimg_rgb.width()
        src_h = bmimg_rgb.height()
        if self.use_resize_padding:
            r_w = self.input_w / src_w
            r_h = self.input_h / src_h
            r = min(r_w, r_h)
            tw = int(round(r * src_w))
            th = int(round(r * src_h))
            tx1, ty1 = self.input_w - tw, self.input_h - th
            tx1 /= 2.0
            ty1 /= 2.0

            ratio = r
            pad_w = int(round(tx1 - 0.1))
            pad_h = int(round(ty1 - 0.1))

            attr = sail.PaddingAtrr()
            attr.set_stx(pad_w)
            attr.set_sty(pad_h)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            preprocess_fn = (
                self.bmcv.vpp_crop_and_resize_padding
                if self.use_vpp else self.bmcv.crop_and_resize_padding
            )
            resized_bmimg = preprocess_fn(
                bmimg_rgb,
                0, 0, src_w, src_h,
                self.input_w, self.input_h,
                attr,
                sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR
            )
        else:
            r_w = self.input_w / src_w
            r_h = self.input_h / src_h
            ratio = min(r_w, r_h)
            pad_w = 0
            pad_h = 0
            preprocess_fn = (
                self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            )
            resized_bmimg = preprocess_fn(bmimg_rgb, self.input_w, self.input_h)

        # 4. 归一化 + scale（convert_to）
        preprocessed_bmimg = sail.BMImage(
            self.handle, self.input_h, self.input_w,
            sail.Format.FORMAT_RGB_PLANAR,
            self.img_dtype
        )
        self.bmcv.convert_to(
            resized_bmimg,
            preprocessed_bmimg,
            (
                (self.ab[0], self.ab[1]),
                (self.ab[2], self.ab[3]),
                (self.ab[4], self.ab[5]),
            ),
        )

        # 5. BMImage -> 设备侧 Tensor（batch=1）
        input_tensor = sail.Tensor(
            self.handle,
            self.input_shape,
            self.input_dtype,
            False,
            False
        )
        self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)

        return input_tensor, ratio, pad_w, pad_h
    
    def postprocess(self, outputs, ratio, pad_w, pad_h, orig_shape):
        """
        后处理输出结果
        
        Args:
            outputs: 模型输出
            ratio: 预处理时的缩放比例
            pad_w: 宽度padding
            pad_h: 高度padding
            orig_shape: 原始图像尺寸 (h, w)
            
        Returns:
            boxes: 检测框 [[x1, y1, x2, y2, conf, cls], ...]
        """
        # YOLOv8输出格式：[batch, 84, 8400] 或 [batch, num_classes+4, num_anchors]
        # 前4个是bbox，后面是类别置信度
        
        output = outputs[self.output_names[0]]
        
        # 如果是 [1, 84, 8400] 格式，需要转置
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))  # [1, 8400, 84]
        
        output = output[0]  # 去掉batch维度 [8400, 84]
        
        # 分离bbox和置信度
        boxes_xywh = output[:, :4]  # [x_center, y_center, width, height]
        class_confs = output[:, 4:]  # 类别置信度
        
        # 获取最大置信度和类别
        class_ids = np.argmax(class_confs, axis=1)
        confidences = np.max(class_confs, axis=1)
        
        # 过滤低置信度
        mask = confidences > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes_xywh) == 0:
            return []
        
        # xywh转xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        
        # 坐标转换：从模型输入尺寸转换回原始图像尺寸
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / ratio
        
        # 裁剪到图像边界
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])
        
        # NMS
        indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)
        
        # 组装结果
        results = []
        for i in indices:
            results.append([
                boxes_xyxy[i, 0],  # x1
                boxes_xyxy[i, 1],  # y1
                boxes_xyxy[i, 2],  # x2
                boxes_xyxy[i, 3],  # y2
                confidences[i],    # conf
                class_ids[i]       # cls
            ])
        
        return results
    
    def _nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def __call__(self, image, verbose=False):
        """
        推理接口（兼容ultralytics YOLO调用方式）
        
        Args:
            image: 输入图像 (numpy array, BGR format)
            verbose: 是否输出详细信息
            
        Returns:
            results: 检测结果列表
        """
        orig_shape = image.shape[:2]

        # 预处理（Bmcv + BMImage + Tensor，纯设备侧）
        preprocess_start = time.perf_counter()
        input_tensor, ratio, pad_w, pad_h = self.preprocess(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        # 推理（SYSO：显式传入输入/输出 Tensor）
        inference_start = time.perf_counter()
        input_tensors = {self.input_name: input_tensor}
        input_shapes = {self.input_name: self.input_shape}
        self.engine.process(
            self.graph_name,
            input_tensors,
            input_shapes,
            self.output_tensors
        )
        # 将输出 Tensor 转回 numpy，保持原 postprocess 接口
        outputs = {
            name: self.output_tensors[name].asnumpy()
            for name in self.output_names
        }
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        # 后处理
        postprocess_start = time.perf_counter()
        detections = self.postprocess(outputs, ratio, pad_w, pad_h, orig_shape)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        self._record_timing(
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            batch_size=1,
            force_log=bool(verbose),
        )

        # 封装成类似ultralytics的结果格式
        result = YOLOResult(detections, orig_shape)
        
        return [result]  # 返回列表以兼容ultralytics接口


class YOLOResult:
    """YOLO结果封装类，兼容ultralytics YOLO返回格式"""
    
    def __init__(self, detections, orig_shape):
        """
        Args:
            detections: 检测结果 [[x1, y1, x2, y2, conf, cls], ...]
            orig_shape: 原始图像尺寸
        """
        self.detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
        self.orig_shape = orig_shape
        self.boxes = Boxes(self.detections)
    
    def __len__(self):
        return len(self.detections)


class Boxes:
    """边界框封装类，兼容ultralytics Boxes格式"""
    
    def __init__(self, detections):
        """
        Args:
            detections: numpy array [[x1, y1, x2, y2, conf, cls], ...]
        """
        self.data = detections
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for i in range(len(self.data)):
            yield Box(self.data[i])
    
    @property
    def xyxy(self):
        """返回xyxy格式的边界框"""
        if len(self.data) == 0:
            return np.empty((0, 4))
        return self.data[:, :4]
    
    @property
    def conf(self):
        """返回置信度"""
        if len(self.data) == 0:
            return np.empty((0,))
        return self.data[:, 4]
    
    @property
    def cls(self):
        """返回类别"""
        if len(self.data) == 0:
            return np.empty((0,))
        return self.data[:, 5]


class Box:
    """单个边界框封装"""
    
    def __init__(self, data):
        """
        Args:
            data: [x1, y1, x2, y2, conf, cls]
        """
        self.data = data
        self.xyxy = [data[:4]]  # 兼容ultralytics格式
        self.conf = [data[4]]
        self.cls = [data[5]]
    
    def cpu(self):
        """返回CPU数据（已经是numpy）"""
        return self
    
    def numpy(self):
        """返回numpy格式"""
        return self.data


# ==================== 工厂函数 ====================
def create_yolo_detector(
    model_path,
    device_id=0,
    conf_threshold=0.25,
    iou_threshold=0.45,
    model_key=None,
    config=None,
    model_type=None,
):
    """
    创建YOLO检测器（按 model_types 配置决定 bmodel 推理实现）。
    
    Args:
        model_path: 模型路径
        device_id: 设备ID（仅用于bmodel）
        conf_threshold: 置信度阈值
        iou_threshold: NMS的IoU阈值
        model_key: config.models / config.model_types 中的模型 key
        config: 完整配置 dict，用于解析 model_types
        model_type: 直接指定的类型字符串，例如 yolov8_fp16
        
    Returns:
        detector: YOLO检测器实例
    """
    if model_path.endswith('.bmodel'):
        if model_type is not None:
            spec = parse_model_type(model_key or "<direct>", model_type)
        else:
            if model_key is None or config is None:
                raise ValueError(
                    "bmodel detector creation requires model_key + config or an explicit model_type"
                )
            spec = resolve_model_type(model_key, config)

        ensure_supported_model_type(spec)
        logger.info(
            "🔧 Using BM1684X detector with bmodel: %s (model_key=%s, model_type=%s)",
            model_path,
            spec.model_key,
            spec.raw,
        )

        if spec.family == "yolov8":
            from core.bm1684x_yolov8_adapter import BM1684X_YOLOv8

            detector = BM1684X_YOLOv8(model_path, device_id, conf_threshold, iou_threshold)
        elif spec.family == "yolo26":
            from core.bm1684x_yolo26_adapter import BM1684X_YOLO26

            detector = BM1684X_YOLO26(model_path, device_id, conf_threshold, iou_threshold)
        else:
            raise NotImplementedError(
                "Unsupported model type for '{}': '{}'. Supported values in this build: yolo26_int8, yolov8_fp16, yolov8_int8".format(
                    spec.model_key,
                    spec.raw,
                )
            )
        # 打印详细信息
        detector.print_model_info()
        return detector

    logger.info(f"Using Ultralytics YOLO with pt model: {model_path}")
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except ImportError:
        logger.error("Ultralytics not available and model is not bmodel format")
        raise


def run_yolo_inference(detector, frame, conf_threshold=None, allowed_classes=None):
    """
    通用 YOLO 推理工具函数。
    
    Args:
        detector: 已创建好的 YOLO 检测器（BM1684X_YOLO 或 ultralytics YOLO）
        frame: BGR 图像 (numpy array)
        conf_threshold: 置信度阈值；为 None 时由模型内部阈值决定
        allowed_classes: 允许的类别 ID 列表；为 None 时不过滤
    
    Returns:
        List[dict]: 每个检测为一个 dict：
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int
            }
    """
    try:
        results = detector(frame)
    except Exception as e:
        logger.error(f"YOLO inference error: {e}", exc_info=True)
        return []

    detections = []

    if not (hasattr(results, "__iter__") and len(results) > 0):
        return detections

    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None:
        return detections

    boxes_data = result.boxes.data
    if hasattr(boxes_data, "cpu"):
        boxes_data = boxes_data.cpu().numpy()
    elif hasattr(boxes_data, "numpy"):
        boxes_data = boxes_data.numpy()

    for det in boxes_data:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det[:6]
        conf = float(conf)
        cls_id = int(cls_id)

        if conf_threshold is not None and conf < conf_threshold:
            continue
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue

        detections.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_id": cls_id,
            }
        )

    return detections


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='BM1684X YOLO Adapter Test')
    parser.add_argument('--model', type=str, required=True, help='bmodel path')
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--device', type=int, default=0, help='Device ID')
    args = parser.parse_args()
    
    # 加载模型
    detector = BM1684X_YOLO(args.model, device_id=args.device)
    
    # 读取测试图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        exit(1)
    
    # 推理
    results = detector(image)
    
    # 显示结果
    print(f"Detected {len(results[0])} objects")
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(f"Class: {cls}, Conf: {conf:.3f}, Box: [{x1}, {y1}, {x2}, {y2}]")
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{cls}: {conf:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    output_path = args.image.replace('.', '_result.')
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")
