"""
BM1684X YOLOv8 detect 专用推理适配器。

参考 sophon-demo/sample/YOLOv8_plus_det/python/yolov8_bmcv.py，
用于接入单输出 OPT 形态的 YOLOv8 detect bmodel。
"""

import logging
import os
import time

import cv2
import numpy as np

from core.bm1684x_yolo_adapter import YOLOResult
from core.yolov8_postprocess import PostProcess

logger = logging.getLogger(__name__)

try:
    import sophon.sail as sail

    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class BM1684X_YOLOv8:
    """YOLOv8 detect 专用推理器。"""

    SUPPORTED_BATCH_SIZES = {1, 2, 3, 4, 8, 9, 16, 32, 64, 128, 256}

    def __init__(
        self,
        bmodel_path,
        device_id=0,
        conf_threshold=0.25,
        iou_threshold=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
    ):
        self.bmodel_path = bmodel_path
        self.device_id = device_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det

        if not SOPHON_AVAILABLE:
            raise ImportError("Sophon SAIL library is required for BM1684X YOLOv8 backend")

        self.engine = None
        self.handle = None
        self.bmcv = None
        self.graph_name = None
        self.input_name = None
        self.output_names = []
        self.output_tensors = {}
        self.output_scales = {}
        self.input_shape = None
        self.input_shapes = None
        self.input_dtype = None
        self.input_scale = None
        self.img_dtype = None
        self.batch_size = None
        self.net_h = None
        self.net_w = None

        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = None
        self.log_box_details = _env_flag("BM_YOLO_BOX_DETAIL", default=False)
        self.box_log_limit = max(1, _env_int("BM_YOLO_BOX_DETAIL_LIMIT", 50))
        self.debug_log_limit = max(1, _env_int("BM_YOLO_DEBUG_LIMIT", 5))
        self._debug_log_count = 0
        self._output_log_done = False
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
            "batch_size": 0,
        }

        self.postprocess = PostProcess(
            conf_thresh=self.conf_threshold,
            nms_thresh=self.iou_threshold,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self._load_model()

    def _load_model(self):
        logger.info("Loading YOLOv8 bmodel from: %s", self.bmodel_path)

        self.engine = sail.Engine(self.bmodel_path, self.device_id, sail.IOMode.SYSO)
        self.handle = sail.Handle(self.device_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.engine.get_graph_names()[0]

        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.input_dtype = self.engine.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.engine.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        self.output_names = self.engine.get_output_names(self.graph_name)
        if len(self.output_names) != 1:
            raise ValueError(
                "YOLOv8 backend only supports single-output detect models, "
                "but got {} outputs".format(len(self.output_names))
            )

        output_name = self.output_names[0]
        output_shape = self.engine.get_output_shape(self.graph_name, output_name)
        output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
        try:
            output_scale = self.engine.get_output_scale(self.graph_name, output_name)
        except Exception:
            output_scale = 1.0
        if output_shape[1] < output_shape[2]:
            raise ValueError(
                "Only support YOLOv8 OPT models with output shape [batch, num_boxes, channels], "
                "please export an OPT detect model."
            )

        self.output_tensors[output_name] = sail.Tensor(
            self.handle,
            output_shape,
            output_dtype,
            True,
            True,
        )
        self.output_scales[output_name] = output_scale

        self.batch_size = self.input_shape[0]
        if self.batch_size not in self.SUPPORTED_BATCH_SIZES:
            raise ValueError(
                "Unsupported batch_size {} for BM1684X YOLOv8 backend".format(self.batch_size)
            )

        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.ab = [x * self.input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]

        logger.info(
            "YOLOv8 backend ready: batch=%s input=%sx%s output=%s output_scale=%s",
            self.batch_size,
            self.net_w,
            self.net_h,
            output_shape,
            output_scale,
        )
        logger.info(
            "YOLOv8 tensor detail: graph=%s input_name=%s input_dtype=%s img_dtype=%s input_scale=%s output_name=%s output_dtype=%s output_scale=%s resize_padding=%s vpp=%s",
            self.graph_name,
            self.input_name,
            self.input_dtype,
            self.img_dtype,
            self.input_scale,
            output_name,
            output_dtype,
            output_scale,
            self.use_resize_padding,
            self.use_vpp,
        )

    def get_input_size(self):
        return (self.net_w, self.net_h)

    def print_model_info(self):
        logger.info(
            "YOLOv8 backend model info: path=%s graph=%s input=%s shape=%s output=%s",
            self.bmodel_path,
            self.graph_name,
            self.input_name,
            self.input_shape,
            self.output_names,
        )

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
                "⏱️ Detection time [%s] #%d: preprocess=%.2fms inference=%.2fms postprocess=%.2fms total=%.2fms avg_total=%.2fms batch=%d",
                os.path.basename(self.bmodel_path),
                self._timing_count,
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms,
                avg_total_ms,
                batch_size,
            )

    def get_last_timing(self):
        """返回最近一次检测耗时（毫秒）。"""
        return dict(self._last_timing_ms)

    def _get_aspect_scaled_ratio(self, src_w, src_h):
        """按 YOLOv8_multi_QT 的逻辑计算缩放比例。"""
        ratio_w = self.net_w / src_w
        ratio_h = self.net_h / src_h
        if ratio_h > ratio_w:
            return ratio_w, True
        return ratio_h, False

    def _resize_padding_params(self, img_w, img_h):
        """
        返回与 YOLOv8_multi_QT 一致的 letterbox 参数。

        这里使用整数 crop 尺寸和整数 tx1/ty1，保证预处理实际 padding
        与后处理回映使用的 padding 完全一致。
        """
        ratio, align_width = self._get_aspect_scaled_ratio(img_w, img_h)
        tx1 = 0
        ty1 = 0
        if align_width:
            tw = self.net_w
            th = max(1, int(img_h * ratio))
            ty1 = int((self.net_h - th) / 2)
        else:
            th = self.net_h
            tw = max(1, int(img_w * ratio))
            tx1 = int((self.net_w - tw) / 2)
        return ratio, tw, th, tx1, ty1

    def _maybe_dequantize_output(self, output_name, output):
        """仅当 asnumpy 返回整数时按 output_scale 反量化。"""
        if not isinstance(output, np.ndarray):
            output = np.asarray(output)
        if np.issubdtype(output.dtype, np.integer):
            scale = float(self.output_scales.get(output_name, 1.0))
            return output.astype(np.float32) * scale
        return output

    def _log_debug_inference_context(self, original_sizes, ratios, txy_list, detections_batch):
        if self._debug_log_count >= self.debug_log_limit:
            return

        for batch_idx, ((org_w, org_h), ratio, (tx1, ty1), dets) in enumerate(
            zip(original_sizes, ratios, txy_list, detections_batch)
        ):
            det_count = len(dets)
            if det_count > 0 and len(dets[0]) >= 6:
                x1, y1, x2, y2, conf, cls_id = dets[0][:6]
                logger.info(
                    "[YOLOv8 DEBUG] batch=%d orig=%dx%d net=%dx%d ratio=(%.6f, %.6f) pad=(%.1f, %.1f) dets=%d first_box=[%.1f, %.1f, %.1f, %.1f] first_conf=%.4f first_cls=%d",
                    batch_idx,
                    org_w,
                    org_h,
                    self.net_w,
                    self.net_h,
                    float(ratio[0]),
                    float(ratio[1]),
                    float(tx1),
                    float(ty1),
                    det_count,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    int(cls_id),
                )
            else:
                logger.info(
                    "[YOLOv8 DEBUG] batch=%d orig=%dx%d net=%dx%d ratio=(%.6f, %.6f) pad=(%.1f, %.1f) dets=0",
                    batch_idx,
                    org_w,
                    org_h,
                    self.net_w,
                    self.net_h,
                    float(ratio[0]),
                    float(ratio[1]),
                    float(tx1),
                    float(ty1),
                )

        self._debug_log_count += 1

    def _log_detection_details(self, detections_batch, images):
        """输出最终检测框详情，便于排查坐标回映是否正确。"""
        for batch_idx, (detections, image) in enumerate(zip(detections_batch, images)):
            img_h, img_w = image.shape[:2]
            det_count = len(detections)
            logger.info(
                "[YOLOv8 BOX DETAIL] batch=%d image=%dx%d detections=%d",
                batch_idx,
                img_w,
                img_h,
                det_count,
            )
            if det_count == 0:
                continue

            limit = min(det_count, self.box_log_limit)
            for det_idx in range(limit):
                det = detections[det_idx]
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = det[:6]
                logger.info(
                    "[YOLOv8 BOX DETAIL] batch=%d det=%d cls=%d conf=%.4f box=[%.1f, %.1f, %.1f, %.1f] size=[%.1f x %.1f]",
                    batch_idx,
                    det_idx,
                    int(cls_id),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(x2 - x1),
                    float(y2 - y1),
                )

            if det_count > limit:
                logger.info(
                    "[YOLOv8 BOX DETAIL] batch=%d omitted=%d (limit=%d)",
                    batch_idx,
                    det_count - limit,
                    self.box_log_limit,
                )

    def resize_bmcv(self, bmimg):
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            ratio_value, tw, th, tx1, ty1 = self._resize_padding_params(img_w, img_h)
            ratio = (ratio_value, ratio_value)
            txy = (tx1, ty1)

            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            preprocess_fn = (
                self.bmcv.vpp_crop_and_resize_padding
                if self.use_vpp
                else self.bmcv.crop_and_resize_padding
            )
            resized_img_rgb = preprocess_fn(
                bmimg,
                0,
                0,
                img_w,
                img_h,
                self.net_w,
                self.net_h,
                attr,
                sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR,
            )
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0.0, 0.0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)

        return resized_img_rgb, ratio, txy

    def preprocess_bmcv(self, image):
        img_h, img_w = image.shape[:2]

        bmimg_bgr = sail.BMImage(
            self.handle,
            img_h,
            img_w,
            sail.Format.FORMAT_BGR_PACKED,
            sail.DATA_TYPE_EXT_1N_BYTE,
        )
        self.bmcv.mat_to_bm_image(image, bmimg_bgr)

        rgb_planar_img = sail.BMImage(
            self.handle,
            img_h,
            img_w,
            sail.Format.FORMAT_RGB_PLANAR,
            sail.DATA_TYPE_EXT_1N_BYTE,
        )
        self.bmcv.convert_format(bmimg_bgr, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)

        preprocessed_bmimg = sail.BMImage(
            self.handle,
            self.net_h,
            self.net_w,
            sail.Format.FORMAT_RGB_PLANAR,
            self.img_dtype,
        )
        self.bmcv.convert_to(
            resized_img_rgb,
            preprocessed_bmimg,
            (
                (self.ab[0], self.ab[1]),
                (self.ab[2], self.ab[3]),
                (self.ab[4], self.ab[5]),
            ),
        )

        return preprocessed_bmimg, ratio, txy

    def _preprocess_images(self, images):
        original_sizes = []
        ratios = []
        txy_list = []
        preprocessed = []

        for image in images:
            ori_h, ori_w = image.shape[:2]
            original_sizes.append((ori_w, ori_h))
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(image)
            preprocessed.append(preprocessed_bmimg)
            ratios.append(ratio)
            txy_list.append(txy)

        return preprocessed, original_sizes, ratios, txy_list

    def _build_input_tensor(self, preprocessed_images):
        img_num = len(preprocessed_images)
        input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, True, True)

        if self.batch_size == 1:
            self.bmcv.bm_image_to_tensor(preprocessed_images[0], input_tensor)
            return input_tensor

        bm_image_array_type = getattr(sail, "BMImageArray{}D".format(self.batch_size), None)
        if bm_image_array_type is None:
            single_tensors = []
            for i in range(self.batch_size):
                src = preprocessed_images[i] if i < img_num else preprocessed_images[-1]
                single_tensors.append(self.bmcv.bm_image_to_tensor(src).asnumpy()[0])
            input_tensor.update_data(np.stack(single_tensors).astype(input_tensor.asnumpy().dtype, copy=False))
            input_tensor.sync_s2d()
            return input_tensor

        bmimgs = bm_image_array_type()
        for i in range(self.batch_size):
            src = preprocessed_images[i] if i < img_num else preprocessed_images[-1]
            bmimgs[i] = src.data()

        self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
        return input_tensor

    def predict(self, input_tensor, img_num):
        input_tensors = {self.input_name: input_tensor}
        self.engine.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)

        outputs_dict = {}
        for name in self.output_names:
            output = self.output_tensors[name].asnumpy()[:img_num]
            dequantized = self._maybe_dequantize_output(name, output)
            outputs_dict[name] = dequantized
            if not self._output_log_done:
                logger.info(
                    "[YOLOv8 OUTPUT] name=%s raw_dtype=%s raw_shape=%s scale=%s dequantized=%s",
                    name,
                    output.dtype,
                    output.shape,
                    self.output_scales.get(name, 1.0),
                    bool(np.issubdtype(output.dtype, np.integer)),
                )
        self._output_log_done = True

        ordered_outputs = []
        for name in self.output_names:
            for key in outputs_dict:
                if name == key or name in key:
                    ordered_outputs.append(outputs_dict[key])
                    break

        if len(ordered_outputs) != len(self.output_names):
            raise KeyError(
                "YOLOv8 output name mismatch, expect {}, got {}".format(
                    self.output_names, list(outputs_dict.keys())
                )
            )

        return ordered_outputs

    def infer(self, images):
        if not images:
            return []
        if len(images) > self.batch_size:
            raise ValueError(
                "Received {} images, but model batch_size is {}".format(
                    len(images), self.batch_size
                )
            )

        preprocess_start = time.perf_counter()
        preprocessed, original_sizes, ratios, txy_list = self._preprocess_images(images)
        input_tensor = self._build_input_tensor(preprocessed)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        outputs = self.predict(input_tensor, len(images))
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        detections = self.postprocess(outputs, original_sizes, ratios, txy_list)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        self._log_debug_inference_context(original_sizes, ratios, txy_list, detections)
        self._record_timing(
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            batch_size=len(images),
        )
        return detections

    def __call__(self, image, verbose=False):
        if isinstance(image, np.ndarray):
            images = [image]
        else:
            images = list(image)

        detections_batch = self.infer(images)
        if verbose or self.log_box_details:
            self._log_detection_details(detections_batch, images)
        if verbose and self._timing_count > 0:
            self._record_timing(
                self._last_timing_ms["preprocess_ms"],
                self._last_timing_ms["inference_ms"],
                self._last_timing_ms["postprocess_ms"],
                batch_size=self._last_timing_ms["batch_size"],
                force_log=True,
            )
        return [
            YOLOResult(detections, img.shape[:2])
            for detections, img in zip(detections_batch, images)
        ]
