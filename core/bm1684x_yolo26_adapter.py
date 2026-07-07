"""
BM1684X YOLO26 detect adapter for post-NMS outputs.

The exported bmodels used in this project follow the `test_yolo26` reference path:
- letterbox resize with the same rounding/padding behavior as `yolo26_bmcv.py`
- post-NMS outputs shaped like `[batch, 6, num_det]`, `[batch, num_det, 6]`,
  or equivalent singleton-expanded forms
"""

import logging
import os
import time

import cv2
import numpy as np

from core.bm1684x_yolo_adapter import YOLOResult

logger = logging.getLogger(__name__)

try:
    import sophon.sail as sail

    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False


class BM1684X_YOLO26:
    """YOLO26 detect backend for BM1684X post-NMS bmodels."""

    SUPPORTED_BATCH_SIZES = {1, 2, 3, 4, 8, 9, 16, 32, 64, 128, 256}

    def __init__(
        self,
        bmodel_path,
        device_id=0,
        conf_threshold=0.25,
        iou_threshold=0.45,
    ):
        del iou_threshold  # post-NMS output already encodes final detections

        self.bmodel_path = bmodel_path
        self.device_id = device_id
        self.conf_threshold = conf_threshold

        if not SOPHON_AVAILABLE:
            raise ImportError("Sophon SAIL library is required for BM1684X YOLO26 backend")

        self.engine = None
        self.handle = None
        self.bmcv = None
        self.graph_name = None
        self.input_name = None
        self.output_names = []
        self.output_tensors = {}
        self.output_scales = {}
        self.output_shapes = {}
        self.detection_output_name = None
        self.input_shape = None
        self.input_shapes = None
        self.input_dtype = None
        self.input_scale = None
        self.img_dtype = None
        self.batch_size = None
        self.net_h = None
        self.net_w = None
        self._input_tensor = None
        self._bm_image_array_type = None
        self._bm_image_array = None
        self._fallback_array_size = None
        self._fallback_bm_image_array_type = None
        self._fallback_bm_image_array = None
        self._fallback_tensor = None
        self._input_tensor_numpy_dtype = None
        self._input_tensor_numpy_buffer = None
        self._single_image_tensors = []
        self._preprocess_slots = []

        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = None
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
        self._output_log_done = False

        self._load_model()

    def _load_model(self):
        logger.info("Loading YOLO26 bmodel from: %s", self.bmodel_path)

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
        for output_name in self.output_names:
            output_shape = self.engine.get_output_shape(self.graph_name, output_name)
            output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
            try:
                output_scale = self.engine.get_output_scale(self.graph_name, output_name)
            except Exception:
                output_scale = 1.0

            self.output_tensors[output_name] = sail.Tensor(
                self.handle,
                output_shape,
                output_dtype,
                True,
                True,
            )
            self.output_scales[output_name] = output_scale
            self.output_shapes[output_name] = tuple(output_shape)

            if self.detection_output_name is None and self._looks_like_detection_output_shape(output_shape):
                self.detection_output_name = output_name

        if self.detection_output_name is None:
            raise ValueError(
                "YOLO26 backend could not find a detection output shaped like "
                "[batch, 6, num_det] / [batch, num_det, 6] (allowing singleton axes); "
                "available outputs: {}".format(self.output_shapes)
            )

        self.batch_size = self.input_shape[0]
        if self.batch_size not in self.SUPPORTED_BATCH_SIZES:
            raise ValueError(
                "Unsupported batch_size {} for BM1684X YOLO26 backend".format(self.batch_size)
            )

        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.ab = [x * self.input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]
        self._init_reusable_input_buffers()

        logger.info(
            "YOLO26 backend ready: batch=%s input=%sx%s detection_output=%s shape=%s scale=%s all_outputs=%s",
            self.batch_size,
            self.net_w,
            self.net_h,
            self.detection_output_name,
            self.output_shapes[self.detection_output_name],
            self.output_scales[self.detection_output_name],
            self.output_shapes,
        )

    def _init_reusable_input_buffers(self):
        self._input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, True, True)
        if self.batch_size <= 1:
            return

        self._bm_image_array_type = getattr(sail, "BMImageArray{}D".format(self.batch_size), None)
        if self._bm_image_array_type is not None:
            self._bm_image_array = self._bm_image_array_type()
            logger.info("YOLO26 reusable input buffers enabled: tensor + BMImageArray%dD", self.batch_size)
        else:
            for fallback_size in sorted(self.SUPPORTED_BATCH_SIZES, reverse=True):
                if fallback_size >= self.batch_size or fallback_size <= 1:
                    continue
                fallback_type = getattr(sail, "BMImageArray{}D".format(fallback_size), None)
                if fallback_type is None:
                    continue
                self._fallback_array_size = int(fallback_size)
                self._fallback_bm_image_array_type = fallback_type
                self._fallback_bm_image_array = fallback_type()
                fallback_shape = [self._fallback_array_size] + [int(dim) for dim in self.input_shape[1:]]
                self._fallback_tensor = sail.Tensor(self.handle, fallback_shape, self.input_dtype, True, True)
                logger.info(
                    "YOLO26 reusable fallback input buffers enabled: BMImageArray%dD chunks for batch=%d",
                    self._fallback_array_size,
                    self.batch_size,
                )
                break
            logger.warning(
                "YOLO26 BMImageArray%dD unavailable; falling back to reusable chunk tensor assembly",
                self.batch_size,
            )

    def _get_input_tensor(self):
        if self._input_tensor is None:
            self._input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, True, True)
        return self._input_tensor

    def _get_input_numpy_buffer(self, input_tensor):
        if self._input_tensor_numpy_dtype is None:
            self._input_tensor_numpy_dtype = input_tensor.asnumpy().dtype
        if self._input_tensor_numpy_buffer is None:
            self._input_tensor_numpy_buffer = np.empty(
                tuple(int(dim) for dim in self.input_shape),
                dtype=self._input_tensor_numpy_dtype,
            )
        return self._input_tensor_numpy_buffer

    def _get_single_image_tensor(self, slot_index):
        single_shape = [1] + [int(dim) for dim in self.input_shape[1:]]
        while len(self._single_image_tensors) <= slot_index:
            self._single_image_tensors.append(
                sail.Tensor(self.handle, single_shape, self.input_dtype, True, True)
            )
        return self._single_image_tensors[slot_index]

    def get_input_size(self):
        return (self.net_w, self.net_h)

    def print_model_info(self):
        logger.info(
            "YOLO26 backend model info: path=%s graph=%s input=%s shape=%s outputs=%s detection_output=%s",
            self.bmodel_path,
            self.graph_name,
            self.input_name,
            self.input_shape,
            self.output_names,
            self.detection_output_name,
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
            avg_preprocess_ms = self._timing_sum_ms["preprocess_ms"] / self._timing_count
            avg_inference_ms = self._timing_sum_ms["inference_ms"] / self._timing_count
            avg_postprocess_ms = self._timing_sum_ms["postprocess_ms"] / self._timing_count
            avg_total_ms = self._timing_sum_ms["total_ms"] / self._timing_count
            logger.info(
                "YOLO26 detection time [%s] #%d: preprocess=%.2fms inference=%.2fms postprocess=%.2fms total=%.2fms avg_pre=%.2fms avg_infer=%.2fms avg_post=%.2fms avg_total=%.2fms batch=%d",
                os.path.basename(self.bmodel_path),
                self._timing_count,
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms,
                avg_preprocess_ms,
                avg_inference_ms,
                avg_postprocess_ms,
                avg_total_ms,
                batch_size,
            )

    def get_last_timing(self):
        return dict(self._last_timing_ms)

    def _resize_padding_params(self, img_w, img_h):
        ratio = min(self.net_w / img_w, self.net_h / img_h)
        tw = max(1, int(round(ratio * img_w)))
        th = max(1, int(round(ratio * img_h)))
        tx1 = (self.net_w - tw) / 2.0
        ty1 = (self.net_h - th) / 2.0
        return ratio, tw, th, tx1, ty1

    @staticmethod
    def _squeeze_output_shape(shape):
        squeezed = tuple(int(dim) for dim in shape if int(dim) != 1)
        return squeezed or (1,)

    def _looks_like_detection_output_shape(self, shape):
        squeezed = self._squeeze_output_shape(shape)
        if len(squeezed) == 2:
            return 6 in squeezed
        if len(squeezed) == 3:
            return squeezed[1] == 6 or squeezed[2] == 6
        return False

    def _maybe_dequantize_output(self, output_name, output):
        if not isinstance(output, np.ndarray):
            output = np.asarray(output)
        if np.issubdtype(output.dtype, np.integer):
            scale = float(self.output_scales.get(output_name, 1.0))
            return output.astype(np.float32) * scale
        return output

    def resize_bmcv(self, bmimg):
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            ratio_value, tw, th, tx1, ty1 = self._resize_padding_params(img_w, img_h)
            ratio = (ratio_value, ratio_value)
            txy = (tx1, ty1)

            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
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

    def _get_preprocess_slot(self, slot_index, img_h, img_w):
        while len(self._preprocess_slots) <= slot_index:
            self._preprocess_slots.append({})

        slot = self._preprocess_slots[slot_index]
        if slot.get("src_shape") != (img_h, img_w):
            slot["src_shape"] = (img_h, img_w)
            slot["bgr"] = sail.BMImage(
                self.handle,
                img_h,
                img_w,
                sail.Format.FORMAT_BGR_PACKED,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
            slot["rgb"] = sail.BMImage(
                self.handle,
                img_h,
                img_w,
                sail.Format.FORMAT_RGB_PLANAR,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
        if slot.get("preprocessed") is None:
            slot["preprocessed"] = sail.BMImage(
                self.handle,
                self.net_h,
                self.net_w,
                sail.Format.FORMAT_RGB_PLANAR,
                self.img_dtype,
            )
        return slot

    def preprocess_bmcv(self, image, slot_index=None):
        img_h, img_w = image.shape[:2]

        if slot_index is None:
            bmimg_bgr = sail.BMImage(
                self.handle,
                img_h,
                img_w,
                sail.Format.FORMAT_BGR_PACKED,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
            rgb_planar_img = sail.BMImage(
                self.handle,
                img_h,
                img_w,
                sail.Format.FORMAT_RGB_PLANAR,
                sail.DATA_TYPE_EXT_1N_BYTE,
            )
            preprocessed_bmimg = sail.BMImage(
                self.handle,
                self.net_h,
                self.net_w,
                sail.Format.FORMAT_RGB_PLANAR,
                self.img_dtype,
            )
        else:
            slot = self._get_preprocess_slot(int(slot_index), img_h, img_w)
            bmimg_bgr = slot["bgr"]
            rgb_planar_img = slot["rgb"]
            preprocessed_bmimg = slot["preprocessed"]

        self.bmcv.mat_to_bm_image(image, bmimg_bgr)
        self.bmcv.convert_format(bmimg_bgr, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)

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
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(image, slot_index=len(preprocessed))
            preprocessed.append(preprocessed_bmimg)
            ratios.append(ratio)
            txy_list.append(txy)

        return preprocessed, original_sizes, ratios, txy_list

    def _build_input_tensor(self, preprocessed_images):
        img_num = len(preprocessed_images)
        input_tensor = self._get_input_tensor()

        if self.batch_size == 1:
            self.bmcv.bm_image_to_tensor(preprocessed_images[0], input_tensor)
            return input_tensor

        if self._bm_image_array is None:
            input_buffer = self._get_input_numpy_buffer(input_tensor)
            offset = 0
            if self._fallback_bm_image_array is not None and self._fallback_tensor is not None:
                chunk_size = int(self._fallback_array_size)
                while offset + chunk_size <= self.batch_size:
                    for j in range(chunk_size):
                        src_index = offset + j
                        src = preprocessed_images[src_index] if src_index < img_num else preprocessed_images[-1]
                        self._fallback_bm_image_array[j] = src.data()
                    self.bmcv.bm_image_to_tensor(self._fallback_bm_image_array, self._fallback_tensor)
                    input_buffer[offset : offset + chunk_size] = self._fallback_tensor.asnumpy()
                    offset += chunk_size
            for i in range(offset, self.batch_size):
                src = preprocessed_images[i] if i < img_num else preprocessed_images[-1]
                single_tensor = self._get_single_image_tensor(i)
                self.bmcv.bm_image_to_tensor(src, single_tensor)
                input_buffer[i] = single_tensor.asnumpy()[0]
            input_tensor.update_data(input_buffer)
            input_tensor.sync_s2d()
            return input_tensor

        for i in range(self.batch_size):
            src = preprocessed_images[i] if i < img_num else preprocessed_images[-1]
            self._bm_image_array[i] = src.data()

        self.bmcv.bm_image_to_tensor(self._bm_image_array, input_tensor)
        return input_tensor

    def _normalize_output_layout(self, output):
        output = np.asarray(output)
        output = np.squeeze(output)

        if output.ndim == 2:
            if output.shape[-1] == 6:
                return output[np.newaxis, ...]
            if output.shape[0] == 6:
                return output.T[np.newaxis, ...]
            raise ValueError(
                "YOLO26 backend expects a 2D detection tensor with one axis sized 6, got {}".format(output.shape)
            )

        if output.ndim != 3:
            raise ValueError("YOLO26 backend expects a 3D tensor, got shape {}".format(output.shape))
        if output.shape[-1] == 6:
            return output
        if output.shape[1] == 6:
            return np.transpose(output, (0, 2, 1))
        raise ValueError(
            "YOLO26 backend expects output shape [batch, 6, num_det] or [batch, num_det, 6], "
            "got {}".format(output.shape)
        )

    def predict(self, input_tensor, img_num):
        input_tensors = {self.input_name: input_tensor}
        self.engine.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)

        output_name = self.detection_output_name
        output = self.output_tensors[output_name].asnumpy()
        if output.ndim >= 3 or (output.ndim >= 1 and output.shape[0] == self.batch_size):
            output = output[:img_num]
        normalized = self._normalize_output_layout(self._maybe_dequantize_output(output_name, output))
        if not self._output_log_done:
            logger.info(
                "YOLO26 output detail: name=%s raw_shape=%s normalized_shape=%s scale=%s",
                output_name,
                output.shape,
                normalized.shape,
                self.output_scales.get(output_name, 1.0),
            )
            self._output_log_done = True
        return normalized

    def postprocess(self, detections, original_sizes, ratios, txy_list):
        outputs = []
        for dets, (org_w, org_h), ratio, (tx1, ty1) in zip(detections, original_sizes, ratios, txy_list):
            dets = np.asarray(dets)
            if dets.size == 0:
                outputs.append([])
                continue

            dets = dets[dets[:, 4] > self.conf_threshold]
            if dets.size == 0:
                outputs.append([])
                continue

            dets = dets.copy()
            dets[:, [0, 2]] = (dets[:, [0, 2]] - tx1) / ratio[0]
            dets[:, [1, 3]] = (dets[:, [1, 3]] - ty1) / ratio[1]
            dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, org_w - 1)
            dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, org_h - 1)
            outputs.append(dets.tolist())

        return outputs

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
        detections = self.predict(input_tensor, len(images))
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        outputs = self.postprocess(detections, original_sizes, ratios, txy_list)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        self._record_timing(
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            batch_size=len(images),
        )
        return outputs

    def __call__(self, image, verbose=False):
        if isinstance(image, np.ndarray):
            images = [image]
        else:
            images = list(image)

        detections_batch = self.infer(images)
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
