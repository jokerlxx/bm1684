"""
YOLOv8 detect 后处理（numpy 版本）。

参考 sophon-demo/sample/YOLOv8_plus_det/python/postprocess_numpy.py，
保持 class-aware NMS 与 letterbox 反算逻辑一致，便于在无 Sophon
运行时依赖的环境中做纯 numpy 单元测试。
"""

import numpy as np


class pseudo_torch_nms:
    """Numpy 版 NMS，实现与 sample 一致的类别偏移式 NMS。"""

    def nms_boxes(self, boxes, scores, iou_thres):
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

            w = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def xywh2xyxy(self, boxes):
        """[x, y, w, h] -> [x1, y1, x2, y2]."""
        converted = boxes.copy() if isinstance(boxes, np.ndarray) else np.copy(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    def nms(self, prediction, conf_thres=0.001, iou_thres=0.5, agnostic=False, max_det=1000):
        return self.non_max_suppression(
            prediction,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=agnostic,
            multi_label=True,
            max_det=max_det,
        )

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.5,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,
    ):
        """
        对检测结果做非极大值抑制。

        Args:
            prediction: [batch, num_boxes, 4 + num_classes + nm]

        Returns:
            list[np.ndarray]: 每张图对应一个 [n, 6 + nm] 数组，格式为 [xyxy, conf, cls, ...]
        """
        del labels

        bs = prediction.shape[0]
        nc = prediction.shape[2] - nm - 4
        mi = 4 + nc
        xc = prediction[:, :, 4:mi].max(2) > conf_thres

        max_wh = 7680
        max_nms = 30000
        multi_label &= nc > 1

        output = [np.zeros((0, 6 + nm), dtype=prediction.dtype)] * bs
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            box = x[:, :4]
            cls = x[:, 4 : nc + 4]
            mask = x[:, nc + 4 :]

            box = self.xywh2xyxy(box)

            if multi_label:
                i, j = (cls > conf_thres).nonzero()
                x = np.concatenate(
                    (box[i], x[i, 4 + j, None], j[:, None].astype(np.float32), mask[i]),
                    axis=1,
                )
            else:
                conf = cls.max(1, keepdims=True)
                cls_ids = cls.argmax(1)
                cls_ids = cls_ids if cls_ids.shape == x[:, 5:].shape else np.expand_dims(cls_ids, 1)
                x = np.concatenate((box, conf, cls_ids.astype(np.float32), mask), axis=1)
                x = x[conf.reshape(-1) > conf_thres]

            if classes is not None and x.shape[0]:
                x = x[np.isin(x[:, 5].astype(np.int32), classes)]

            n = x.shape[0]
            if not n:
                continue

            x_argsort = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[x_argsort]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes = x[:, :4] + c
            scores = x[:, 4]

            keep = self.nms_boxes(boxes, scores, iou_thres)
            if keep.shape[0] > max_det:
                keep = keep[:max_det]

            output[xi] = x[keep]

        return output


class PostProcess:
    """YOLOv8 detect 的批处理后处理入口。"""

    def __init__(self, conf_thresh=0.001, nms_thresh=0.7, agnostic=False, multi_label=True, max_det=300):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.nms = pseudo_torch_nms()

    def __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """
        Args:
            preds_batch: 单输出模型时为长度 1 的 list，元素形如 [batch, num_boxes, 4 + nc]
            org_size_batch: [(orig_w, orig_h), ...]
            ratios_batch: [(ratio_x, ratio_y), ...]
            txy_batch: [(tx, ty), ...]
        """
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            detections = np.concatenate(preds_batch)
        else:
            raise NotImplementedError("Only single-output YOLOv8 detect models are supported.")

        outputs = self.nms.non_max_suppression(
            detections,
            self.conf_thresh,
            self.nms_thresh,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            multi_label=self.multi_label,
            classes=None,
        )

        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(outputs, org_size_batch, ratios_batch, txy_batch):
            if not len(det):
                continue

            coords = det[:, :4]
            coords[:, [0, 2]] = np.round((coords[:, [0, 2]] - tx1) / ratio[0])
            coords[:, [1, 3]] = np.round((coords[:, [1, 3]] - ty1) / ratio[1])

            coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)
            coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)
            det[:, :4] = coords

        return outputs
