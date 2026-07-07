import numpy as np

from core.yolov8_postprocess import PostProcess, pseudo_torch_nms


def test_xywh_to_xyxy_conversion():
    nms = pseudo_torch_nms()
    boxes = np.array([[50.0, 60.0, 20.0, 10.0]], dtype=np.float32)

    converted = nms.xywh2xyxy(boxes)

    np.testing.assert_allclose(converted, np.array([[40.0, 55.0, 60.0, 65.0]], dtype=np.float32))


def test_postprocess_rescales_letterboxed_boxes_back_to_original_image():
    postprocess = PostProcess(conf_thresh=0.25, nms_thresh=0.7, multi_label=False, max_det=300)
    preds_batch = [
        np.array(
            [[[320.0, 320.0, 64.0, 128.0, 0.9, 0.1]]],
            dtype=np.float32,
        )
    ]

    outputs = postprocess(
        preds_batch,
        org_size_batch=[(200, 100)],
        ratios_batch=[(3.2, 3.2)],
        txy_batch=[(0.0, 160.0)],
    )

    assert outputs[0].shape == (1, 6)
    np.testing.assert_allclose(outputs[0][0, :4], np.array([90.0, 30.0, 110.0, 70.0]), atol=1e-4)


def test_same_class_boxes_are_suppressed_by_nms():
    postprocess = PostProcess(conf_thresh=0.25, nms_thresh=0.5, multi_label=False, max_det=300)
    preds_batch = [
        np.array(
            [
                [
                    [50.0, 50.0, 40.0, 40.0, 0.95, 0.05],
                    [52.0, 52.0, 40.0, 40.0, 0.90, 0.10],
                ]
            ],
            dtype=np.float32,
        )
    ]

    outputs = postprocess(
        preds_batch,
        org_size_batch=[(100, 100)],
        ratios_batch=[(1.0, 1.0)],
        txy_batch=[(0.0, 0.0)],
    )

    assert outputs[0].shape[0] == 1
    assert outputs[0][0, 4] == np.max(outputs[0][:, 4])


def test_different_classes_survive_class_aware_nms():
    postprocess = PostProcess(conf_thresh=0.25, nms_thresh=0.5, multi_label=False, max_det=300)
    preds_batch = [
        np.array(
            [
                [
                    [50.0, 50.0, 40.0, 40.0, 0.95, 0.05],
                    [50.0, 50.0, 40.0, 40.0, 0.10, 0.90],
                ]
            ],
            dtype=np.float32,
        )
    ]

    outputs = postprocess(
        preds_batch,
        org_size_batch=[(100, 100)],
        ratios_batch=[(1.0, 1.0)],
        txy_batch=[(0.0, 0.0)],
    )

    assert outputs[0].shape[0] == 2
    assert set(outputs[0][:, 5].astype(int).tolist()) == {0, 1}


def test_max_det_limits_number_of_detections():
    postprocess = PostProcess(conf_thresh=0.25, nms_thresh=0.5, multi_label=False, max_det=2)
    preds_batch = [
        np.array(
            [
                [
                    [20.0, 20.0, 10.0, 10.0, 0.91, 0.05],
                    [60.0, 20.0, 10.0, 10.0, 0.90, 0.05],
                    [100.0, 20.0, 10.0, 10.0, 0.89, 0.05],
                ]
            ],
            dtype=np.float32,
        )
    ]

    outputs = postprocess(
        preds_batch,
        org_size_batch=[(200, 100)],
        ratios_batch=[(1.0, 1.0)],
        txy_batch=[(0.0, 0.0)],
    )

    assert outputs[0].shape[0] == 2
    np.testing.assert_allclose(outputs[0][:, 4], np.array([0.91, 0.90]), atol=1e-6)


def test_empty_and_low_confidence_predictions_return_empty_outputs():
    postprocess = PostProcess(conf_thresh=0.25, nms_thresh=0.5, multi_label=False, max_det=300)
    low_conf_preds = [
        np.array(
            [[[20.0, 20.0, 10.0, 10.0, 0.10, 0.05]]],
            dtype=np.float32,
        )
    ]
    empty_preds = [np.zeros((1, 0, 6), dtype=np.float32)]

    low_conf_outputs = postprocess(
        low_conf_preds,
        org_size_batch=[(100, 100)],
        ratios_batch=[(1.0, 1.0)],
        txy_batch=[(0.0, 0.0)],
    )
    empty_outputs = postprocess(
        empty_preds,
        org_size_batch=[(100, 100)],
        ratios_batch=[(1.0, 1.0)],
        txy_batch=[(0.0, 0.0)],
    )

    assert low_conf_outputs[0].shape == (0, 6)
    assert empty_outputs[0].shape == (0, 6)
