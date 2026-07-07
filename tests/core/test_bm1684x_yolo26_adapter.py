import numpy as np
import pytest

from core.bm1684x_yolo26_adapter import BM1684X_YOLO26


@pytest.fixture
def adapter():
    return object.__new__(BM1684X_YOLO26)


def test_resize_padding_params_match_reference_rounding(adapter):
    adapter.net_w = 320
    adapter.net_h = 320

    ratio, tw, th, tx1, ty1 = adapter._resize_padding_params(1920, 1080)

    assert ratio == pytest.approx(320.0 / 1920.0)
    assert tw == 320
    assert th == 180
    assert tx1 == pytest.approx(0.0)
    assert ty1 == pytest.approx(70.0)


def test_detection_shape_probe_accepts_singleton_expanded_layouts(adapter):
    assert adapter._looks_like_detection_output_shape((1, 6, 300))
    assert adapter._looks_like_detection_output_shape((1, 300, 6))
    assert adapter._looks_like_detection_output_shape((1, 1, 300, 6))
    assert adapter._looks_like_detection_output_shape((1, 300, 6, 1))
    assert not adapter._looks_like_detection_output_shape((1, 84, 8400))


def test_normalize_output_layout_handles_reference_and_singleton_forms(adapter):
    canonical = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    cases = [
        canonical,
        np.transpose(canonical, (0, 2, 1)),
        canonical[0],
        canonical[0].T,
        np.expand_dims(np.transpose(canonical, (0, 2, 1)), axis=1),
        canonical[..., np.newaxis],
    ]

    for output in cases:
        normalized = adapter._normalize_output_layout(output)
        assert normalized.shape == canonical.shape
        assert np.array_equal(normalized, canonical)


def test_normalize_output_layout_rejects_non_detection_shapes(adapter):
    with pytest.raises(ValueError, match="YOLO26 backend expects"):
        adapter._normalize_output_layout(np.zeros((1, 300, 7), dtype=np.float32))
