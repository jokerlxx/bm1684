import queue

import cv2
import numpy as np

from ingestion.decode_hub import DecodeHub, PreviewScaler, _DecoderPoolChannel, normalize_source_spec


class FakeCap:
    def __init__(self, frames):
        self._frames = list(frames)
        self.set_calls = []

    def read(self):
        if not self._frames:
            return False, None
        item = self._frames.pop(0)
        return item

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True


def test_normalize_source_spec_keeps_legacy_rtsp_entries_as_rtsp():
    spec = normalize_source_spec({"name": "通道 1", "ip": "rtsp://camera/stream"}, default_input_mode=0)

    assert spec["source"] == "rtsp://camera/stream"
    assert spec["source_type"] == "rtsp"
    assert spec["input_mode"] == 0


def test_normalize_source_spec_supports_local_video_file_entries():
    spec = normalize_source_spec({"name": "演示视频", "source_type": "file", "source": "/data/demo.mp4"}, default_input_mode=0)

    assert spec["source"] == "/data/demo.mp4"
    assert spec["source_type"] == "file"
    assert spec["input_mode"] == 1


def test_decoder_pool_channel_loops_local_video_file_with_opencv_capture():
    channel = _DecoderPoolChannel(
        {"source_type": "file", "source": "/data/demo.mp4"},
        stream_id=0,
        input_mode=0,
        fps=20,
        device_id=0,
    )
    channel.decoder = None
    channel.cap = FakeCap([(False, None), (True, "looped-frame")])

    frame = channel.read()

    assert frame == "looped-frame"
    assert channel.cap.set_calls == [(cv2.CAP_PROP_POS_FRAMES, 0)]


def test_fit_detect_frame_uses_detection_limit_before_shared_memory():
    hub = object.__new__(DecodeHub)
    hub.frame_hub = type("FrameHub", (), {"config": type("Config", (), {"max_width": 1920, "max_height": 1080})()})()
    hub.detect_frame_max_width = 960
    hub.detect_frame_max_height = 540
    hub._detect_resize_logged = set()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resized = hub._fit_detect_frame(frame, stream_id=0)

    assert resized.shape == (540, 960, 3)


def test_preview_scaler_uses_configured_target_size():
    scaler = PreviewScaler(requested_backend="cv2", target_width=1280, target_height=720)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    resized = scaler.resize(frame)

    assert resized.shape == (720, 1280, 3)


def test_decode_hub_uses_full_preview_size_for_single_stream_and_cell_size_for_multi_stream():
    single = DecodeHub(
        sources=[{"source_type": "file", "source": "/tmp/a.mp4"}],
        preview_frame_queues=[queue.Queue()],
        frame_hub=type("FrameHub", (), {"config": type("Config", (), {"max_width": 1920, "max_height": 1080})()})(),
        control_queue=queue.Queue(),
        bm_cfg={"scale_backend": "cv2", "preview_frame_width": 1280, "preview_frame_height": 720},
    )
    multi = DecodeHub(
        sources=[
            {"source_type": "file", "source": "/tmp/a.mp4"},
            {"source_type": "file", "source": "/tmp/b.mp4"},
        ],
        preview_frame_queues=[queue.Queue(), queue.Queue()],
        frame_hub=type("FrameHub", (), {"config": type("Config", (), {"max_width": 1920, "max_height": 1080})()})(),
        control_queue=queue.Queue(),
        bm_cfg={"scale_backend": "cv2", "preview_frame_width": 1280, "preview_frame_height": 720},
    )

    assert (single.preview_target_width, single.preview_target_height) == (1280, 720)
    assert (multi.preview_target_width, multi.preview_target_height) == (320, 180)
