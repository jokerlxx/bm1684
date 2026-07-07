"""
多进程最新帧中心。

目标：
1. 预览链路继续用独立 mp.Queue(maxsize=1) 传递缩小后的帧；
2. 检测链路改为从共享内存中的“每路最新帧”主动拉取，避免完整大图在多个进程间重复排队。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import shared_memory
from typing import Dict, Iterable, List, Optional

import numpy as np


logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))

_META_CONNECTED = 0
_META_GENERATION = 1
_META_FRAME_NUMBER = 2
_META_WIDTH = 3
_META_HEIGHT = 4
_META_CHANNELS = 5
_META_BYTES = 6
_META_SOURCE_WIDTH = 7
_META_SOURCE_HEIGHT = 8
_META_FIELD_COUNT = 9


def _to_timestamp(value) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    try:
        return float(value)
    except Exception:
        return time.time()


def _from_timestamp(value) -> datetime:
    return datetime.fromtimestamp(float(value), tz=BEIJING_TZ)


@dataclass
class FrameHubConfig:
    stream_count: int = 9
    max_width: int = 1920
    max_height: int = 1080
    channels: int = 3

    @property
    def slot_bytes(self) -> int:
        return int(self.max_width) * int(self.max_height) * int(self.channels)


class LatestFrameHub:
    """
    为每一路维护一个固定大小的共享内存槽位。

    元数据通过 Manager.dict 共享，图像数据通过 shared_memory 共享。
    """

    def __init__(self, config: Optional[FrameHubConfig] = None, manager=None):
        self.config = config or FrameHubConfig()
        self.stream_count = int(self.config.stream_count)
        self.slot_bytes = int(self.config.slot_bytes)

        self._locks = [mp.Lock() for _ in range(self.stream_count)]
        self._meta_ints = mp.Array(
            ctypes.c_longlong,
            self.stream_count * _META_FIELD_COUNT,
            lock=False,
        )
        self._meta_timestamps = mp.Array(
            ctypes.c_double,
            self.stream_count,
            lock=False,
        )
        self._shm_names: List[str] = []
        self._local_shm: Dict[int, shared_memory.SharedMemory] = {}

        for stream_id in range(self.stream_count):
            shm = shared_memory.SharedMemory(create=True, size=self.slot_bytes)
            self._shm_names.append(shm.name)
            shm.close()
            base = self._meta_offset(stream_id)
            self._meta_ints[base + _META_CHANNELS] = int(self.config.channels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_local_shm"] = {}
        return state

    def _attach(self, stream_id: int) -> shared_memory.SharedMemory:
        stream_id = int(stream_id)
        shm = self._local_shm.get(stream_id)
        if shm is None:
            shm = shared_memory.SharedMemory(name=self._shm_names[stream_id])
            self._local_shm[stream_id] = shm
        return shm

    def _meta_offset(self, stream_id: int) -> int:
        return int(stream_id) * _META_FIELD_COUNT

    def _meta_snapshot_locked(self, stream_id: int) -> dict:
        base = self._meta_offset(stream_id)
        return {
            "stream_id": int(stream_id),
            "generation": int(self._meta_ints[base + _META_GENERATION]),
            "connected": bool(self._meta_ints[base + _META_CONNECTED]),
            "frame_number": int(self._meta_ints[base + _META_FRAME_NUMBER]),
            "timestamp": float(self._meta_timestamps[stream_id]),
            "width": int(self._meta_ints[base + _META_WIDTH]),
            "height": int(self._meta_ints[base + _META_HEIGHT]),
            "channels": int(self._meta_ints[base + _META_CHANNELS]),
            "bytes": int(self._meta_ints[base + _META_BYTES]),
            "source_width": int(self._meta_ints[base + _META_SOURCE_WIDTH]),
            "source_height": int(self._meta_ints[base + _META_SOURCE_HEIGHT]),
        }

    def _generation(self, stream_id: int) -> int:
        base = self._meta_offset(stream_id)
        return int(self._meta_ints[base + _META_GENERATION])

    def _coerce_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(frame)
        if arr.ndim != 3:
            raise ValueError(f"unsupported frame ndim: {arr.ndim}")
        if arr.shape[2] != self.config.channels:
            raise ValueError(f"unsupported frame channels: {arr.shape[2]}")
        needed = int(arr.nbytes)
        if needed > self.slot_bytes:
            raise ValueError(
                f"frame too large for slot ({arr.shape}, {needed} > {self.slot_bytes})"
            )
        return arr

    def write_frame(
        self,
        stream_id: int,
        frame: np.ndarray,
        frame_number: int,
        timestamp,
        *,
        connected: bool = True,
        source_size=None,
    ) -> None:
        stream_id = int(stream_id)
        arr = self._coerce_frame(frame)
        timestamp_value = _to_timestamp(timestamp)
        src_w, src_h = (0, 0)
        if source_size and len(source_size) >= 2:
            src_w = int(source_size[0])
            src_h = int(source_size[1])
        base = self._meta_offset(stream_id)
        with self._locks[stream_id]:
            shm = self._attach(stream_id)
            shm.buf[: arr.nbytes] = memoryview(arr).cast("B")
            self._meta_ints[base + _META_BYTES] = int(arr.nbytes)
            self._meta_ints[base + _META_WIDTH] = int(arr.shape[1])
            self._meta_ints[base + _META_HEIGHT] = int(arr.shape[0])
            self._meta_ints[base + _META_CHANNELS] = int(arr.shape[2])
            self._meta_ints[base + _META_FRAME_NUMBER] = int(frame_number)
            self._meta_timestamps[stream_id] = float(timestamp_value)
            self._meta_ints[base + _META_CONNECTED] = 1 if connected else 0
            self._meta_ints[base + _META_SOURCE_WIDTH] = int(src_w or arr.shape[1])
            self._meta_ints[base + _META_SOURCE_HEIGHT] = int(src_h or arr.shape[0])
            self._meta_ints[base + _META_GENERATION] += 1

    def mark_disconnected(self, stream_id: int) -> None:
        stream_id = int(stream_id)
        base = self._meta_offset(stream_id)
        with self._locks[stream_id]:
            self._meta_ints[base + _META_CONNECTED] = 0
            self._meta_ints[base + _META_GENERATION] += 1

    def clear_stream(self, stream_id: int) -> None:
        stream_id = int(stream_id)
        base = self._meta_offset(stream_id)
        with self._locks[stream_id]:
            self._meta_ints[base + _META_CONNECTED] = 0
            self._meta_ints[base + _META_WIDTH] = 0
            self._meta_ints[base + _META_HEIGHT] = 0
            self._meta_ints[base + _META_BYTES] = 0
            self._meta_ints[base + _META_FRAME_NUMBER] = 0
            self._meta_timestamps[stream_id] = 0.0
            self._meta_ints[base + _META_SOURCE_WIDTH] = 0
            self._meta_ints[base + _META_SOURCE_HEIGHT] = 0
            self._meta_ints[base + _META_GENERATION] += 1

    def read_frame(self, stream_id: int) -> Optional[dict]:
        stream_id = int(stream_id)
        with self._locks[stream_id]:
            meta = self._meta_snapshot_locked(stream_id)
            if not meta["connected"]:
                return None
            width = int(meta["width"])
            height = int(meta["height"])
            channels = int(meta["channels"] or self.config.channels)
            byte_count = int(meta["bytes"])
            if width <= 0 or height <= 0 or channels <= 0 or byte_count <= 0:
                return None
            shm = self._attach(stream_id)
            frame = np.ndarray((height, width, channels), dtype=np.uint8, buffer=shm.buf[:byte_count]).copy()
            return {
                "stream_id": stream_id,
                "frame": frame,
                "frame_number": int(meta["frame_number"]),
                "timestamp": _from_timestamp(meta["timestamp"]),
                "generation": int(meta["generation"]),
                "connected": bool(meta["connected"]),
                "shape": frame.shape,
                "original_width": int(meta["source_width"] or width),
                "original_height": int(meta["source_height"] or height),
            }

    def snapshot_meta(self) -> List[dict]:
        snapshots = []
        for stream_id in range(self.stream_count):
            with self._locks[stream_id]:
                snapshots.append(self._meta_snapshot_locked(stream_id))
        return snapshots

    def iter_updated_frames(
        self,
        last_generations: Optional[Dict[int, int]] = None,
        enabled_streams: Optional[Iterable[int]] = None,
    ) -> List[dict]:
        seen = last_generations or {}
        allowed = None if enabled_streams is None else {int(x) for x in enabled_streams}
        updated = []
        for stream_id in range(self.stream_count):
            if allowed is not None and stream_id not in allowed:
                continue
            generation = self._generation(stream_id)
            if generation <= int(seen.get(stream_id, 0)):
                continue
            frame_data = self.read_frame(stream_id)
            if frame_data is not None:
                updated.append(frame_data)
            else:
                seen[stream_id] = generation
        updated.sort(key=lambda item: (item["frame_number"], item["stream_id"]))
        return updated

    def close(self):
        for shm in list(self._local_shm.values()):
            try:
                shm.close()
            except Exception:
                pass
        self._local_shm.clear()
        for name in self._shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.unlink()
                shm.close()
            except FileNotFoundError:
                pass
            except Exception:
                pass


class FrameHubQueueAdapter:
    """
    用于兼容现有检测器的 queue 读取模式。

    检测器继续通过 get/get_nowait 读取 frame_data，
    但底层数据来自 LatestFrameHub 的“每路最新帧”快照，而不是共享大图队列。
    """

    def __init__(self, frame_hub: LatestFrameHub, enabled_streams: Optional[Iterable[int]] = None):
        self.frame_hub = frame_hub
        self.enabled_streams = None if enabled_streams is None else {int(x) for x in enabled_streams}
        self._seen_generations: Dict[int, int] = {}
        self._buffer: List[dict] = []

    def set_enabled_streams(self, enabled_streams: Optional[Iterable[int]]):
        self.enabled_streams = None if enabled_streams is None else {int(x) for x in enabled_streams}

    def _refill(self):
        self._buffer = self.frame_hub.iter_updated_frames(
            self._seen_generations,
            enabled_streams=self.enabled_streams,
        )

    def get_nowait(self):
        if not self._buffer:
            self._refill()
        if not self._buffer:
            raise queue.Empty()
        frame_data = self._buffer.pop(0)
        self._seen_generations[frame_data["stream_id"]] = int(frame_data["generation"])
        return frame_data

    def get(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)


def adapt_frame_source(frame_source):
    if isinstance(frame_source, LatestFrameHub):
        return FrameHubQueueAdapter(frame_source)
    return frame_source
