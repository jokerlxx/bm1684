import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


NINE_STREAM_MIN_INTERVALS = {
    "helmet_detection": 1.0,
    "fall_detection": 1.5,
    "fight_detection": 1.5,
    "crowd_detection": 2.0,
    "window_door_detection": 1.5,
    "ventilator_detection": 2.0,
}


@dataclass
class TimeShareConfig:
    enabled: bool = True
    min_update_interval_s: float = 0.5
    max_active_streams_per_detector: int = 1
    frame_cache_ttl_s: float = 1.5
    tick_sleep_s: float = 0.005

    @staticmethod
    def from_config(config: dict, detector_section: Optional[str] = None) -> "TimeShareConfig":
        raw = (config or {}).get("detection_timeshare") or {}
        detector_raw = dict((config or {}).get(detector_section) or {})
        advanced_raw = ((config or {}).get("advanced_algorithm_params") or {}).get(detector_section) or {}
        detector_raw.update(advanced_raw)
        stream_count = len((config or {}).get("video_streams") or []) or 1
        min_update_interval_s = float(detector_raw.get("min_update_interval_s", raw.get("min_update_interval_s", 0.5)))
        if (
            detector_section
            and stream_count >= 9
            and "min_update_interval_s" not in detector_raw
            and detector_section in NINE_STREAM_MIN_INTERVALS
        ):
            min_update_interval_s = max(min_update_interval_s, float(NINE_STREAM_MIN_INTERVALS[detector_section]))
        return TimeShareConfig(
            enabled=bool(raw.get("enabled", True)),
            min_update_interval_s=min_update_interval_s,
            max_active_streams_per_detector=max(1, int(raw.get("max_active_streams_per_detector", 1))),
            frame_cache_ttl_s=float(raw.get("frame_cache_ttl_s", 1.5)),
            tick_sleep_s=float(raw.get("tick_sleep_s", 0.005)),
        )


class LatestFrameCache:
    def __init__(self, cfg: TimeShareConfig):
        self.cfg = cfg
        self._latest: Dict[int, dict] = {}
        self._latest_ts: Dict[int, float] = {}
        self._last_infer_ts: Dict[int, float] = {}

    def update(self, frame_data: dict) -> None:
        stream_id = int(frame_data.get("stream_id", 0))
        now = time.monotonic()
        self._latest[stream_id] = frame_data
        self._latest_ts[stream_id] = now

    def prune(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.monotonic()
        ttl = self.cfg.frame_cache_ttl_s
        if ttl <= 0:
            return
        expired = [sid for sid, ts in self._latest_ts.items() if now - ts > ttl]
        for sid in expired:
            self._latest.pop(sid, None)
            self._latest_ts.pop(sid, None)

    def pick_due_streams(self, enabled_streams: Optional[Iterable[int]] = None) -> List[int]:
        now = time.monotonic()
        self.prune(now)
        candidates = list(self._latest.keys())
        if enabled_streams is not None:
            allowed = {int(x) for x in enabled_streams}
            candidates = [sid for sid in candidates if sid in allowed]
        min_iv = max(0.0, float(self.cfg.min_update_interval_s))
        due = [sid for sid in candidates if now - self._last_infer_ts.get(sid, 0.0) >= min_iv]
        due.sort(key=lambda sid: self._last_infer_ts.get(sid, 0.0))
        return due[: max(1, int(self.cfg.max_active_streams_per_detector))]

    def get_latest(self, stream_id: int) -> Optional[dict]:
        return self._latest.get(int(stream_id))

    def stream_ids(self) -> List[int]:
        return list(self._latest.keys())

    def mark_inferred(self, stream_id: int) -> None:
        self._last_infer_ts[int(stream_id)] = time.monotonic()
