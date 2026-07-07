from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


JobId = Tuple[int, str]

DEFAULT_DETECTOR_PRIORITY: Dict[str, int] = {
    "fall": 10,
    "fight": 10,
    "helmet": 9,
    "crowd": 9,
    "ventilator": 9,
    "window_door_inside": 3,
    "window_door_outside": 3,
}


@dataclass
class InferenceJob:
    stream_id: int
    detector_name: str
    model_keys: List[str]
    interval_s: float
    priority: int
    next_due: float = 0.0
    last_served_mono: float = 0.0

    @property
    def job_id(self) -> JobId:
        return (self.stream_id, self.detector_name)


class JobTable:
    def __init__(self) -> None:
        self._jobs: Dict[JobId, InferenceJob] = {}

    def active_jobs(self) -> List[InferenceJob]:
        return list(self._jobs.values())

    def job_stream_ids(self) -> Set[int]:
        return {job.stream_id for job in self._jobs.values()}

    def due_jobs(self, now: float) -> List[InferenceJob]:
        return [job for job in self._jobs.values() if now >= job.next_due]

    def mark_served(self, job: InferenceJob, now: float) -> None:
        job.last_served_mono = now
        job.next_due = now + max(0.0, job.interval_s)

    def rebuild(
        self,
        enabled_detectors: Set[str],
        detector_streams: Dict[str, Set[int]],
        detector_inputs: Dict[str, List[str]],
        interval_for_detector: Callable[[str], float],
        priorities: Optional[Dict[str, int]] = None,
        now: float = 0.0,
    ) -> None:
        priority_map = dict(DEFAULT_DETECTOR_PRIORITY)
        if priorities:
            priority_map.update(priorities)

        preserved: Dict[JobId, InferenceJob] = dict(self._jobs)
        new_jobs: Dict[JobId, InferenceJob] = {}

        for detector_name in sorted(enabled_detectors):
            streams = detector_streams.get(detector_name)
            if streams is None:
                continue
            model_keys = list(detector_inputs.get(detector_name, []))
            if not model_keys:
                continue
            interval_s = max(0.0, float(interval_for_detector(detector_name)))
            priority = int(priority_map.get(detector_name, 5))
            for stream_id in sorted(streams):
                job_id = (int(stream_id), detector_name)
                existing = preserved.get(job_id)
                if existing is not None:
                    existing.interval_s = interval_s
                    existing.priority = priority
                    existing.model_keys = list(model_keys)
                    new_jobs[job_id] = existing
                else:
                    new_jobs[job_id] = InferenceJob(
                        stream_id=int(stream_id),
                        detector_name=detector_name,
                        model_keys=list(model_keys),
                        interval_s=interval_s,
                        priority=priority,
                        next_due=now,
                        last_served_mono=0.0,
                    )

        self._jobs = new_jobs


def pick_jobs_for_tick(
    due_jobs: Iterable[InferenceJob],
    max_jobs: int,
) -> List[InferenceJob]:
    ranked = sorted(
        due_jobs,
        key=lambda job: (-job.priority, job.last_served_mono, job.stream_id, job.detector_name),
    )
    limit = max(1, int(max_jobs))
    return ranked[:limit]


def group_single_model_batches(
    ready: Iterable[tuple[InferenceJob, object]],
    max_batch_for_model: Callable[[str], int],
) -> List[tuple[str, List[tuple[InferenceJob, object]]]]:
    buckets: Dict[str, List[tuple[InferenceJob, object]]] = {}
    for job, context in ready:
        if job.detector_name == "ventilator":
            continue
        model_key = job.model_keys[0]
        buckets.setdefault(model_key, []).append((job, context))

    batches: List[tuple[str, List[tuple[InferenceJob, object]]]] = []
    for model_key, items in buckets.items():
        limit = max(1, int(max_batch_for_model(model_key)))
        chunk = items[:limit]
        if chunk:
            batches.append((model_key, chunk))

    batches.sort(key=lambda item: -max(job.priority for job, _ in item[1]))
    return batches
