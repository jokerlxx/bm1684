from __future__ import annotations

import queue
import signal
import sys
import time
import logging
from importlib import import_module
from typing import Any, Dict, Iterable, Optional

from core.frame_hub import adapt_frame_source
from core.logging_utils import ensure_root_logging, log_model_inference_metrics

from app.algorithms.base import algorithm_config
from app.application.detector_registry import DetectorRegistry
from app.model_runtime.model_type_registry import resolve_model_type
from app.model_runtime.runtime import ModelManager, ModelSpec
from app.pipeline.job_scheduler import (
    DEFAULT_DETECTOR_PRIORITY,
    JobTable,
    group_single_model_batches,
    pick_jobs_for_tick,
)
from app.pipeline.messages import FrameContext, InferenceResult
from app.pipeline.time_share import LatestFrameCache, TimeShareConfig


MODEL_CONSUMERS = {
    "fall_detection": ["fall"],
    "fight_detection": ["fight"],
    "helmet_detection": ["helmet", "crowd", "ventilator"],
    "ventilator_equipment": [],
    "window_door_inside": ["window_door_inside"],
    "window_door_outside": ["window_door_outside"],
}

DETECTOR_INPUTS = {
    "fall": ["fall_detection"],
    "fight": ["fight_detection"],
    "helmet": ["helmet_detection"],
    "crowd": ["helmet_detection"],
    "ventilator": ["ventilator_equipment", "helmet_detection"],
    "window_door_inside": ["window_door_inside"],
    "window_door_outside": ["window_door_outside"],
}

MODEL_INTERVAL_DEFAULTS = {
    "helmet_detection": 0.2,
    "fall_detection": 0.3,
    "fight_detection": 0.3,
    "ventilator_equipment": 1.0,
    "window_door_inside": 1.0,
    "window_door_outside": 1.0,
}

MODEL_ALIASES = {
    "crowd_person": "helmet_detection",
    "ventilator_helmet": "helmet_detection",
}


logger = logging.getLogger(__name__)


class BatchModelScheduler:
    def __init__(
        self,
        frame_source,
        result_queues: Dict[str, Any],
        control_queue,
        config: Dict[str, Any],
        registry: Optional[DetectorRegistry] = None,
        model_manager: Optional[ModelManager] = None,
        preview_status: Optional[Any] = None,
    ):
        self.frame_source = adapt_frame_source(frame_source)
        self.result_queues = result_queues
        self.control_queue = control_queue
        self.config = config or {}
        self.registry = registry or DetectorRegistry()
        self.model_manager = model_manager or ModelManager()
        self.preview_status = preview_status
        self.algorithms = self._build_algorithms()
        self.states: Dict[str, Dict[str, Any]] = {name: {} for name in self.algorithms}
        self.enabled_detectors: set[str] = set()
        self.detector_streams: Dict[str, set[int]] = {}
        self.running = False
        raw_ts = dict(self.config.get("detection_timeshare") or {})
        batch_cfg = self.config.get("batch_scheduler") or {}
        raw_ts["min_update_interval_s"] = float(batch_cfg.get("stream_interval_s", raw_ts.get("min_update_interval_s", 0.2)))
        raw_ts["max_active_streams_per_detector"] = max(1, int(batch_cfg.get("batch_streams", raw_ts.get("max_active_streams_per_detector", 9))))
        ts_config = dict(self.config)
        ts_config["detection_timeshare"] = raw_ts
        self.time_share_config = TimeShareConfig.from_config(ts_config)
        self.model_intervals = self._model_intervals()
        batch_cfg = self.config.get("batch_scheduler") or {}
        self.max_models_per_tick = max(1, int(batch_cfg.get("max_models_per_tick", 1)))
        self.preview_backpressure_enabled = bool(batch_cfg.get("preview_backpressure_enabled", True))
        self.preview_backpressure_fps = float(batch_cfg.get("preview_backpressure_fps", 18.0))
        self.preview_backpressure_sleep_s = float(batch_cfg.get("preview_backpressure_sleep_s", 0.05))
        self.preview_backpressure_consecutive = max(1, int(batch_cfg.get("preview_backpressure_consecutive", 3)))
        self._preview_low_fps_count = 0
        self._last_metrics_log_ts = 0.0
        self._metrics_log_interval_s = max(1.0, float(batch_cfg.get("metrics_log_interval_s", 5.0)))
        self._metrics = self._new_metrics()
        self.job_intervals_config = dict(batch_cfg.get("job_intervals_s") or {})
        raw_priority = dict(batch_cfg.get("detector_priority") or {})
        self.detector_priority_config = dict(DEFAULT_DETECTOR_PRIORITY)
        for key, value in raw_priority.items():
            try:
                self.detector_priority_config[str(key)] = int(value)
            except Exception:
                continue
        self.max_jobs_per_tick = max(1, int(batch_cfg.get("max_jobs_per_tick", 4)))
        legacy_max_models = int(batch_cfg.get("max_models_per_tick", 2))
        self.max_model_inferences_per_tick = max(
            1,
            int(batch_cfg.get("max_model_inferences_per_tick", legacy_max_models)),
        )
        self.job_table = JobTable()
        self.latest_model_results: Dict[int, Dict[str, InferenceResult]] = {}
        self.latest_contexts: Dict[int, FrameContext] = {}

    @staticmethod
    def _new_metrics() -> Dict[str, Any]:
        return {
            "frames_absorbed": 0,
            "ticks_no_models": 0,
            "ticks_no_due_streams": 0,
            "ticks_no_frame_items": 0,
            "ticks_backpressure": 0,
            "inferences": {},
            "infer_ms": {},
            "timing_ms": {},
            "batches": {},
            "batch_sizes": {},
            "job_served": {},
            "waste_inferences": 0,
        }

    def _build_algorithms(self):
        algorithms = {}
        for name, spec in self.registry.all().items():
            cls = spec.algorithm_cls
            if isinstance(cls, str):
                module_name, class_name = cls.split(":", 1)
                cls = getattr(import_module(module_name), class_name)
            algorithms[name] = cls(self.config, detector_name=spec.detector_name)
        return algorithms

    def _model_intervals(self) -> Dict[str, float]:
        raw = self.config.get("batch_scheduler", {}).get("model_intervals_s", {})
        intervals = dict(MODEL_INTERVAL_DEFAULTS)
        for key, value in raw.items():
            try:
                intervals[str(key)] = max(0.0, float(value))
            except Exception:
                logger.exception("Invalid model_intervals_s entry: key=%s value=%s", key, value)
                continue
        return intervals

    def _job_interval(self, detector_name: str) -> float:
        configured = self.job_intervals_config.get(detector_name)
        if configured is not None:
            return max(0.0, float(configured))
        for model_key in DETECTOR_INPUTS.get(detector_name, []):
            interval = self.model_intervals.get(model_key)
            if interval is not None:
                return max(0.0, float(interval))
        return max(0.0, float(self.time_share_config.min_update_interval_s))

    def _rebuild_jobs(self) -> None:
        self.job_table.rebuild(
            self.enabled_detectors,
            self.detector_streams,
            DETECTOR_INPUTS,
            self._job_interval,
            self.detector_priority_config,
            now=time.monotonic(),
        )

    def _active_model_keys(self) -> list[str]:
        keys = set()
        for job in self.job_table.active_jobs():
            keys.update(job.model_keys)
        if keys:
            return sorted(keys)
        for detector_name in self.enabled_detectors:
            for model_key in DETECTOR_INPUTS.get(detector_name, []):
                keys.add(model_key)
        return sorted(keys)

    def _build_model_specs(self):
        model_specs = []
        models = self.config.get("models") or {}
        model_types = self.config.get("model_types") or {}
        thresholds_by_model = {
            "fall_detection": "fall_detection",
            "fight_detection": "fight_detection",
            "helmet_detection": "helmet_detection",
            "ventilator_equipment": "ventilator_detection",
            "window_door_inside": "window_door_detection",
            "window_door_outside": "window_door_detection",
        }
        for model_key in self._active_model_keys():
            model_path = models.get(model_key)
            if not model_path:
                alias_key = next((src for src, dst in MODEL_ALIASES.items() if dst == model_key and models.get(src)), None)
                model_path = models.get(alias_key) if alias_key else None
            if not model_path:
                raise ValueError(f"Missing model path for {model_key}")
            model_type = resolve_model_type(model_key, self.config).raw if model_key in model_types else None
            section = thresholds_by_model.get(model_key, model_key)
            model_specs.append(
                ModelSpec(
                    model_key=model_key,
                    model_path=model_path,
                    model_type=model_type,
                    device_id=int(self.config.get("bm1684x", {}).get("device_id", 0)),
                    thresholds=algorithm_config(self.config, section),
                )
            )
        return model_specs

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.stop()
        sys.exit(0)

    def _send_result(self, detector_name: str, payload):
        result_queue = self.result_queues.get(detector_name)
        if result_queue is None:
            return
        try:
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put(payload, block=False)
        except Exception:
            pass

    def _check_control_commands(self):
        try:
            while True:
                cmd = self.control_queue.get_nowait()
                if cmd == "stop":
                    self.stop()
                    continue
                if not isinstance(cmd, dict):
                    continue
                detector = cmd.get("detector")
                if cmd.get("cmd") == "enable" and detector:
                    self.enabled_detectors.add(str(detector))
                    streams = cmd.get("stream_ids")
                    if streams is not None:
                        self.detector_streams[str(detector)] = {int(x) for x in streams}
                elif cmd.get("cmd") == "disable" and detector:
                    self.enabled_detectors.discard(str(detector))
                    self.detector_streams.pop(str(detector), None)
                elif cmd.get("cmd") == "set_streams" and detector:
                    self.detector_streams[str(detector)] = {int(x) for x in cmd.get("stream_ids") or []}
                elif cmd.get("cmd") == "reload_config":
                    new_config = cmd.get("config")
                    if isinstance(new_config, dict):
                        self.config = new_config
                        self.algorithms = self._build_algorithms()
                        self.states = {name: {} for name in self.algorithms}
                        batch_cfg = self.config.get("batch_scheduler") or {}
                        self.job_intervals_config = dict(batch_cfg.get("job_intervals_s") or {})
                        raw_priority = dict(batch_cfg.get("detector_priority") or {})
                        self.detector_priority_config = dict(DEFAULT_DETECTOR_PRIORITY)
                        for key, value in raw_priority.items():
                            try:
                                self.detector_priority_config[str(key)] = int(value)
                            except Exception:
                                continue
                        self.max_jobs_per_tick = max(1, int(batch_cfg.get("max_jobs_per_tick", 4)))
                        legacy_max_models = int(batch_cfg.get("max_models_per_tick", 2))
                        self.max_model_inferences_per_tick = max(
                            1,
                            int(batch_cfg.get("max_model_inferences_per_tick", legacy_max_models)),
                        )
                        self.model_intervals = self._model_intervals()
                        self._load_active_models()
                        self.latest_model_results.clear()
                self._rebuild_jobs()
        except queue.Empty:
            return
        except Exception:
            return

    def _enabled_streams_union(self) -> Optional[set[int]]:
        if not self.enabled_detectors:
            return set()
        streams: set[int] = set()
        has_any = False
        for detector in self.enabled_detectors:
            detector_streams = self.detector_streams.get(detector)
            if detector_streams is None:
                return None
            streams.update(detector_streams)
            has_any = True
        return streams if has_any else set()

    def _detector_enabled_for_stream(self, detector_name: str, stream_id: int) -> bool:
        if detector_name not in self.enabled_detectors:
            return False
        streams = self.detector_streams.get(detector_name)
        return streams is None or int(stream_id) in streams

    def _preview_under_pressure(self) -> bool:
        if not self.preview_backpressure_enabled or self.preview_status is None:
            self._preview_low_fps_count = 0
            return False
        try:
            fps = float(dict(self.preview_status).get("compose_fps") or 0.0)
        except Exception:
            self._preview_low_fps_count = 0
            return False
        if 0.0 < fps < self.preview_backpressure_fps:
            self._preview_low_fps_count += 1
        else:
            self._preview_low_fps_count = 0
        return self._preview_low_fps_count >= self.preview_backpressure_consecutive

    def _record_inference_metrics(
        self,
        model_key: str,
        batch_size: int,
        total_ms: Optional[float] = None,
        timings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._metrics["inferences"][model_key] = self._metrics["inferences"].get(model_key, 0) + 1
        self._metrics["batches"][model_key] = self._metrics["batches"].get(model_key, 0) + int(batch_size)
        self._metrics["batch_sizes"].setdefault(model_key, []).append(int(batch_size))
        timings = dict(timings or {})
        if total_ms is None:
            total_ms = timings.get("total_ms")
        if total_ms is not None:
            self._metrics["infer_ms"].setdefault(model_key, []).append(float(total_ms))
            timings.setdefault("total_ms", total_ms)
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            value = timings.get(key)
            if value is not None:
                self._metrics["timing_ms"].setdefault(model_key, {}).setdefault(key, []).append(float(value))

    def _record_job_served(self, job) -> None:
        label = f"ch{int(job.stream_id) + 1}:{job.detector_name}"
        self._metrics["job_served"][label] = self._metrics["job_served"].get(label, 0) + 1

    def _maybe_log_metrics(self, active_models: Iterable[str], enabled_streams: Optional[set[int]]) -> None:
        now = time.monotonic()
        if now - self._last_metrics_log_ts < self._metrics_log_interval_s:
            return
        self._last_metrics_log_ts = now
        active = sorted(active_models)
        stream_desc = "all" if enabled_streams is None else ",".join(str(s + 1) for s in sorted(enabled_streams))
        window_s = max(0.001, float(self._metrics_log_interval_s))
        infer_parts = []
        for model_key in sorted(self._metrics["inferences"]):
            count = self._metrics["inferences"].get(model_key, 0)
            batches = self._metrics["batches"].get(model_key, 0)
            size_samples = self._metrics["batch_sizes"].get(model_key) or []
            batch_avg = sum(size_samples) / len(size_samples) if size_samples else 0.0
            samples = self._metrics["infer_ms"].get(model_key) or []
            avg_ms = sum(samples) / len(samples) if samples else 0.0
            max_ms = max(samples) if samples else 0.0
            timing_samples = self._metrics["timing_ms"].get(model_key) or {}
            pre_samples = timing_samples.get("preprocess_ms") or []
            infer_samples = timing_samples.get("inference_ms") or []
            post_samples = timing_samples.get("postprocess_ms") or []
            if pre_samples or infer_samples or post_samples:
                avg_pre = sum(pre_samples) / len(pre_samples) if pre_samples else 0.0
                avg_infer = sum(infer_samples) / len(infer_samples) if infer_samples else 0.0
                avg_post = sum(post_samples) / len(post_samples) if post_samples else 0.0
                infer_parts.append(
                    f"{model_key}:n={count},frames={batches},batch_avg={batch_avg:.1f},pre={avg_pre:.1f}ms,infer={avg_infer:.1f}ms,post={avg_post:.1f}ms,total={avg_ms:.1f}ms,max={max_ms:.1f}ms"
                )
            else:
                infer_parts.append(
                    f"{model_key}:n={count},frames={batches},batch_avg={batch_avg:.1f},avg={avg_ms:.1f}ms,max={max_ms:.1f}ms"
                )
        job_parts = []
        for label in sorted(self._metrics["job_served"]):
            served = self._metrics["job_served"].get(label, 0)
            job_parts.append(f"{label}={served / window_s:.1f}")
        logger.info(
            "Batch scheduler metrics: detectors=%s models=%s streams=%s absorbed=%d skips(no_models=%d,no_streams=%d,no_frames=%d,backpressure=%d) waste=%d jobs=[%s] infer=[%s]",
            sorted(self.enabled_detectors),
            active,
            stream_desc or "none",
            self._metrics["frames_absorbed"],
            self._metrics["ticks_no_models"],
            self._metrics["ticks_no_due_streams"],
            self._metrics["ticks_no_frame_items"],
            self._metrics["ticks_backpressure"],
            int(self._metrics.get("waste_inferences", 0)),
            ", ".join(job_parts) or "none",
            "; ".join(infer_parts) or "none",
        )
        self._metrics = self._new_metrics()

    def _normalize_outputs_for_detector(self, detector_name: str, stream_id: int):
        results = self.latest_model_results.get(stream_id, {})
        inputs = DETECTOR_INPUTS.get(detector_name, [])
        if any(model_key not in results for model_key in inputs):
            return None
        if detector_name == "ventilator":
            return {
                "ventilator_equipment": results["ventilator_equipment"].detections,
                "ventilator_helmet": results["helmet_detection"].detections,
            }
        if len(inputs) == 1:
            return results[inputs[0]].detections
        return {model_key: results[model_key].detections for model_key in inputs}


    @staticmethod
    def _attach_coord_size(payload, context: FrameContext) -> None:
        frame = context.frame
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            payload["coord_width"] = int(shape[1])
            payload["coord_height"] = int(shape[0])

    @staticmethod
    def _detections_to_dicts(detections) -> list[dict]:
        return [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in (detections or [])
        ]

    def _attach_raw_model_detections(self, payload, detector_name: str, stream_id: int) -> None:
        results = self.latest_model_results.get(stream_id, {})
        raw = {}
        for model_key in DETECTOR_INPUTS.get(detector_name, []):
            result = results.get(model_key)
            if result is None:
                continue
            raw[model_key] = self._detections_to_dicts(result.detections)
        if raw:
            payload["raw_model_detections"] = raw

    def _infer_calls_for_job(self, job) -> int:
        return len(job.model_keys) if job.detector_name == "ventilator" else 1

    def _store_inference_result(
        self,
        stream_id: int,
        model_key: str,
        inference_result: InferenceResult,
        detector_name: Optional[str] = None,
    ) -> None:
        self.latest_model_results.setdefault(stream_id, {})[model_key] = inference_result
        timing = inference_result.timings or {}
        batch_size = int((inference_result.raw_meta or {}).get("batch_size") or 1)
        self._record_inference_metrics(model_key, batch_size, timings=timing)
        if timing.get("total_ms") is not None:
            log_model_inference_metrics(
                stream_id,
                model_key,
                timings=timing,
                detector_name=detector_name,
                model_path=(inference_result.raw_meta or {}).get("model_path"),
                batch_size=batch_size,
            )

    def _run_ventilator_job(self, job, context: FrameContext) -> bool:
        stream_id = job.stream_id
        for model_key in job.model_keys:
            try:
                results = self.model_manager.infer_batch(model_key, [context.frame])
            except Exception:
                logger.exception(
                    "Job inference failed: detector=%s model=%s stream=%s",
                    job.detector_name,
                    model_key,
                    stream_id + 1,
                )
                return False
            if not results:
                return False
            self._store_inference_result(stream_id, model_key, results[0], detector_name=job.detector_name)
        self._run_consumers("helmet_detection", stream_id)
        return True

    def _run_model_batch(
        self,
        model_key: str,
        batch_items: list[tuple[Any, FrameContext]],
    ) -> bool:
        frames = [context.frame for _, context in batch_items]
        try:
            results = self.model_manager.infer_batch(model_key, frames)
        except Exception:
            logger.exception("Batch inference failed: model=%s streams=%s", model_key, [j.stream_id + 1 for j, _ in batch_items])
            return False
        if not results:
            return False
        for (job, _), inference_result in zip(batch_items, results):
            self._store_inference_result(job.stream_id, model_key, inference_result, detector_name=job.detector_name)
            self._run_consumers(model_key, job.stream_id)
        return True

    def _run_consumers(self, model_key: str, stream_id: int):
        context = self.latest_contexts.get(stream_id)
        if context is None:
            return
        for detector_name in MODEL_CONSUMERS.get(model_key, []):
            if not self._detector_enabled_for_stream(detector_name, stream_id):
                continue
            inference_outputs = self._normalize_outputs_for_detector(detector_name, stream_id)
            if inference_outputs is None:
                continue
            try:
                result = self.algorithms[detector_name].process(context, inference_outputs, self.states[detector_name])
                payload = result.to_dict()
                self._attach_coord_size(payload, context)
                self._attach_raw_model_detections(payload, detector_name, stream_id)
                self._send_result(detector_name, payload)
            except Exception:
                continue

    def _load_active_models(self):
        self.model_manager.close()
        self.model_manager.load_all(self._build_model_specs())

    def start(self):
        ensure_root_logging()
        self.setup_signal_handlers()
        self.running = True
        loaded_for: set[str] = set()
        cache = LatestFrameCache(self.time_share_config)
        while self.running:
            self._check_control_commands()
            active_models = set(self._active_model_keys())
            if active_models != loaded_for:
                if active_models:
                    self._load_active_models()
                else:
                    self.model_manager.close()
                loaded_for = active_models
                self.latest_model_results.clear()

            enabled_streams = self.job_table.job_stream_ids()
            try:
                while True:
                    frame_data = self.frame_source.get_nowait()
                    stream_id = int(frame_data.get("stream_id", 0))
                    if stream_id in enabled_streams:
                        cache.update(frame_data)
                        self._metrics["frames_absorbed"] += 1
            except Exception:
                pass

            if not active_models or not enabled_streams:
                self._metrics["ticks_no_models"] += 1
                self._maybe_log_metrics(active_models, enabled_streams)
                time.sleep(self.time_share_config.tick_sleep_s)
                continue

            now = time.monotonic()
            due_jobs = self.job_table.due_jobs(now)
            if not due_jobs:
                self._metrics["ticks_no_due_streams"] += 1
                self._maybe_log_metrics(active_models, enabled_streams)
                time.sleep(self.time_share_config.tick_sleep_s)
                continue

            if self._preview_under_pressure():
                self._metrics["ticks_backpressure"] += 1
                self._maybe_log_metrics(active_models, enabled_streams)
                time.sleep(self.preview_backpressure_sleep_s)
                continue

            selected_jobs = pick_jobs_for_tick(due_jobs, self.max_jobs_per_tick)
            ready_jobs: list[tuple[Any, FrameContext]] = []
            for job in selected_jobs:
                frame_data = cache.get_latest(job.stream_id)
                if not frame_data:
                    self._metrics["ticks_no_frame_items"] += 1
                    continue
                context = FrameContext.from_payload(frame_data)
                self.latest_contexts[job.stream_id] = context
                ready_jobs.append((job, context))

            infer_budget = self.max_model_inferences_per_tick
            served_any = False

            model_batches = group_single_model_batches(
                ready_jobs,
                self.model_manager.get_max_batch,
            )
            for model_key, batch_items in model_batches:
                if infer_budget < 1:
                    break
                if not self._run_model_batch(model_key, batch_items):
                    continue
                infer_budget -= 1
                for job, _ in batch_items:
                    self.job_table.mark_served(job, now)
                    self._record_job_served(job)
                served_any = True

            for job, context in ready_jobs:
                if job.detector_name != "ventilator":
                    continue
                infer_needed = self._infer_calls_for_job(job)
                if infer_budget < infer_needed:
                    continue
                if not self._run_ventilator_job(job, context):
                    continue
                infer_budget -= infer_needed
                self.job_table.mark_served(job, now)
                self._record_job_served(job)
                served_any = True

            if not served_any and selected_jobs:
                self._metrics["ticks_no_frame_items"] += 1
            self._maybe_log_metrics(active_models, enabled_streams)

        self.model_manager.close()

    def stop(self):
        self.running = False


def run_batch_model_scheduler(frame_hub, result_queues, control_queue, config, preview_status=None):
    scheduler = BatchModelScheduler(
        frame_source=frame_hub,
        result_queues=result_queues,
        control_queue=control_queue,
        config=config,
        preview_status=preview_status,
    )
    scheduler.start()
