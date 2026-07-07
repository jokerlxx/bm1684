from __future__ import annotations

import queue
import signal
import sys
import time
from importlib import import_module
import threading
from typing import Any, Dict, Optional

from core.frame_hub import adapt_frame_source
from core.logging_utils import ensure_root_logging, log_model_inference_metrics

from app.algorithms.base import algorithm_config
from app.application.detector_registry import DetectorRegistry, DetectorSpec
from app.model_runtime.model_type_registry import resolve_model_type
from app.model_runtime.runtime import ModelManager, ModelSpec
from app.pipeline.messages import FrameContext
from app.pipeline.time_share import LatestFrameCache, TimeShareConfig


class DetectorWorker:
    def __init__(
        self,
        detector_spec: DetectorSpec,
        frame_source,
        result_queue,
        control_queue,
        config: Dict[str, Any],
        model_overrides: Optional[Dict[str, str]] = None,
        registry: Optional[DetectorRegistry] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.detector_spec = detector_spec
        self.registry = registry or DetectorRegistry()
        self.frame_source = adapt_frame_source(frame_source)
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config or {}
        self.model_overrides = dict(model_overrides or {})
        self.model_manager = model_manager or ModelManager()
        self.algorithm = self._build_algorithm(detector_spec)
        self.state: Dict[str, Any] = {}
        self.enabled = False
        self.enabled_streams = None
        self.running = False
        self.time_share_config = TimeShareConfig.from_config(self.config, detector_spec.default_config_section)

    def _build_algorithm(self, detector_spec: DetectorSpec):
        algorithm_cls = detector_spec.algorithm_cls
        if isinstance(algorithm_cls, str):
            module_name, class_name = algorithm_cls.split(":", 1)
            algorithm_cls = getattr(import_module(module_name), class_name)
        return algorithm_cls(self.config, detector_name=detector_spec.detector_name)

    def _build_model_specs(self):
        model_specs = []
        for model_key in self.detector_spec.model_keys:
            model_path = self.model_overrides.get(model_key) or (self.config.get("models") or {}).get(model_key)
            if not model_path:
                raise ValueError(f"Missing model path for {model_key}")
            model_type = None
            model_types = self.config.get("model_types") or {}
            if model_key in model_types:
                model_type = resolve_model_type(model_key, self.config).raw
            model_specs.append(
                ModelSpec(
                    model_key=model_key,
                    model_path=model_path,
                    model_type=model_type,
                    device_id=int(self.config.get("bm1684x", {}).get("device_id", 0)),
                    thresholds=algorithm_config(self.config, self.detector_spec.default_config_section),
                )
            )
        return model_specs

    def setup_signal_handlers(self):
        if threading.current_thread() is not threading.main_thread():
            return
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.stop()
        sys.exit(0)

    def _send_result(self, payload):
        try:
            if self.result_queue.full():
                self.result_queue.get_nowait()
            self.result_queue.put(payload, block=False)
        except Exception:
            pass

    def _check_control_commands(self):
        try:
            while True:
                cmd = self.control_queue.get_nowait()
                if isinstance(cmd, dict) and cmd.get("cmd") == "set_streams":
                    self.enabled_streams = set(cmd.get("stream_ids") or [])
                    continue
                if isinstance(cmd, dict) and cmd.get("cmd") == "reload_config":
                    new_config = cmd.get("config")
                    if isinstance(new_config, dict):
                        self.config = new_config
                        self.algorithm = self._build_algorithm(self.detector_spec)
                        self.state = {}
                        self.time_share_config = TimeShareConfig.from_config(
                            self.config,
                            self.detector_spec.default_config_section,
                        )
                        self.model_manager.close()
                        self.model_manager.load_all(self._build_model_specs())
                    continue
                if cmd == "stop":
                    self.stop()
                elif cmd == "enable":
                    self.enabled = True
                elif cmd == "disable":
                    self.enabled = False
        except queue.Empty:
            return
        except Exception:
            return


    @staticmethod
    def _attach_coord_size(payload, frame_context: FrameContext) -> None:
        frame = frame_context.frame
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

    def _resolve_inference_outputs(self, frame_context: FrameContext):
        outputs = {}
        raw_outputs = {}
        for model_key in self.detector_spec.model_keys:
            result = self.model_manager.infer(model_key, frame_context.frame)
            timing = result.timings or {}
            if timing.get("total_ms") is not None:
                log_model_inference_metrics(
                    frame_context.stream_id,
                    model_key,
                    timings=timing,
                    detector_name=self.detector_spec.detector_name,
                    model_path=(result.raw_meta or {}).get("model_path"),
                )
            outputs[model_key] = self.detector_spec.result_mapper(result)
            raw_outputs[model_key] = self._detections_to_dicts(result.detections)
        if len(outputs) == 1:
            return next(iter(outputs.values())), raw_outputs
        return outputs, raw_outputs

    def start(self):
        ensure_root_logging()
        self.setup_signal_handlers()
        self.model_manager.load_all(self._build_model_specs())
        self.running = True
        self.run()

    def run(self):
        cache = LatestFrameCache(self.time_share_config)
        last_disabled_push_ts = 0.0

        while self.running:
            self._check_control_commands()

            got_any = False
            try:
                while True:
                    frame_data = self.frame_source.get_nowait()
                    cache.update(frame_data)
                    got_any = True
            except Exception:
                pass

            if not self.enabled:
                now_mono = time.monotonic()
                if got_any and now_mono - last_disabled_push_ts >= 0.5:
                    last_disabled_push_ts = now_mono
                    for stream_id in cache.stream_ids():
                        frame_data = cache.get_latest(stream_id)
                        if not frame_data:
                            continue
                        frame_context = FrameContext.from_payload(frame_data)
                        result = self.algorithm.build_disabled_result(frame_context, self.state)
                        self._send_result(result.to_dict())
                time.sleep(self.time_share_config.tick_sleep_s if self.time_share_config.enabled else 0.01)
                continue

            if not self.time_share_config.enabled:
                if not cache.stream_ids():
                    try:
                        frame_data = self.frame_source.get(timeout=0.1)
                    except Exception:
                        continue
                    frame_context = FrameContext.from_payload(frame_data)
                    subsample = int(self.config.get("detection_inference_subsample", 1))
                    if subsample > 1 and (frame_context.frame_number % subsample != 0):
                        continue
                    if self.enabled_streams is not None and frame_context.stream_id not in self.enabled_streams:
                        continue
                    cache.update(frame_data)

            due_streams = cache.pick_due_streams(self.enabled_streams)
            if not due_streams:
                time.sleep(self.time_share_config.tick_sleep_s)
                continue

            for stream_id in due_streams:
                frame_data = cache.get_latest(stream_id)
                if not frame_data:
                    continue
                frame_context = FrameContext.from_payload(frame_data)
                cache.mark_inferred(stream_id)
                try:
                    inference_outputs, raw_outputs = self._resolve_inference_outputs(frame_context)
                    result = self.algorithm.process(frame_context, inference_outputs, self.state)
                    payload = result.to_dict()
                    self._attach_coord_size(payload, frame_context)
                    if raw_outputs:
                        payload["raw_model_detections"] = raw_outputs
                    self._send_result(payload)
                except Exception:
                    continue

    def stop(self):
        self.running = False
        self.model_manager.close()
