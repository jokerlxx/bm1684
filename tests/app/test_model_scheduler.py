import queue

from app.pipeline.messages import InferenceResult
from app.pipeline.model_scheduler import BatchModelScheduler


class EmptyFrameSource:
    def get_nowait(self):
        raise queue.Empty()


class EmptyRegistry:
    def all(self):
        return {}


class DummyModelManager:
    def __init__(self):
        self.loaded_specs = []
        self.close_count = 0

    def load_all(self, specs):
        self.loaded_specs = list(specs)
        return {}

    def close(self):
        self.close_count += 1


def make_scheduler(preview_status):
    return BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=queue.Queue(),
        config={
            "detection_timeshare": {"enabled": True},
            "batch_scheduler": {
                "preview_backpressure_enabled": True,
                "preview_backpressure_fps": 18.0,
                "preview_backpressure_consecutive": 3,
            },
        },
        registry=EmptyRegistry(),
        model_manager=DummyModelManager(),
        preview_status=preview_status,
    )


def test_preview_backpressure_requires_consecutive_low_fps_samples():
    scheduler = make_scheduler({"compose_fps": 12.0})

    assert scheduler._preview_under_pressure() is False
    assert scheduler._preview_under_pressure() is False
    assert scheduler._preview_under_pressure() is True

    scheduler.preview_status["compose_fps"] = 20.0
    assert scheduler._preview_under_pressure() is False


def test_scheduler_metrics_accumulate_inference_samples():
    scheduler = make_scheduler({"compose_fps": 20.0})

    scheduler._record_inference_metrics("helmet_detection", batch_size=4, total_ms=12.5)
    scheduler._record_inference_metrics("helmet_detection", batch_size=2, total_ms=15.5)
    scheduler._record_job_served(type("Job", (), {"stream_id": 0, "detector_name": "helmet"})())

    assert scheduler._metrics["inferences"]["helmet_detection"] == 2
    assert scheduler._metrics["batches"]["helmet_detection"] == 6
    assert scheduler._metrics["batch_sizes"]["helmet_detection"] == [4, 2]
    assert scheduler._metrics["job_served"]["ch1:helmet"] == 1
    assert scheduler._metrics["infer_ms"]["helmet_detection"] == [12.5, 15.5]


def test_active_models_follow_single_detector_selection():
    scheduler = make_scheduler({"compose_fps": 20.0})

    scheduler.enabled_detectors = {"helmet"}
    assert scheduler._active_model_keys() == ["helmet_detection"]

    scheduler.enabled_detectors = {"fall"}
    assert scheduler._active_model_keys() == ["fall_detection"]

    scheduler.enabled_detectors = {"ventilator"}
    assert scheduler._active_model_keys() == ["helmet_detection", "ventilator_equipment"]


def test_model_specs_merge_advanced_algorithm_thresholds():
    scheduler = BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=queue.Queue(),
        config={
            "models": {"helmet_detection": "helmet.bmodel"},
            "model_types": {"helmet_detection": "yolo26_int8"},
            "bm1684x": {"device_id": 0},
            "helmet_detection": {"conf_threshold": 0.25},
            "advanced_algorithm_params": {
                "helmet_detection": {
                    "iou_threshold": 0.3,
                    "max_age": 2,
                }
            },
        },
        registry=EmptyRegistry(),
        model_manager=DummyModelManager(),
        preview_status={"compose_fps": 20.0},
    )
    scheduler.enabled_detectors = {"helmet"}

    specs = scheduler._build_model_specs()

    assert specs[0].thresholds["conf_threshold"] == 0.25
    assert specs[0].thresholds["iou_threshold"] == 0.3
    assert specs[0].thresholds["max_age"] == 2


def test_reload_config_command_rebuilds_active_model_specs():
    manager = DummyModelManager()
    control_queue = queue.Queue()
    scheduler = BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=control_queue,
        config={
            "models": {"helmet_detection": "helmet.bmodel"},
            "model_types": {"helmet_detection": "yolo26_int8"},
            "bm1684x": {"device_id": 0},
            "helmet_detection": {"conf_threshold": 0.25},
        },
        registry=EmptyRegistry(),
        model_manager=manager,
        preview_status={"compose_fps": 20.0},
    )
    scheduler.enabled_detectors = {"helmet"}

    control_queue.put(
        {
            "cmd": "reload_config",
            "config": {
                "models": {"helmet_detection": "helmet.bmodel"},
                "model_types": {"helmet_detection": "yolo26_int8"},
                "bm1684x": {"device_id": 0},
                "helmet_detection": {"conf_threshold": 0.72},
            },
        }
    )
    scheduler._check_control_commands()

    assert scheduler.config["helmet_detection"]["conf_threshold"] == 0.72
    assert manager.loaded_specs[0].thresholds["conf_threshold"] == 0.72


class RecordingModelManager:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []
        self.loaded_specs = []

    def load_all(self, specs):
        self.loaded_specs = list(specs)
        return {}

    def infer_batch(self, model_key, frames):
        frame_list = list(frames)
        batch_size = len(frame_list) or 1
        self.calls.append((model_key, batch_size))
        return [
            InferenceResult(model_key=model_key, detections=[], raw_meta={"batch_size": 1})
            for _ in frame_list
        ]

    def close(self):
        pass


def test_job_inference_runs_only_bound_model_per_stream():
    scheduler = BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=queue.Queue(),
        config={"detection_timeshare": {"enabled": True}},
        registry=EmptyRegistry(),
        model_manager=RecordingModelManager(),
        preview_status={"compose_fps": 20.0},
    )
    scheduler.enabled_detectors = {"fight", "fall"}
    scheduler.detector_streams = {"fight": {0}, "fall": {1}}
    scheduler._rebuild_jobs()

    fight_job = next(job for job in scheduler.job_table.active_jobs() if job.detector_name == "fight")
    fall_job = next(job for job in scheduler.job_table.active_jobs() if job.detector_name == "fall")

    class DummyFrame:
        shape = (360, 640, 3)

    context_fight = type("Ctx", (), {"frame": DummyFrame(), "stream_id": 0})()
    context_fall = type("Ctx", (), {"frame": DummyFrame(), "stream_id": 1})()

    assert scheduler._run_model_batch("fight_detection", [(fight_job, context_fight)])
    assert scheduler._run_model_batch("fall_detection", [(fall_job, context_fall)])

    manager = scheduler.model_manager
    assert manager.calls == [("fight_detection", 1), ("fall_detection", 1)]


def test_model_batch_runs_multiple_streams_together():
    scheduler = BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=queue.Queue(),
        config={"detection_timeshare": {"enabled": True}},
        registry=EmptyRegistry(),
        model_manager=RecordingModelManager(),
        preview_status={"compose_fps": 20.0},
    )
    scheduler.enabled_detectors = {"fight"}
    scheduler.detector_streams = {"fight": {0, 2}}
    scheduler._rebuild_jobs()

    jobs = scheduler.job_table.active_jobs()
    class DummyFrame:
        shape = (360, 640, 3)

    contexts = [
        (job, type("Ctx", (), {"frame": DummyFrame(), "stream_id": job.stream_id})())
        for job in jobs
    ]
    assert scheduler._run_model_batch("fight_detection", contexts)
    manager = scheduler.model_manager
    assert manager.calls == [("fight_detection", 2)]


def test_rebuild_jobs_from_enable_command():
    control_queue = queue.Queue()
    scheduler = BatchModelScheduler(
        frame_source=EmptyFrameSource(),
        result_queues={},
        control_queue=control_queue,
        config={"detection_timeshare": {"enabled": True}},
        registry=EmptyRegistry(),
        model_manager=RecordingModelManager(),
        preview_status={"compose_fps": 20.0},
    )
    control_queue.put({"cmd": "enable", "detector": "fight", "stream_ids": [0, 2]})
    scheduler._check_control_commands()

    jobs = scheduler.job_table.active_jobs()
    assert len(jobs) == 2
    assert {job.stream_id for job in jobs} == {0, 2}
    assert all(job.detector_name == "fight" for job in jobs)
