import json

from app.application.orchestrator import SystemController, SystemOrchestrator


def base_config():
    models = {
        "fall_detection": "fall.bmodel",
        "ventilator_equipment": "ventilator.bmodel",
        "ventilator_helmet": "helmet.bmodel",
        "fight_detection": "fight.bmodel",
        "crowd_person": "helmet.bmodel",
        "helmet_detection": "helmet.bmodel",
        "window_door_inside": "window_in.bmodel",
        "window_door_outside": "window_out.bmodel",
    }
    return {
        "fps": 20,
        "models": models,
        "model_types": {key: "yolo26_int8" for key in models},
        "queue_sizes": {"frame_queue": 1, "result_queue": 1, "display_queue": 1},
        "output": {"video_output_dir": "alarm_videos"},
        "bm1684x": {},
        "video_streams": [
            {"name": "通道1", "source_type": "rtsp", "ip": "rtsp://one"},
            {"name": "通道2", "source_type": "rtsp", "ip": "rtsp://two"},
        ],
        "tasks": [
            {"id": "t1", "name": "旧任务1", "stream_index": 1, "detectors": ["fall", "helmet"]},
            {"id": "t2", "name": "旧任务2", "stream_index": 2, "detectors": ["fight"]},
        ],
        "batch_scheduler": {"enabled": True},
    }


def make_controller(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(base_config(), ensure_ascii=False), encoding="utf-8")
    return SystemController(config_path)


def test_controller_accepts_multi_detector_task(tmp_path):
    controller = make_controller(tmp_path)

    result = controller.add_task(
        {"name": "多算法任务", "stream_index": 1, "detectors": ["fall", "helmet"]}
    )

    assert result["status"] == "success"
    tasks = controller.refresh_config()["tasks"]
    assert tasks == [
        {
            "id": result["id"],
            "name": "多算法任务",
            "stream_index": 1,
            "detectors": ["fall", "helmet"],
        }
    ]


def test_controller_saves_single_detector_task_as_only_task(tmp_path):
    controller = make_controller(tmp_path)

    result = controller.add_task({"name": "安全帽", "stream_index": 1, "detectors": ["helmet"]})

    assert result["status"] == "success"
    tasks = controller.refresh_config()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["stream_index"] == 1
    assert tasks[0]["detectors"] == ["helmet"]


def test_controller_saves_up_to_four_valid_video_streams(tmp_path):
    controller = make_controller(tmp_path)

    result = controller.save_video_streams(
        {
            "streams": [
                {"name": "空", "source_type": "rtsp", "source": ""},
                {"name": "入口", "source_type": "rtsp", "source": "rtsp://entry"},
                {"name": "出口", "source_type": "rtsp", "source": "rtsp://exit"},
            ]
        }
    )

    assert result["status"] == "success"
    streams = controller.refresh_config()["video_streams"]
    assert len(streams) == 2
    assert streams[0]["name"] == "入口"
    assert streams[0]["source"] == "rtsp://entry"
    assert streams[1]["name"] == "出口"
    assert streams[1]["source"] == "rtsp://exit"


def test_start_task_keeps_only_one_running_detector():
    orchestrator = object.__new__(SystemOrchestrator)
    orchestrator.config = {
        "tasks": [{"id": "t1", "stream_index": 1, "detectors": ["helmet"]}],
    }
    orchestrator.batch_scheduler_enabled = True
    orchestrator.detector_running = {"fall": True, "helmet": False}
    orchestrator.detector_streams = {"fall": {0}}
    orchestrator.task_running = {"old": True}
    stopped = []
    started = []

    def stop_detector(name):
        stopped.append(name)
        orchestrator.detector_running[name] = False
        orchestrator.detector_streams.pop(name, None)
        return True

    def start_detector(name, enabled_streams=None):
        started.append((name, set(enabled_streams or [])))
        orchestrator.detector_running[name] = True
        orchestrator.detector_streams[name] = set(enabled_streams or [])
        return True

    orchestrator.stop_detector = stop_detector
    orchestrator.start_detector = start_detector

    result = orchestrator.start_task("t1")

    assert result["status"] == "success"
    assert stopped == ["fall"]
    assert started == [("helmet", {0})]
    assert orchestrator.detector_running["fall"] is False
    assert orchestrator.detector_running["helmet"] is True
    assert orchestrator.detector_streams == {"helmet": {0}}
    assert orchestrator.task_running == {"old": False, "t1": True}


def test_start_task_reports_detector_start_failure():
    orchestrator = object.__new__(SystemOrchestrator)
    orchestrator.config = {
        "tasks": [{"id": "t1", "stream_index": 1, "detectors": ["helmet"]}],
    }
    orchestrator.detector_running = {"helmet": False}
    orchestrator.detector_streams = {}
    orchestrator.task_running = {}
    orchestrator._stop_other_detectors = lambda keep_detector: None
    orchestrator.start_detector = lambda name, enabled_streams=None: False

    result = orchestrator.start_task("t1")

    assert result["status"] == "error"
    assert "启动失败" in result["message"]
    assert orchestrator.detector_running["helmet"] is False
    assert orchestrator.detector_streams == {}
    assert orchestrator.task_running == {"t1": False}


def test_update_running_task_hot_switches_detector(tmp_path):
    controller = make_controller(tmp_path)
    config = controller.refresh_config()
    config["tasks"] = [{"id": "t1", "name": "跌倒", "stream_index": 1, "detectors": ["fall"]}]
    controller.config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    controller.refresh_config()

    class FakeScheduler:
        running = True

        def __init__(self, config):
            self.config = config
            self.task_running = {"t1": True}
            self.started = []

        def start_task(self, task_id):
            self.started.append((task_id, self.config["tasks"][0]["detectors"][0]))
            return {"status": "success", "message": "任务已启动"}

    controller.scheduler = FakeScheduler(controller.config)

    result = controller.update_task(
        {"id": "t1", "name": "安全帽", "stream_index": 1, "detectors": ["helmet"]}
    )

    assert result["status"] == "success"
    assert "算法已切换" in result["message"]
    assert controller.scheduler.started == [("t1", "helmet")]
    assert controller.scheduler.config["tasks"][0]["detectors"] == ["helmet"]
    assert controller.refresh_config()["tasks"][0]["detectors"] == ["helmet"]


def test_update_task_ignores_algorithm_params_payload(tmp_path):
    controller = make_controller(tmp_path)
    config = controller.refresh_config()
    config["tasks"] = [{"id": "t1", "name": "安全帽", "stream_index": 1, "detectors": ["helmet"]}]
    config["helmet_detection"] = {"conf_threshold": 0.25}
    controller.config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    controller.refresh_config()

    class FakeScheduler:
        running = True

        def __init__(self, config):
            self.config = config
            self.task_running = {}
            self.reloaded = []

        def reload_algorithm_config(self, config):
            self.reloaded.append(config["helmet_detection"]["conf_threshold"])
            return True

    controller.scheduler = FakeScheduler(controller.config)

    result = controller.update_task(
        {
            "id": "t1",
            "name": "安全帽",
            "stream_index": 1,
            "detectors": ["helmet"],
            "algorithm_params": {"helmet_detection": {"conf_threshold": 0.66}},
        }
    )

    assert result["status"] == "success"
    assert controller.refresh_config()["helmet_detection"]["conf_threshold"] == 0.25
    assert controller.scheduler.reloaded == [0.25]
