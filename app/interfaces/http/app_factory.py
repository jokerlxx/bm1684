"""
Layered HTTP app factory.

The routes remain backward compatible with the previous backend/app.py module.
"""

import json
import logging
import os
import queue as queue_module
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from app.application.agent.safety_agent import SafetyAgent
from app.infrastructure.logging import log_push_fps


logger = logging.getLogger("StreamService")
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))

_alert_subscribers = []
_status_subscribers = []


class _LatestMjpegFrameBuffer:
    """Single consumer for display output; HTTP clients only read the cached latest frame."""

    def __init__(self, get_scheduler, queue_attr="output_queue", name="MJPEGFrameFanout", log_name="MJPEG推流"):
        self._get_scheduler = get_scheduler
        self._queue_attr = queue_attr
        self._log_name = log_name
        self._condition = threading.Condition()
        self._frame_bytes = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name=name)
        self._thread.start()

    def _publish(self, frame_bytes):
        if not frame_bytes:
            return
        with self._condition:
            self._frame_bytes = bytes(frame_bytes)
            self._seq += 1
            self._condition.notify_all()

    def wait_next(self, last_seq, timeout=1.0):
        with self._condition:
            if self._seq == last_seq:
                self._condition.wait(timeout=timeout)
            return self._seq, self._frame_bytes

    def _worker(self):
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_log_interval = 60
        while self._running:
            scheduler = self._get_scheduler()
            if scheduler is None or not getattr(scheduler, "running", False):
                time.sleep(0.1)
                continue
            source_queue = getattr(scheduler, self._queue_attr, None)
            if source_queue is None:
                time.sleep(0.1)
                continue
            try:
                frame = source_queue.get(timeout=0.5)
                while True:
                    try:
                        frame = source_queue.get_nowait()
                    except queue_module.Empty:
                        break
                    except Exception:
                        break
                if isinstance(frame, (bytes, bytearray, memoryview)):
                    frame_bytes = bytes(frame)
                elif isinstance(frame, np.ndarray):
                    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        continue
                    frame_bytes = buffer.tobytes()
                else:
                    continue
                self._publish(frame_bytes)
                frame_count += 1
                fps_frame_count += 1
                if fps_frame_count >= fps_log_interval:
                    elapsed = time.time() - fps_start_time
                    actual_fps = (fps_frame_count / elapsed) if elapsed > 0 else 0.0
                    try:
                        queue_size = source_queue.qsize()
                    except Exception:
                        queue_size = 0
                    logger.info("%s frames=%d fps=%.2f queue=%d", self._thread.name, frame_count, actual_fps, queue_size)
                    log_push_fps(self._log_name, actual_fps, queue_size=queue_size)
                    fps_start_time = time.time()
                    fps_frame_count = 0
            except queue_module.Empty:
                continue
            except Exception as exc:
                if "Empty" not in str(exc):
                    logger.error("Error reading MJPEG frame: %s", exc)
                time.sleep(0.05)


def create_app(handlers, get_scheduler, get_alert_queue=None):
    app = Flask(__name__)
    mjpeg_frames = _LatestMjpegFrameBuffer(get_scheduler)
    blank_frame_bytes = None

    def _safe_get_status():
        try:
            return handlers["status"]()
        except Exception as exc:
            return {"system_running": False, "detectors": {}, "error": str(exc)}

    def _alert_broadcast_worker():
        while True:
            alert_queue = get_alert_queue() if get_alert_queue else None
            if alert_queue is None:
                time.sleep(1.0)
                continue
            try:
                event = alert_queue.get(timeout=1.0)
            except Exception:
                continue
            for subscriber in list(_alert_subscribers):
                try:
                    subscriber.put_nowait(event)
                except Exception:
                    pass

    if get_alert_queue:
        threading.Thread(target=_alert_broadcast_worker, daemon=True).start()

    def _status_broadcast_worker():
        last_payload = None
        while True:
            payload = _safe_get_status()
            if _status_subscribers:
                try:
                    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                except Exception:
                    text = json.dumps({"system_running": False, "detectors": {}}, ensure_ascii=False, sort_keys=True)
                if text != last_payload:
                    last_payload = text
                    for subscriber in list(_status_subscribers):
                        try:
                            subscriber.put_nowait(payload)
                        except Exception:
                            pass
            time.sleep(1.0)

    threading.Thread(target=_status_broadcast_worker, daemon=True).start()

    def generate_frames(frame_buffer=mjpeg_frames, stream_name="Video feed"):
        logger.info("%s started", stream_name)
        frame_count = 0
        last_seq = 0
        nonlocal blank_frame_bytes
        try:
            while True:
                scheduler = get_scheduler()
                if scheduler is None or not getattr(scheduler, "running", False):
                    if blank_frame_bytes is None:
                        # 生成带四宫格线的空白帧，确保预览页始终显示四宫格
                        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        blank_frame[:] = (42, 23, 15)
                        # 四宫格分割线（白色半透明）
                        mid_x, mid_y = 640, 360
                        cv2.line(blank_frame, (mid_x, 0), (mid_x, 719), (80, 80, 80), 2)
                        cv2.line(blank_frame, (0, mid_y), (1279, mid_y), (80, 80, 80), 2)
                        # 通道标签
                        cells = [(10, 30, 1), (mid_x + 10, 30, 2), (10, mid_y + 30, 3), (mid_x + 10, mid_y + 30, 4)]
                        for cx, cy, ch in cells:
                            cv2.putText(blank_frame, f"Channel {ch}", (cx, cy),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
                        # 状态提示
                        cv2.putText(blank_frame, "System Stopped - Click Start to Begin",
                                   (320, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                        ok, buffer = cv2.imencode(".jpg", blank_frame)
                        if ok:
                            blank_frame_bytes = buffer.tobytes()
                    if blank_frame_bytes:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank_frame_bytes + b"\r\n"
                    time.sleep(0.1)
                    continue

                try:
                    seq, frame_bytes = frame_buffer.wait_next(last_seq, timeout=1.0)
                    if frame_bytes is None or seq == last_seq:
                        continue
                    last_seq = seq
                    frame_count += 1
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                except Exception as exc:
                    if "Empty" not in str(exc):
                        logger.error("Error generating frame: %s", exc)
                    time.sleep(0.01)
        except GeneratorExit:
            logger.info("%s stopped after %s frames", stream_name, frame_count)

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/preview/<path:filename>")
    def preview_hls_file(filename):
        scheduler = get_scheduler() if get_scheduler else None
        if scheduler is None or not hasattr(scheduler, "get_preview_hls_dir"):
            return jsonify({"status": "error", "message": "预览目录不可用"}), 503
        return send_from_directory(scheduler.get_preview_hls_dir(), filename, conditional=True)

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(mjpeg_frames, "Video feed"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/start", methods=["POST"])
    def api_start():
        return jsonify(handlers["start"]())

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        return jsonify(handlers["stop"]())

    @app.route("/api/toggle_detector", methods=["POST"])
    def api_toggle_detector():
        return jsonify(handlers["toggle_detector"](request.get_json() or {}))

    @app.route("/api/status")
    def api_status():
        return jsonify(handlers["status"]())

    @app.route("/api/events")
    def api_events():
        def gen():
            client_queue = queue_module.Queue()
            _alert_subscribers.append(client_queue)
            try:
                while True:
                    try:
                        event = client_queue.get(timeout=30.0)
                        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    except queue_module.Empty:
                        yield ": keepalive\n\n"
            finally:
                if client_queue in _alert_subscribers:
                    _alert_subscribers.remove(client_queue)

        return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/status_events")
    def api_status_events():
        def gen():
            client_queue = queue_module.Queue()
            _status_subscribers.append(client_queue)
            try:
                yield "data: " + json.dumps(_safe_get_status(), ensure_ascii=False) + "\n\n"
            except Exception:
                pass
            try:
                while True:
                    try:
                        event = client_queue.get(timeout=30.0)
                        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    except queue_module.Empty:
                        yield ": keepalive\n\n"
            finally:
                if client_queue in _status_subscribers:
                    _status_subscribers.remove(client_queue)

        return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if "alerts_history" in handlers:
        @app.route("/api/alerts/history")
        def api_alerts_history():
            return jsonify(handlers["alerts_history"]())

    if "alerts_file" in handlers:
        @app.route("/api/alerts/file")
        def api_alerts_file():
            result = handlers["alerts_file"](request)
            if isinstance(result, tuple):
                path, download_name = result
                return send_file(path, as_attachment=True, download_name=download_name)
            return jsonify(result)

    if "alerts_cleanup" in handlers:
        @app.route("/api/alerts/cleanup", methods=["POST"])
        def api_alerts_cleanup():
            return jsonify(handlers["alerts_cleanup"](request))

    if "alerts_clear" in handlers:
        @app.route("/api/alerts/clear", methods=["POST"])
        def api_alerts_clear():
            return jsonify(handlers["alerts_clear"](request))

    if "alerts_delete_batch" in handlers:
        @app.route("/api/alerts/delete_batch", methods=["POST"])
        def api_alerts_delete_batch():
            return jsonify(handlers["alerts_delete_batch"](request.get_json() or {}))

    if "video_streams_get" in handlers:
        @app.route("/api/video_streams")
        def api_video_streams_get():
            return jsonify(handlers["video_streams_get"]())

    if "video_streams_save" in handlers:
        @app.route("/api/video_streams", methods=["POST"])
        def api_video_streams_save():
            return jsonify(handlers["video_streams_save"](request.get_json() or {}))

    if "tasks_get" in handlers:
        @app.route("/api/tasks")
        def api_tasks_get():
            result = handlers["tasks_get"]()
            if result.get("status") == "success" and result.get("tasks"):
                scheduler = get_scheduler() if get_scheduler else None
                running = scheduler.get_task_running_states() if (scheduler and callable(getattr(scheduler, "get_task_running_states", None))) else {}
                for task in result["tasks"]:
                    task["running"] = running.get(str(task.get("id")), False)
            return jsonify(result)

    if "tasks_add" in handlers:
        @app.route("/api/tasks", methods=["POST"])
        def api_tasks_add():
            return jsonify(handlers["tasks_add"](request.get_json() or {}))

    if "tasks_update" in handlers:
        @app.route("/api/tasks/update", methods=["POST"])
        def api_tasks_update():
            return jsonify(handlers["tasks_update"](request.get_json() or {}))

    if "tasks_delete" in handlers:
        @app.route("/api/tasks/delete", methods=["POST"])
        def api_tasks_delete():
            return jsonify(handlers["tasks_delete"](request.get_json() or {}))

    if get_scheduler:
        @app.route("/api/tasks/start", methods=["POST"])
        def api_tasks_start():
            data = request.get_json() or {}
            task_id = data.get("id")
            if not task_id:
                return jsonify({"status": "error", "message": "缺少任务 id"})
            scheduler = get_scheduler()
            if not scheduler or not getattr(scheduler, "start_task", None):
                return jsonify({"status": "error", "message": "系统未就绪"})
            return jsonify(scheduler.start_task(task_id))

        @app.route("/api/tasks/stop", methods=["POST"])
        def api_tasks_stop():
            data = request.get_json() or {}
            task_id = data.get("id")
            if not task_id:
                return jsonify({"status": "error", "message": "缺少任务 id"})
            scheduler = get_scheduler()
            if not scheduler or not getattr(scheduler, "stop_task", None):
                return jsonify({"status": "error", "message": "系统未就绪"})
            return jsonify(scheduler.stop_task(task_id))

    # ── AI Safety Agent ──────────────────────────────────────
    _agent_config = None
    try:
        from app.bootstrap.config import load_config
        _agent_config = load_config()
    except Exception:
        pass

    _safety_agent = SafetyAgent(
        status_handler=lambda: (handlers.get("status") or (lambda: {}))(),
        alerts_handler=lambda: (handlers.get("alerts_history") or (lambda: {"items": []}))(),
        tasks_handler=lambda: (handlers.get("tasks_get") or (lambda: {"tasks": []}))(),
        streams_handler=lambda: (handlers.get("video_streams_get") or (lambda: {"streams": []}))(),
        scheduler_getter=get_scheduler,
        config=_agent_config,
    )

    @app.route("/api/agent/chat", methods=["POST"])
    def api_agent_chat():
        try:
            data = request.get_json() or {}
            message = (data.get("message") or "").strip()
            if not message:
                return jsonify({
                    "success": False,
                    "answer": "请发送一条消息。",
                    "error": "empty message",
                })
            result = _safety_agent.chat(message)
            return jsonify(result)
        except Exception as exc:
            logger.exception("AI Safety Agent error: %s", exc)
            return jsonify({
                "success": False,
                "answer": "AI安全助手处理失败，请稍后重试。",
                "error": str(exc),
            })

    return app
