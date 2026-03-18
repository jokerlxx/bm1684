"""
后端 Web 服务：Flask API、视频流与实时告警推送（SSE）。
通过 create_app(handlers, get_scheduler, get_alert_queue, get_hls_dir) 创建应用，与 main 解耦。
解码-处理-编码流水线：HLS 流由编码服务写入 get_hls_dir()，本层提供静态访问供前端低延迟播放。
"""

import os
import time
import json
import logging
import threading
import queue as queue_module
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_from_directory, send_file
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger('StreamService')

# 前端静态目录（backend 的上一级下的 frontend）
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# SSE 告警订阅者队列列表（主进程内广播）
_alert_subscribers = []
_status_subscribers = []


def create_app(handlers, get_scheduler, get_alert_queue=None, get_hls_dir=None):
    """
    创建 Flask 应用（事件驱动微服务架构 - Web 层）。
    :param handlers: 路由实现字典
    :param get_scheduler: 无参可调用对象，返回当前 MainScheduler 实例（可为 None）
    :param get_alert_queue: 无参可调用对象，返回告警事件队列（mp.Queue）；可为 None
    :param get_hls_dir: 无参可调用对象，返回 HLS 输出目录绝对路径（用于解码-处理-编码流水线低延迟播放）；可为 None
    """
    app = Flask(__name__)

    def _safe_get_status():
        """获取系统状态（用于 SSE 推送）。"""
        try:
            return handlers['status']()
        except Exception as e:
            return {'system_running': False, 'detectors': {}, 'error': str(e)}

    def _alert_broadcast_worker():
        """后台线程：从告警队列取事件并广播给所有 SSE 订阅者。"""
        while True:
            q = get_alert_queue() if get_alert_queue else None
            if q is None:
                time.sleep(1.0)
                continue
            try:
                ev = q.get(timeout=1.0)
            except Exception:
                continue
            for sub in list(_alert_subscribers):
                try:
                    sub.put_nowait(ev)
                except Exception:
                    pass

    if get_alert_queue:
        t = threading.Thread(target=_alert_broadcast_worker, daemon=True)
        t.start()
        logger.info("Alert broadcast thread started (SSE)")

    def _status_broadcast_worker():
        """后台线程：定期广播系统状态（供前端实时状态条使用）。"""
        last_payload = None
        while True:
            payload = _safe_get_status()
            # 只在有订阅者时进行（降低无用开销）
            if _status_subscribers:
                try:
                    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                except Exception:
                    text = json.dumps({'system_running': False, 'detectors': {}}, ensure_ascii=False, sort_keys=True)
                # 降噪：状态不变时不广播（但仍会由 SSE keepalive 维持连接）
                if text != last_payload:
                    last_payload = text
                    for sub in list(_status_subscribers):
                        try:
                            sub.put_nowait(payload)
                        except Exception:
                            pass
            time.sleep(1.0)

    st = threading.Thread(target=_status_broadcast_worker, daemon=True)
    st.start()
    logger.info("Status broadcast thread started (SSE)")

    def generate_frames():
        """生成 MJPEG 视频帧流。"""
        logger.info("🎥 Video feed started")
        frame_count = 0
        try:
            while True:
                scheduler = get_scheduler()
                if scheduler is None or not getattr(scheduler, 'running', False):
                    # 使用与运行时常规输出相近的分辨率，避免停止时预览区域显得过小
                    blank_h, blank_w = 720, 1280
                    blank_frame = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
                    # 统一预览底色：深蓝（与九宫格空格一致）
                    blank_frame[:] = (42, 23, 15)  # BGR for #0f172a
                    ret, buffer = cv2.imencode('.jpg', blank_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.1)
                    continue
                try:
                    # 低延迟策略：只取“最新一帧”，把队列里积压的旧帧全部丢弃
                    frame = None
                    while True:
                        try:
                            frame = scheduler.output_queue.get_nowait()
                        except Exception:
                            break
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    if frame is None or not isinstance(frame, np.ndarray):
                        continue
                    frame_count += 1
                    # 降低 MJPEG 编码质量以提高吞吐（HLS 为主；MJPEG 作为回退）
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    if "Empty" not in str(e):
                        logger.error("Error generating frame: %s", e)
                    time.sleep(0.01)
        except GeneratorExit:
            logger.info("🎥 Video feed stopped (generated %s frames)", frame_count)
        except Exception as e:
            logger.error("Video feed error: %s", e)

    @app.route('/')
    def index():
        """主页：返回前端静态页面。"""
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.route('/video_feed')
    def video_feed():
        """视频流端点（MJPEG，作为 HLS 不可用时的回退）。"""
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    # 解码-处理-编码流水线：HLS 流静态服务，供前端低延迟播放已标注视频
    if get_hls_dir:
        @app.route('/hls/<path:filename>')
        def serve_hls(filename):
            """
            HLS 静态文件服务（低延迟 + 禁缓存）。
            现象修复：浏览器对 m3u8/ts 发生缓存或 304 命中时，可能回退播放到旧分片，表现为“突然显示很久以前画面再跳回当前”。
            """
            try:
                hls_dir = get_hls_dir()
                if hls_dir is None or not os.path.isdir(hls_dir):
                    return Response("HLS not ready", status=404)

                # 强制禁缓存：避免 m3u8/ts 被缓存或条件请求命中旧内容（304）
                resp = send_from_directory(
                    hls_dir,
                    filename,
                    max_age=0,
                )
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"

                # 明确 Content-Type（部分播放器会更稳定）
                lower = filename.lower()
                if lower.endswith(".m3u8"):
                    resp.headers["Content-Type"] = "application/vnd.apple.mpegurl"
                elif lower.endswith(".ts"):
                    resp.headers["Content-Type"] = "video/mp2t"
                return resp
            except Exception as e:
                logger.warning("Serve HLS file %s: %s", filename, e)
                return Response(str(e), status=404)

    @app.route('/api/start', methods=['POST'])
    def api_start():
        result = handlers['start']()
        return jsonify(result)

    @app.route('/api/stop', methods=['POST'])
    def api_stop():
        result = handlers['stop']()
        return jsonify(result)

    @app.route('/api/toggle_detector', methods=['POST'])
    def api_toggle_detector():
        data = request.get_json() or {}
        result = handlers['toggle_detector'](data)
        return jsonify(result)

    @app.route('/api/status')
    def api_status():
        result = handlers['status']()
        return jsonify(result)

    # 实时告警推送：Server-Sent Events
    @app.route('/api/events')
    def api_events():
        """SSE 流：实时推送告警事件，供前端 EventSource 消费。"""
        def gen():
            client_q = queue_module.Queue()
            _alert_subscribers.append(client_q)
            try:
                while True:
                    try:
                        ev = client_q.get(timeout=30.0)
                        yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    except queue_module.Empty:
                        yield ": keepalive\n\n"
            finally:
                if client_q in _alert_subscribers:
                    _alert_subscribers.remove(client_q)

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 实时状态推送：Server-Sent Events（系统运行状态 + 各检测器状态）
    @app.route('/api/status_events')
    def api_status_events():
        """SSE 流：实时推送系统状态，供前端状态条消费。"""
        def gen():
            client_q = queue_module.Queue()
            _status_subscribers.append(client_q)
            # 首次立即推送一次
            try:
                yield "data: " + json.dumps(_safe_get_status(), ensure_ascii=False) + "\n\n"
            except Exception:
                pass
            try:
                while True:
                    try:
                        ev = client_q.get(timeout=30.0)
                        yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    except queue_module.Empty:
                        yield ": keepalive\n\n"
            finally:
                if client_q in _status_subscribers:
                    _status_subscribers.remove(client_q)

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 历史数据回溯：告警文件列表（含 category）
    if "alerts_history" in handlers:
        @app.route('/api/alerts/history')
        def api_alerts_history():
            result = handlers["alerts_history"]()
            return jsonify(result)

    # 告警文件下载：安全地按文件名提供下载
    if "alerts_file" in handlers:
        @app.route('/api/alerts/file')
        def api_alerts_file():
            result = handlers["alerts_file"](request)
            if isinstance(result, tuple):
                path, download_name = result
                return send_file(path, as_attachment=True, download_name=download_name)
            return jsonify(result)

    # 清除过期告警文件（超过 7 天）
    if "alerts_cleanup" in handlers:
        @app.route('/api/alerts/cleanup', methods=['POST'])
        def api_alerts_cleanup():
            result = handlers["alerts_cleanup"](request)
            return jsonify(result)

    # 清除全部告警文件
    if "alerts_clear" in handlers:
        @app.route('/api/alerts/clear', methods=['POST'])
        def api_alerts_clear():
            result = handlers["alerts_clear"](request)
            return jsonify(result)

    # 批量删除告警文件
    if "alerts_delete_batch" in handlers:
        @app.route('/api/alerts/delete_batch', methods=['POST'])
        def api_alerts_delete_batch():
            data = request.get_json() or {}
            result = handlers["alerts_delete_batch"](data)
            return jsonify(result)

    # 视频流服务：获取 / 保存 1~9 路视频流配置（名称 + IP/RTSP）
    if "video_streams_get" in handlers:
        @app.route('/api/video_streams')
        def api_video_streams_get():
            result = handlers["video_streams_get"]()
            return jsonify(result)

    if "video_streams_save" in handlers:
        @app.route('/api/video_streams', methods=['POST'])
        def api_video_streams_save():
            data = request.get_json() or {}
            result = handlers["video_streams_save"](data)
            return jsonify(result)

    # 任务管理：获取 / 新增 / 更新 / 删除 / 启动 / 停止
    if "tasks_get" in handlers:
        @app.route('/api/tasks')
        def api_tasks_get():
            result = handlers["tasks_get"]()
            if result.get("status") == "success" and result.get("tasks"):
                s = get_scheduler() if get_scheduler else None
                running = s.get_task_running_states() if (s and callable(getattr(s, 'get_task_running_states', None))) else {}
                for t in result["tasks"]:
                    t["running"] = running.get(str(t.get("id")), False)
            return jsonify(result)

    if "tasks_add" in handlers:
        @app.route('/api/tasks', methods=['POST'])
        def api_tasks_add():
            data = request.get_json() or {}
            result = handlers["tasks_add"](data)
            return jsonify(result)

    if "tasks_update" in handlers:
        @app.route('/api/tasks/update', methods=['POST'])
        def api_tasks_update():
            data = request.get_json() or {}
            result = handlers["tasks_update"](data)
            return jsonify(result)

    if "tasks_delete" in handlers:
        @app.route('/api/tasks/delete', methods=['POST'])
        def api_tasks_delete():
            data = request.get_json() or {}
            result = handlers["tasks_delete"](data)
            return jsonify(result)

    # 任务单独启动 / 停止
    if get_scheduler:
        @app.route('/api/tasks/start', methods=['POST'])
        def api_tasks_start():
            data = request.get_json() or {}
            task_id = data.get('id')
            if not task_id:
                return jsonify({'status': 'error', 'message': '缺少任务 id'})
            s = get_scheduler()
            if not s or not getattr(s, 'start_task', None):
                return jsonify({'status': 'error', 'message': '系统未就绪'})
            result = s.start_task(task_id)
            return jsonify(result)

        @app.route('/api/tasks/stop', methods=['POST'])
        def api_tasks_stop():
            data = request.get_json() or {}
            task_id = data.get('id')
            if not task_id:
                return jsonify({'status': 'error', 'message': '缺少任务 id'})
            s = get_scheduler()
            if not s or not getattr(s, 'stop_task', None):
                return jsonify({'status': 'error', 'message': '系统未就绪'})
            result = s.stop_task(task_id)
            return jsonify(result)

    if "preview_mode_set" in handlers:
        @app.route('/api/preview_mode', methods=['POST'])
        def api_preview_mode():
            data = request.get_json() or {}
            result = handlers["preview_mode_set"](data)
            return jsonify(result)

    return app
