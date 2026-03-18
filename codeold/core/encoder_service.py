"""
编码服务 (Encoder Service) - 解码-处理-编码流水线中的编码阶段
职责：
1. 从已标注帧队列读取图像帧（由展示服务绘制检测框后的帧）
2. 使用 ffmpeg 将帧编码为 HLS 视频流（低延迟）
3. 输出 .m3u8 与 .ts 片段供 Web 通过 HLS 播放，实现检测结果与画面同步
"""

import os
import sys
import time
import logging
import signal
import subprocess
import multiprocessing as mp
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [EncoderService] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EncoderService:
    """HLS 编码服务 - 将已标注帧编码为 HLS 流"""

    def __init__(self, frame_queue, control_queue, output_dir, fps=25,
                 hls_time=1, hls_list_size=5):
        """
        Args:
            frame_queue: 已标注帧队列（展示服务写入）
            control_queue: 控制命令队列
            output_dir: HLS 输出目录（写入 live.m3u8 与 segment_*.ts）
            fps: 帧率
            hls_time: 每个 HLS 片段时长（秒），越小延迟越低
            hls_list_size: 播放列表保留的片段数量
        """
        self.frame_queue = frame_queue
        self.control_queue = control_queue
        self.output_dir = Path(output_dir)
        self.fps = max(1, int(fps))
        self.hls_time = max(1, min(4, int(hls_time)))
        self.hls_list_size = max(3, min(20, int(hls_list_size)))
        self.running = False
        self._proc = None
        self._width = self._height = None
        self._keyint = max(1, self.fps * self.hls_time)
        # 控制 ffmpeg 重启频率，避免频繁拉起导致 CPU 抖动
        self._last_restart_time = 0.0

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _start_ffmpeg(self, width, height):
        """启动 ffmpeg：stdin 为 rawvideo bgr24，输出 HLS"""
        self._ensure_output_dir()
        seg_pattern = str(self.output_dir / 'segment_%03d.ts')
        m3u8_path = str(self.output_dir / 'live.m3u8')
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-x264-params', f'keyint={self._keyint}:min-keyint={self._keyint}',
            '-f', 'hls',
            '-hls_time', str(self.hls_time),
            '-hls_list_size', str(self.hls_list_size),
            '-hls_flags', 'delete_segments+append_list',
            '-hls_segment_filename', seg_pattern,
            m3u8_path
        ]
        try:
            now = time.time()
            self._last_restart_time = now
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            logger.info("HLS encoder started: %s, %dx%d @ %dfps", m3u8_path, width, height, self.fps)
            return True
        except FileNotFoundError:
            logger.error("ffmpeg not found; install ffmpeg for HLS encoding")
            return False
        except Exception as e:
            logger.error("Failed to start ffmpeg: %s", e)
            return False

    def _check_control(self):
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                if cmd == 'stop':
                    self.running = False
        except Exception:
            pass

    def run(self):
        """主循环：从队列取帧，写入 ffmpeg stdin"""
        logger.info("Encoder service starting (HLS)...")
        self.running = True
        frame_count = 0
        last_log = 0

        while self.running:
            self._check_control()
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except Exception:
                continue

            if frame is None or not isinstance(frame, np.ndarray):
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue

            h, w = frame.shape[:2]
            # 分辨率变化或还未启动 ffmpeg 时，重新/首次启动编码进程
            if self._width != w or self._height != h or self._proc is None:
                # 若刚发生过 pipe 断开，短暂冷却，避免疯狂重启 ffmpeg
                if self._proc is None and self._last_restart_time:
                    if time.time() - self._last_restart_time < 1.0:
                        continue
                if self._proc is not None:
                    try:
                        self._proc.stdin.close()
                        self._proc.terminate()
                        self._proc.wait(timeout=2)
                    except Exception:
                        pass
                    self._proc = None
                self._width, self._height = w, h
                if not self._start_ffmpeg(w, h):
                    continue

            try:
                # 保护性判断，避免 NoneType.stdin 错误
                if self._proc is None or self._proc.stdin is None:
                    logger.warning("ffmpeg process is not ready, skipping one frame")
                    continue
                self._proc.stdin.write(frame.tobytes())
                frame_count += 1
                if frame_count - last_log >= 100:
                    logger.info("HLS encoder: %d frames written", frame_count)
                    last_log = frame_count
            except BrokenPipeError:
                logger.warning("ffmpeg pipe broken, will cooldown before restart...")
                self._proc = None
                self._last_restart_time = time.time()
            except Exception as e:
                logger.warning("Encoder write error: %s", e)
                self._proc = None

        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        logger.info("Encoder service stopped (wrote %d frames)", frame_count)


def run_encoder_service(frame_queue, control_queue, output_dir, fps=25,
                        hls_time=1, hls_list_size=5):
    """进程入口：从已标注帧队列编码为 HLS"""
    service = EncoderService(
        frame_queue, control_queue, output_dir, fps=fps,
        hls_time=hls_time, hls_list_size=hls_list_size
    )
    service.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HLS Encoder Service')
    parser.add_argument('--output-dir', type=str, default='./hls_output', help='HLS output directory')
    parser.add_argument('--fps', type=int, default=25, help='Frame rate')
    parser.add_argument('--hls-time', type=int, default=1, help='Segment duration (seconds)')
    args = parser.parse_args()
    q = mp.Queue(maxsize=5)
    ctrl = mp.Queue()
    run_encoder_service(q, ctrl, args.output_dir, args.fps, args.hls_time)
