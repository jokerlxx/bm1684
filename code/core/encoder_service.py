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

try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [EncoderService] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EncoderService:
    """HLS 编码服务 - 将已标注帧编码为 HLS 流
    
    迁移说明：
    - 原实现通过 ffmpeg 子进程（libx264）进行编码与 HLS 分片。
    - 新实现优先使用 Sophon 硬件编码器（sail.Encoder，h264_bm），在本进程内做 TS 分片与 m3u8 维护。
    - 若 Sophon 不可用，则回退到原 ffmpeg 行为（以保持兼容性）。
    """

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
        self._ensure_output_dir()
        # 关键：启动时清理旧的 HLS 输出，避免前端误拉“历史分片”导致画面回跳
        self._reset_hls_output()
        self.fps = max(1, int(fps))
        self.hls_time = max(1, min(4, int(hls_time)))
        self.hls_list_size = max(3, min(20, int(hls_list_size)))
        self.running = False
        # 通用状态
        self._width = self._height = None
        self._keyint = max(1, self.fps * self.hls_time)
        # ffmpeg 相关（回退路径）
        self._proc = None
        # Sophon 相关（硬件编码 + HLS）
        self._use_sophon = SOPHON_AVAILABLE
        self._device_id = 0
        self._handle = sail.Handle(self._device_id) if self._use_sophon else None
        self._bmcv = sail.Bmcv(self._handle) if self._use_sophon and self._handle is not None else None
        self._encoder = None
        self._bmimg_nv12 = None
        self._segment_index = 0
        self._segment_start_time = None
        self._segment_frames = 0
        self._segments = []  # (filename, duration)
        self._m3u8_path = self.output_dir / "live.m3u8"

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _reset_hls_output(self):
        """清理旧分片与 playlist，避免播放端回跳到历史画面。"""
        try:
            # 删除旧分片
            for p in self.output_dir.glob("segment_*.ts"):
                try:
                    p.unlink()
                except Exception:
                    pass
            # 删除旧 playlist（与可能残留的临时文件）
            for name in ("live.m3u8", "live.m3u8.tmp"):
                p = self.output_dir / name
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        # 重置内部状态
        self._segment_index = 0
        self._segment_start_time = None
        self._segment_frames = 0
        self._segments = []

    # ==================== ffmpeg 回退实现（保留） ====================
    def _start_ffmpeg(self, width, height):
        """启动 ffmpeg：stdin 为 rawvideo bgr24，输出 HLS（仅在未启用 Sophon 时使用）"""
        self._ensure_output_dir()
        seg_pattern = str(self.output_dir / 'segment_%03d.ts')
        m3u8_path = str(self._m3u8_path)
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
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("HLS encoder (ffmpeg) started: %s, %dx%d @ %dfps", m3u8_path, width, height, self.fps)
            return True
        except FileNotFoundError:
            logger.error("ffmpeg not found; install ffmpeg for HLS encoding")
            return False
        except Exception as e:
            logger.error("Failed to start ffmpeg: %s", e)
            return False

    def _stop_ffmpeg(self):
        """停止并清理当前 ffmpeg 进程（若存在）。"""
        if self._proc is None:
            return
        try:
            if getattr(self._proc, "stdin", None) is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    # ==================== Sophon 硬件编码 + HLS 分片 ====================
    def _start_sophon_segment(self, width, height):
        """开启一个新的 TS 片段编码（使用 Sophon 硬件编码器）。"""
        if not self._use_sophon or self._bmcv is None or self._handle is None:
            return False
        self._ensure_output_dir()
        self._segment_index += 1
        self._segment_frames = 0
        self._segment_start_time = time.time()
        seg_name = f"segment_{self._segment_index:03d}.ts"
        seg_path = self.output_dir / seg_name
        enc_params = (
            f"width={width}:height={height}:bitrate=2000:gop={self._keyint}:"
            f"gop_preset=2:framerate={self.fps}"
        )
        try:
            # 这里使用 NV12 作为编码输入格式，保持与 yolov8_bmcv 示例一致
            self._encoder = sail.Encoder(str(seg_path), self._device_id, 'h264_bm', 'NV12', enc_params, 10)
            if not self._encoder.is_opened():
                logger.error("Failed to open Sophon encoder for segment: %s", seg_path)
                self._encoder = None
                return False
            # 分配一个复用的 NV12 BMImage 作为编码输入缓冲
            self._bmimg_nv12 = sail.BMImage(
                self._handle, height, width,
                sail.Format.FORMAT_NV12, sail.DATA_TYPE_EXT_1N_BYTE
            )
            logger.info("HLS encoder (Sophon) started segment: %s, %dx%d @ %dfps", seg_path, width, height, self.fps)
            return True
        except Exception as e:
            logger.error("Failed to start Sophon encoder: %s", e)
            self._encoder = None
            self._bmimg_nv12 = None
            return False

    def _close_sophon_segment(self):
        """关闭当前 TS 片段并更新 m3u8。"""
        if not self._use_sophon:
            return
        if self._encoder is None:
            return
        try:
            self._encoder.release()
        except Exception:
            pass
        self._encoder = None
        # 计算本片段时长
        if self._segment_start_time is None or self._segment_frames <= 0:
            duration = float(self.hls_time)
        else:
            elapsed = time.time() - self._segment_start_time
            duration = max(0.5, elapsed)
        seg_name = f"segment_{self._segment_index:03d}.ts"
        self._segments.append((seg_name, duration))
        # 限制 m3u8 里片段数量
        if len(self._segments) > self.hls_list_size:
            # 删除过旧 ts 文件
            old_seg, _ = self._segments[0]
            old_path = self.output_dir / old_seg
            try:
                if old_path.exists():
                    old_path.unlink()
            except Exception:
                pass
            self._segments = self._segments[-self.hls_list_size:]
        self._write_m3u8()

    def _write_m3u8(self):
        """根据当前片段列表写出简单 HLS 播放列表。"""
        if not self._segments:
            return
        target_duration = max(int(self.hls_time), 1)
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:" + str(max(0, self._segment_index - len(self._segments) + 1)),
        ]
        for name, dur in self._segments:
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(name)
        tmp_path = str(self._m3u8_path) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp_path, self._m3u8_path)
        except Exception as e:
            logger.error("Failed to write m3u8: %s", e)

    def _check_control(self):
        try:
            while not self.control_queue.empty():
                cmd = self.control_queue.get_nowait()
                if cmd == 'stop':
                    self.running = False
        except Exception:
            pass

    def run(self):
        """主循环：从队列取帧，编码为 HLS 流（优先 Sophon 硬件编码）。"""
        logger.info("Encoder service starting (HLS)...")
        if self._use_sophon:
            logger.info("Sophon SAIL detected, using hardware encoder for HLS (fallback to ffmpeg if needed)")
        else:
            logger.info("Sophon SAIL not available, using ffmpeg for HLS")
        self.running = True
        frame_count = 0
        last_log = 0
        last_restart_time = 0.0

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
            # 尺寸变化时需要重建编码器 / ffmpeg 管道
            if self._width != w or self._height != h:
                # 关闭旧编码器/管道
                if self._use_sophon:
                    self._close_sophon_segment()
                else:
                    self._stop_ffmpeg()
                self._width, self._height = w, h
                # 尝试启动 Sophon 编码；失败时自动切换到 ffmpeg
                if self._use_sophon:
                    if not self._start_sophon_segment(w, h):
                        logger.warning("Fall back to ffmpeg HLS encoder because Sophon encoder failed")
                        self._use_sophon = False
                        if not self._start_ffmpeg(w, h):
                            continue
                else:
                    if not self._start_ffmpeg(w, h):
                        continue
            else:
                # 尺寸未变，根据当前模式检查底层是否健康
                if self._use_sophon:
                    if self._encoder is None:
                        # Sophon 编码器异常关闭，尝试重启一次；失败则切回 ffmpeg
                        if not self._start_sophon_segment(w, h):
                            logger.warning("Sophon encoder restart failed, switching to ffmpeg")
                            self._use_sophon = False
                            if not self._start_ffmpeg(w, h):
                                continue
                else:
                    if self._proc is None:
                        now = time.time()
                        if now - last_restart_time < 0.8:
                            continue
                        last_restart_time = now
                        if not self._start_ffmpeg(w, h):
                            continue

            try:
                if self._use_sophon:
                    # 使用 Sophon 硬件编码：frame(BGR) -> NV12 BMImage -> Encoder.video_write
                    if self._encoder is None or self._bmimg_nv12 is None:
                        raise RuntimeError("Sophon encoder not ready")
                    # 先将 numpy BGR 图像转为设备侧 BGR BMImage，再转换为 NV12
                    bmimg_bgr = sail.BMImage(
                        self._handle, h, w,
                        sail.Format.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE
                    )
                    self._bmcv.mat_to_bm_image(frame, bmimg_bgr)
                    self._bmcv.convert_format(bmimg_bgr, self._bmimg_nv12)
                    self._encoder.video_write(self._bmimg_nv12)
                    self._segment_frames += 1
                    frame_count += 1
                    # 判断是否需要切片滚动
                    elapsed = time.time() - (self._segment_start_time or time.time())
                    if elapsed >= self.hls_time and self._segment_frames > 0:
                        self._close_sophon_segment()
                        # 开启下一片段
                        self._start_sophon_segment(w, h)
                else:
                    if self._proc is None or getattr(self._proc, "stdin", None) is None:
                        raise BrokenPipeError("ffmpeg process not ready")
                    self._proc.stdin.write(frame.tobytes())
                    frame_count += 1

                if frame_count - last_log >= 100:
                    logger.info("HLS encoder: %d frames written", frame_count)
                    last_log = frame_count
            except BrokenPipeError:
                logger.warning("ffmpeg pipe broken, restarting...")
                self._stop_ffmpeg()
            except Exception as e:
                logger.warning("Encoder write error: %s", e)
                if self._use_sophon:
                    # 尝试关闭当前片段并重启下一个；失败则切换到 ffmpeg
                    self._close_sophon_segment()
                    if not self._start_sophon_segment(w, h):
                        logger.warning("Sophon encoder restart failed, switching to ffmpeg")
                        self._use_sophon = False
                        self._stop_ffmpeg()
                else:
                    self._stop_ffmpeg()

        # 退出循环时，清理资源
        if self._use_sophon:
            self._close_sophon_segment()
        else:
            self._stop_ffmpeg()
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
