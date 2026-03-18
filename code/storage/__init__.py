"""
存储层（Storage Layer）：告警图片/视频持久化与历史数据回溯。
实际保存逻辑在 core.display_service 的 VideoBufferManager 中（异步写入 alarm_videos 目录），
本包提供历史列表接口与路径约定，供 Web API 使用。
告警列表的显示时间以文件的修改时间（系统时间）为准，避免摄像头时间错误导致展示时间不准确；
同时保留从文件名解析的时间戳作为附加字段，便于后续需要时使用。
"""

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 北京时间，与 display_service 生成文件名时一致
BEIJING_TZ = timezone(timedelta(hours=8))


def get_alarm_output_dir(config_output=None):
    """从配置或默认值获取告警输出目录。"""
    if config_output and isinstance(config_output, dict):
        return config_output.get("video_output_dir", "./alarm_videos")
    return "./alarm_videos"


# 告警类型前缀（用于从文件名解析类别；长前缀需在前以优先匹配）
ALARM_CATEGORY_PREFIXES = [
    "window_door_inside",
    "window_door_outside",
    "window_door",
    "fall",
    "ventilator",
    "fight",
    "crowd",
    "helmet",
]

# 文件名中的时间格式：{type}_YYYYMMDD_HHMMSS 或 {type}_YYYYMMDD_HHMMSS_frame
_FILENAME_TIME_PATTERN = re.compile(r"_(\d{8})_(\d{6})(?:_frame)?$", re.IGNORECASE)


def _parse_alarm_time_from_filename(name):
    """
    从告警文件名解析告警发生时间（与录像首帧/摄像头左上角时间一致）。
    例如 helmet_20251219_014312.mp4 或 helmet_20251219_014312_frame.jpg -> 2025-12-19 01:43:12 北京时。
    成功返回 Unix 时间戳，失败返回 None。
    """
    base = Path(name).stem
    m = _FILENAME_TIME_PATTERN.search(base)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
        h, mi, s = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
        dt = datetime(y, mo, d, h, mi, s, tzinfo=BEIJING_TZ)
        return dt.timestamp()
    except (ValueError, IndexError):
        return None


def _category_from_filename(name):
    """从文件名解析告警类别，如 helmet_20251218_023317.mp4 -> helmet。"""
    base = name.split(".")[0]
    for prefix in ALARM_CATEGORY_PREFIXES:
        if base == prefix or base.startswith(prefix + "_"):
            return prefix
    return "other"


def list_alarm_files(output_dir, limit=100):
    """
    列出告警目录下的视频/图片文件，用于历史回溯 API。
    返回按告警时间倒序的列表，每项为 {path, name, mtime, type, category, name_time}。
    - mtime：文件修改时间（系统时间，供前端展示与排序使用）
    - name_time：从文件名解析得到的时间戳（如 helmet_20251219_014312），解析失败为 None
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    items = []
    for f in output_path.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf in (".mp4", ".avi", ".mkv"):
            item_type = "video"
        elif suf in (".jpg", ".jpeg", ".png"):
            item_type = "image"
        else:
            continue
        # 显示时间：以文件修改时间为准（系统时间），避免摄像头时间错误
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0

        # 额外保留从文件名解析的时间戳，供需要精确对齐摄像头时使用
        name_time = _parse_alarm_time_from_filename(f.name)
        # 尝试从文件名解析通道号：例如 helmet_ch2_20251219_014312.mp4 -> channel = 2
        channel = None
        base = f.stem
        try:
            import re
            m_ch = re.search(r"_ch(\d+)_", base, re.IGNORECASE)
            if m_ch:
                channel = int(m_ch.group(1))
        except Exception:
            channel = None
        category = _category_from_filename(f.name)
        items.append({
            "path": str(f),
            "name": f.name,
            "mtime": mtime,
            "type": item_type,
            "category": category,
            "name_time": name_time,
            "channel": channel,
            "size": f.stat().st_size if f.exists() else 0,
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]


# 支持的告警文件扩展名（用于清理/清除时只删此类文件）
_ALARM_EXTENSIONS = (".mp4", ".avi", ".mkv", ".jpg", ".jpeg", ".png")


def _is_alarm_file(path):
    return path.suffix.lower() in _ALARM_EXTENSIONS


def cleanup_old_alarm_files(output_dir, max_age_days=7):
    """
    删除告警目录中超过 max_age_days 天的视频和图片（按文件修改时间判断）。
    返回删除的文件数量。
    """
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    deleted = 0
    for f in output_path.iterdir():
        if not f.is_file() or not _is_alarm_file(f):
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError:
            pass
    return deleted


def clear_all_alarm_files(output_dir):
    """
    删除告警目录下所有告警视频和图片文件。
    返回删除的文件数量。
    """
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        return 0
    deleted = 0
    for f in output_path.iterdir():
        if not f.is_file() or not _is_alarm_file(f):
            continue
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


__all__ = ["get_alarm_output_dir", "list_alarm_files", "cleanup_old_alarm_files", "clear_all_alarm_files"]
