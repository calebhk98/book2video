from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def extract_last_frame(video_path: str | Path, output_path: str | Path) -> str | None:
    if not ffmpeg_available():
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.08",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output) if result.returncode == 0 and output.exists() else None


def has_audio(video_path: str | Path) -> bool:
    if not ffmpeg_available():
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())
