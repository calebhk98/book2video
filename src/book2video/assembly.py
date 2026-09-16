from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from book2video.media import ffmpeg_available, has_audio
from book2video.schemas import SelectedClip


class FFmpegAssembler:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 24):
        self.width = width
        self.height = height
        self.fps = fps

    def _normalize(self, source: Path, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not ffmpeg_available() or not source.exists():
            return False

        # Normalization is attempted here so providers with slightly different codecs,
        # frame rates, or missing audio tracks can still be concatenated predictably.
        vf = (
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,fps={self.fps}"
        )
        if has_audio(source):
            cmd = [
                "ffmpeg", "-y", "-i", str(source),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                str(dest),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(source),
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-shortest",
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                str(dest),
            ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0 and dest.exists()

    def assemble(self, clips: list[SelectedClip], run_dir: str | Path) -> str | None:
        run_dir = Path(run_dir)
        normalized = run_dir / "normalized"
        selected_paths: list[Path] = []
        for i, clip in enumerate(clips):
            source = Path(clip.selected_path)
            dest = normalized / f"{i:04d}_{clip.clip_id}.mp4"
            if self._normalize(source, dest):
                selected_paths.append(dest)

        if not selected_paths:
            return None

        concat_file = run_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in selected_paths),
            encoding="utf-8",
        )
        output = run_dir / "rough_cut.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(output),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(output) if result.returncode == 0 and output.exists() else None
