from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

from book2video.media import ffmpeg_available
from book2video.providers.base import (
    VideoCapabilities,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProvider,
)


class MockVideoProvider(VideoProvider):
    """Creates black MP4s with silent audio when ffmpeg exists; useful for pipeline tests."""

    def __init__(self, max_duration_sec: float = 10.0,
        estimated_vram_gb: float | None = None, simulated_delay_sec: float = 0.0) -> None:
        self._capabilities = VideoCapabilities(
            max_duration_sec=max_duration_sec,
            estimated_vram_gb=estimated_vram_gb,
            min_duration_sec=0.5,
            supports_image_refs=True,
            supports_video_continuation=True,
            outputs_audio=True,
        )
        self.simulated_delay_sec = simulated_delay_sec

    @property
    def provider_id(self) -> str:
        return "mock-video"

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        started = time.perf_counter()
        if self.simulated_delay_sec:
            await asyncio.sleep(self.simulated_delay_sec)
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if ffmpeg_available():
            duration = max(0.5, request.duration_sec)
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=640x360:r=24:d={duration}",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(out),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            code = await proc.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg mock generation failed for {request.clip_id}")
        else:
            out.write_bytes(b"")

        return VideoGenerationResult(
            output_path=str(out),
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"devices": list(devices), "mock": True},
        )
