from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

from book2video.providers.base import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult, VideoProvider
from book2video.providers.provider_utils import render_template, video_template_variables
from book2video.providers.lifecycle import run_lifecycle_command


class CommandVideoProvider(VideoProvider):
    """Generic local adapter for native Wan/LightX2V/SkyReels scripts or an MCP-facing wrapper executable."""

    def __init__(
        self,
        *,
        command: list[str] | str,
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 1.0,
        supports_multi_gpu_job: bool = True,
        supports_video_continuation: bool = True,
        outputs_audio: bool = False,
        provider_name: str = "command-video",
        timeout_sec: float = 3600.0,
        env: dict[str, str] | None = None,
        write_request_json: bool = True,
        estimated_vram_gb: float | None = None,
        prepare_command: list[str] | str | None = None,
        release_command: list[str] | str | None = None,
        lifecycle_timeout_sec: float = 300.0,
    ) -> None:
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout_sec = timeout_sec
        self.env = env or {}
        self.write_request_json = write_request_json
        self.prepare_command = prepare_command
        self.release_command = release_command
        self.lifecycle_timeout_sec = lifecycle_timeout_sec
        self._provider_id = provider_name
        self._capabilities = VideoCapabilities(
            max_duration_sec=max_duration_sec,
            min_duration_sec=min_duration_sec,
            supports_image_refs=True,
            supports_video_continuation=supports_video_continuation,
            supports_audio_refs=True,
            supports_multi_gpu_job=supports_multi_gpu_job,
            outputs_audio=outputs_audio,
            estimated_vram_gb=estimated_vram_gb,
        )


    async def prepare(self, device_groups: list[tuple[str, ...]]) -> None:
        env = {"BOOK2VIDEO_DEVICE_GROUPS": json.dumps([list(x) for x in device_groups])}
        await run_lifecycle_command(
            self.prepare_command, timeout_sec=self.lifecycle_timeout_sec, env=env
        )

    async def release(self) -> None:
        await run_lifecycle_command(self.release_command, timeout_sec=self.lifecycle_timeout_sec)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        started = time.perf_counter()
        output = Path(request.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        request_json = output.with_suffix(".request.json")
        payload = request.model_dump(mode="json") | {"devices": list(devices)}
        if self.write_request_json:
            request_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        variables = video_template_variables(request, devices, include_data_uris=False)
        variables["REQUEST_JSON"] = str(request_json)
        rendered = [str(render_template(piece, variables)) for piece in self.command]

        proc_env = os.environ.copy()
        proc_env.update(self.env)
        # Exposing the requested cards is attempted as a convenience for scripts that honor CUDA_VISIBLE_DEVICES.
        # A model-specific wrapper remains free to interpret the explicit device list differently.
        cuda_ids = [d.split(":", 1)[1] for d in devices if d.startswith("cuda:") and ":" in d]
        if cuda_ids:
            proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_ids)
        proc_env["BOOK2VIDEO_DEVICES"] = ",".join(devices)

        proc = await asyncio.create_subprocess_exec(*rendered, env=proc_env)
        try:
            code = await asyncio.wait_for(proc.wait(), timeout=self.timeout_sec)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Video command timed out for {request.clip_id}")
        if code != 0:
            raise RuntimeError(f"Video command exited with code {code}: {rendered!r}")
        if not output.exists():
            raise RuntimeError(f"Video command completed but did not create {output}")

        return VideoGenerationResult(
            output_path=str(output),
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"command": rendered, "devices": list(devices), "request_json": str(request_json)},
        )
