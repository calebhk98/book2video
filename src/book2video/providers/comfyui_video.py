from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

import httpx

from book2video.providers.base import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult, VideoProvider
from book2video.providers.provider_utils import render_template
from book2video.providers.lifecycle import run_lifecycle_command


class ComfyUIVideoProvider(VideoProvider):
    """Generic ComfyUI API-workflow adapter with optional one-server-per-GPU routing."""

    def __init__(
        self,
        *,
        workflow_path: str,
        server_url: str = "http://127.0.0.1:8188",
        server_by_device: dict[str, str] | None = None,
        max_duration_sec: float = 10.0,
        estimated_vram_gb: float | None = None,
        min_duration_sec: float = 1.0,
        poll_interval_sec: float = 2.0,
        timeout_sec: float = 3600.0,
        provider_name: str = "comfyui",
        output_extensions: list[str] | None = None,
        prepare_command: list[str] | str | None = None,
        release_command: list[str] | str | None = None,
        lifecycle_timeout_sec: float = 300.0,
    ) -> None:
        self.workflow_path = Path(workflow_path)
        self.server_url = server_url.rstrip("/")
        self.server_by_device = {k: v.rstrip("/") for k, v in (server_by_device or {}).items()}
        self.poll_interval_sec = poll_interval_sec
        self.timeout_sec = timeout_sec
        self._provider_id = provider_name
        self.output_extensions = tuple(output_extensions or [".mp4", ".webm", ".mov", ".mkv", ".gif"])
        self.prepare_command = prepare_command
        self.release_command = release_command
        self.lifecycle_timeout_sec = lifecycle_timeout_sec
        self._capabilities = VideoCapabilities(
            max_duration_sec=max_duration_sec,
            estimated_vram_gb=estimated_vram_gb,
            min_duration_sec=min_duration_sec,
            supports_image_refs=True,
            supports_video_continuation=True,
            supports_audio_refs=False,
            supports_multi_gpu_job=False,
            outputs_audio=False,
        )

    async def prepare(self, device_groups: list[tuple[str, ...]]) -> None:
        await run_lifecycle_command(self.prepare_command, timeout_sec=self.lifecycle_timeout_sec)

    async def release(self) -> None:
        await run_lifecycle_command(self.release_command, timeout_sec=self.lifecycle_timeout_sec)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    def _server_for(self, devices: tuple[str, ...]) -> str:
        if devices and devices[0] in self.server_by_device:
            return self.server_by_device[devices[0]]
        return self.server_url

    async def _upload_image(self, client: httpx.AsyncClient, server: str, path: str) -> str:
        source = Path(path)
        with source.open("rb") as f:
            response = await client.post(
                f"{server}/upload/image",
                files={"image": (source.name, f, "application/octet-stream")},
                data={"type": "input", "overwrite": "true"},
            )
        response.raise_for_status()
        data = response.json()
        sub = data.get("subfolder") or ""
        return f"{sub}/{data['name']}".strip("/")

    def _find_output_descriptor(self, history: Any) -> dict[str, Any] | None:
        if isinstance(history, dict):
            if "filename" in history and str(history["filename"]).lower().endswith(self.output_extensions):
                return history
            for value in history.values():
                found = self._find_output_descriptor(value)
                if found:
                    return found
        elif isinstance(history, list):
            for value in history:
                found = self._find_output_descriptor(value)
                if found:
                    return found
        return None

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        started = time.perf_counter()
        server = self._server_for(devices)
        workflow = copy.deepcopy(json.loads(self.workflow_path.read_text(encoding="utf-8")))
        output = Path(request.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
            char_names = [await self._upload_image(client, server, p) for p in request.character_reference_images]
            loc_names = [await self._upload_image(client, server, p) for p in request.location_reference_images]
            continuation_frame = None
            if request.continuation_frame_path:
                continuation_frame = await self._upload_image(client, server, request.continuation_frame_path)

            variables: dict[str, Any] = {
                "PROMPT": request.intent.visual_prompt,
                "MOTION_PROMPT": request.intent.motion_prompt,
                "NEGATIVE_PROMPT": request.intent.negative_prompt,
                "SEED": request.seed,
                "DURATION": request.duration_sec,
                "OUTPUT_PREFIX": f"book2video_{request.clip_id}_t{request.take_number}",
                "CHARACTER_REFERENCES": char_names,
                "LOCATION_REFERENCES": loc_names,
                "CONTINUATION_FRAME": continuation_frame,
                "CONTINUATION_VIDEO_PATH": request.continuation_video_path,
            }
            for i, name in enumerate(char_names):
                variables[f"CHAR_REF_{i}"] = name
            for i, name in enumerate(loc_names):
                variables[f"LOC_REF_{i}"] = name
            workflow = render_template(workflow, variables)

            queued = await client.post(f"{server}/prompt", json={"prompt": workflow})
            queued.raise_for_status()
            queued_data = queued.json()
            if queued_data.get("error"):
                raise RuntimeError(f"ComfyUI rejected workflow: {queued_data}")
            prompt_id = queued_data["prompt_id"]

            deadline = time.monotonic() + self.timeout_sec
            history_data: Any = None
            while time.monotonic() < deadline:
                history = await client.get(f"{server}/history/{prompt_id}")
                history.raise_for_status()
                data = history.json()
                if prompt_id in data:
                    history_data = data[prompt_id]
                    break
                await asyncio.sleep(self.poll_interval_sec)
            if history_data is None:
                raise RuntimeError(f"ComfyUI timed out waiting for {prompt_id}")

            descriptor = self._find_output_descriptor(history_data.get("outputs", history_data))
            if not descriptor:
                raise RuntimeError("ComfyUI completed but no video-like output descriptor was found in history.")
            params = {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
            media = await client.get(f"{server}/view", params=params)
            media.raise_for_status()
            output.write_bytes(media.content)

        return VideoGenerationResult(
            output_path=str(output),
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"server": server, "prompt_id": prompt_id, "devices": list(devices)},
        )
