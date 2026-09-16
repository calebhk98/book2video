from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from book2video.providers.base import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult, VideoProvider
from book2video.providers.provider_utils import (
    download_file,
    file_to_data_uri,
    recursive_find_first_url,
    render_template,
    video_template_variables,
)


class RunwayVideoProvider(VideoProvider):
    """Runway Model Router adapter with an optional payload template for model-specific inputs."""

    def __init__(
        self,
        *,
        config_id: str = "preview-fast",
        api_secret: str | None = None,
        api_secret_env: str = "RUNWAYML_API_SECRET",
        base_url: str = "https://api.dev.runwayml.com/v1",
        api_version: str = "2024-11-06",
        aspect_ratio: str = "16:9",
        reference_role: str = "first",
        payload_template: dict[str, Any] | None = None,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 1.0,
        poll_interval_sec: float = 2.0,
        timeout_sec: float = 3600.0,
    ) -> None:
        self.config_id = config_id
        self.api_secret = api_secret or os.getenv(api_secret_env, "")
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.aspect_ratio = aspect_ratio
        self.reference_role = reference_role
        self.payload_template = payload_template
        self.poll_interval_sec = poll_interval_sec
        self.timeout_sec = timeout_sec
        self._capabilities = VideoCapabilities(
            max_duration_sec=max_duration_sec,
            min_duration_sec=min_duration_sec,
            supports_image_refs=True,
            supports_video_continuation=True,
            supports_audio_refs=True,
            outputs_audio=True,
        )

    @property
    def provider_id(self) -> str:
        return "runway"

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        if not self.api_secret:
            raise RuntimeError("Missing RUNWAYML_API_SECRET or configured api_secret.")
        started = time.perf_counter()
        variables = video_template_variables(request, devices, include_data_uris=bool(self.payload_template))
        if self.payload_template:
            body = render_template(self.payload_template, variables)
        else:
            refs = []
            # The router's documented generic example is first-frame oriented. A single first image is
            # attempted here as a safe baseline; multi-character reference semantics are left to payload_template.
            first_path = request.continuation_frame_path
            if not first_path:
                candidates = request.character_reference_images + request.location_reference_images
                first_path = candidates[0] if candidates else None
            if first_path:
                refs.append({"uri": file_to_data_uri(first_path), "role": self.reference_role})
            body = {
                "configId": self.config_id,
                "input": {
                    "referenceImages": refs,
                    "promptText": " ".join(
                        x for x in [request.intent.visual_prompt, request.intent.motion_prompt] if x
                    ),
                    "aspectRatio": self.aspect_ratio,
                    "duration": request.duration_sec,
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_secret}",
            "X-Runway-Version": self.api_version,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
            response = await client.post(f"{self.base_url}/generate/video", headers=headers, json=body)
            response.raise_for_status()
            task = response.json()
            task_id = task.get("id") or task.get("taskId")
            if not task_id:
                # Some router responses may already include an output; this keeps the adapter usable if that shape changes.
                media_url = recursive_find_first_url(task.get("output"))
                if not media_url:
                    raise RuntimeError(f"Runway response had no task id or output: {task}")
                await download_file(media_url, request.output_path, timeout=self.timeout_sec)
                return VideoGenerationResult(output_path=request.output_path, seed=request.seed, metadata={"routing": task.get("routing")})

            deadline = time.monotonic() + self.timeout_sec
            while True:
                status_response = await client.get(f"{self.base_url}/tasks/{task_id}", headers=headers)
                status_response.raise_for_status()
                task = status_response.json()
                status = str(task.get("status", "")).upper()
                if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                    break
                if status in {"FAILED", "CANCELED", "CANCELLED"}:
                    raise RuntimeError(f"Runway task failed: {task}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Runway task timed out: {task_id}")
                await asyncio.sleep(self.poll_interval_sec)
        media_url = recursive_find_first_url(task.get("output"))
        if not media_url:
            raise RuntimeError(f"Runway task completed without a media URL: {task}")
        await download_file(media_url, request.output_path, timeout=self.timeout_sec)
        return VideoGenerationResult(
            output_path=request.output_path,
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"task_id": task_id, "routing": task.get("routing")},
        )
