from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from urllib.parse import quote

import httpx

from book2video.providers.base import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult, VideoProvider
from book2video.providers.provider_utils import (
    download_file,
    get_by_path,
    recursive_find_first_url,
    render_template,
    video_template_variables,
)


class FalVideoProvider(VideoProvider):
    """Generic fal queue adapter; the argument template can track model-specific field names without core changes."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "FAL_KEY",
        arguments_template: dict[str, Any] | None = None,
        output_path: str | None = None,
        queue_base_url: str = "https://queue.fal.run",
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 1.0,
        poll_interval_sec: float = 2.0,
        timeout_sec: float = 3600.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.arguments_template = arguments_template or {
            "prompt": "{{PROMPT}}",
            "duration": "{{DURATION}}",
            "seed": "{{SEED}}",
            "image_urls": "{{REFERENCE_DATA_URIS}}",
        }
        self.output_path = output_path
        self.queue_base_url = queue_base_url.rstrip("/")
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
        return "fal"

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        if not self.api_key:
            raise RuntimeError("Missing FAL_KEY or configured api_key.")
        started = time.perf_counter()
        variables = video_template_variables(request, devices)
        arguments = render_template(self.arguments_template, variables)
        headers = {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}
        submit_url = f"{self.queue_base_url}/{self.model}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
            response = await client.post(submit_url, headers=headers, json=arguments)
            response.raise_for_status()
            job = response.json()
            status_url = job.get("status_url") or job.get("statusUrl")
            response_url = job.get("response_url") or job.get("responseUrl")
            if not status_url and job.get("request_id"):
                status_url = f"{submit_url}/requests/{quote(job['request_id'], safe='')}/status"
            deadline = time.monotonic() + self.timeout_sec
            while status_url:
                status = await client.get(status_url, headers=headers)
                status.raise_for_status()
                state = status.json()
                state_name = str(state.get("status", "")).upper()
                if state_name in {"COMPLETED", "SUCCEEDED"}:
                    response_url = response_url or state.get("response_url") or state.get("responseUrl")
                    break
                if state_name in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                    raise RuntimeError(f"fal generation failed: {state}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"fal generation timed out: {job.get('request_id')}")
                await asyncio.sleep(self.poll_interval_sec)
            if response_url:
                result_response = await client.get(response_url, headers=headers)
                result_response.raise_for_status()
                result = result_response.json()
            else:
                result = job
        candidate = get_by_path(result, self.output_path) if self.output_path else result
        media_url = recursive_find_first_url(candidate)
        if not media_url:
            raise RuntimeError(f"Could not find a media URL in fal output: {candidate!r}")
        await download_file(media_url, request.output_path, timeout=self.timeout_sec)
        return VideoGenerationResult(
            output_path=request.output_path,
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"request_id": job.get("request_id")},
        )
