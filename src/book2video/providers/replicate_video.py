from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx

from book2video.providers.base import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult, VideoProvider
from book2video.providers.provider_utils import (
    download_file,
    get_by_path,
    recursive_find_first_url,
    render_template,
    video_template_variables,
)


class ReplicateVideoProvider(VideoProvider):
    """Generic Replicate prediction adapter; model-specific input keys live in input_template."""

    def __init__(
        self,
        *,
        model: str | None = None,
        version: str | None = None,
        api_token: str | None = None,
        api_token_env: str = "REPLICATE_API_TOKEN",
        input_template: dict[str, Any] | None = None,
        output_path: str | None = "output",
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 1.0,
        poll_interval_sec: float = 2.0,
        timeout_sec: float = 3600.0,
    ) -> None:
        if not model and not version:
            raise ValueError("Replicate provider needs model=owner/name or version=...")
        self.model = model
        self.version = version
        self.api_token = api_token or os.getenv(api_token_env, "")
        self.input_template = input_template or {
            "prompt": "{{PROMPT}}",
            "duration": "{{DURATION}}",
            "seed": "{{SEED}}",
            "reference_images": "{{REFERENCE_DATA_URIS}}",
        }
        self.output_path = output_path
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
        return "replicate"

    @property
    def capabilities(self) -> VideoCapabilities:
        return self._capabilities

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        if not self.api_token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN or configured api_token.")
        started = time.perf_counter()
        variables = video_template_variables(request, devices)
        payload_input = render_template(self.input_template, variables)
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        if self.model and ":" not in self.model and not self.version:
            owner, name = self.model.split("/", 1)
            url = f"https://api.replicate.com/v1/models/{owner}/{name}/predictions"
            body = {"input": payload_input}
        else:
            version = self.version or self.model
            url = "https://api.replicate.com/v1/predictions"
            body = {"version": version, "input": payload_input}

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            prediction = response.json()
            get_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
            deadline = time.monotonic() + self.timeout_sec
            while prediction.get("status") not in {"succeeded", "failed", "canceled"}:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Replicate timed out for prediction {prediction.get('id')}")
                await asyncio.sleep(self.poll_interval_sec)
                polled = await client.get(get_url, headers=headers)
                polled.raise_for_status()
                prediction = polled.json()
        if prediction.get("status") != "succeeded":
            raise RuntimeError(f"Replicate prediction failed: {prediction.get('error') or prediction}")
        candidate = get_by_path(prediction, self.output_path) if self.output_path else prediction
        media_url = recursive_find_first_url(candidate)
        if not media_url:
            raise RuntimeError(f"Could not find a media URL in Replicate output: {candidate!r}")
        await download_file(media_url, request.output_path, timeout=self.timeout_sec)
        return VideoGenerationResult(
            output_path=request.output_path,
            seed=request.seed,
            generation_seconds=time.perf_counter() - started,
            metadata={"prediction_id": prediction.get("id"), "metrics": prediction.get("metrics")},
        )
