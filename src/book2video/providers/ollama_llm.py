from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest
from book2video.providers.provider_utils import (
    build_structured_prompt,
    file_to_base64,
    materialize_attachment_images,
    parse_json_text,
    temporary_directory,
)

T = TypeVar("T", bound=BaseModel)


class OllamaLLMProvider(LLMProvider):
    """Local Ollama adapter; use a vision-capable model if the video-judge stage should inspect frames."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_sec: float = 600.0,
        temperature: float = 0.0,
        video_frame_samples: int = 6,
        max_attachment_width: int = 960,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "30m",
        unload_on_release: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.video_frame_samples = video_frame_samples
        self.max_attachment_width = max_attachment_width
        self.options = options or {}
        self.keep_alive = keep_alive
        self.unload_on_release = unload_on_release

    async def prepare(self) -> None:
        # Ollama loads lazily on the first real request. Keeping the hook lightweight avoids an
        # extra inference call while still giving the phase manager a consistent lifecycle API.
        return None

    async def release(self) -> None:
        if not self.unload_on_release:
            return
        body = {"model": self.model, "keep_alive": 0}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=body)
                response.raise_for_status()
        except Exception:
            # Release is best-effort here: an older Ollama build or a stopped server should not
            # invalidate hours of completed planning work.
            return

    @property
    def provider_id(self) -> str:
        return "ollama"

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        prompt = build_structured_prompt(request, response_model)
        with temporary_directory("book2video_ollama_") as tmp:
            visuals = materialize_attachment_images(
                request,
                tmp,
                video_frame_samples=self.video_frame_samples,
                max_width=self.max_attachment_width,
            )
            message: dict[str, Any] = {"role": "user", "content": prompt}
            if visuals:
                message["content"] += "\n\nVisual inputs follow in the same order as these labels:\n" + "\n".join(
                    f"- {label}" for label, _ in visuals
                )
                message["images"] = [file_to_base64(path) for _, path in visuals]

            body = {
                "model": self.model,
                "messages": [message],
                "stream": False,
                "format": response_model.model_json_schema(),
                "options": {"temperature": self.temperature, **self.options},
                "keep_alive": self.keep_alive,
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                data = response.json()

        return response_model.model_validate(parse_json_text(data["message"]["content"]))
