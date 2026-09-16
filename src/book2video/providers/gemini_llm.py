from __future__ import annotations

import os
from typing import Any, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest
from book2video.providers.provider_utils import (
    build_structured_prompt,
    file_to_base64,
    materialize_attachment_images,
    mime_type,
    parse_json_text,
    temporary_directory,
)

T = TypeVar("T", bound=BaseModel)


class GeminiLLMProvider(LLMProvider):
    """Gemini generateContent adapter with JSON-schema output and sampled video frames for judging."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_sec: float = 600.0,
        temperature: float = 0.0,
        video_frame_samples: int = 6,
        max_attachment_width: int = 960,
        extra_generation_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.video_frame_samples = video_frame_samples
        self.max_attachment_width = max_attachment_width
        self.extra_generation_config = extra_generation_config or {}

    @property
    def provider_id(self) -> str:
        return "gemini"

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key; set GEMINI_API_KEY or configure api_key.")
        prompt = build_structured_prompt(request, response_model)

        with temporary_directory("book2video_gemini_") as tmp:
            visuals = materialize_attachment_images(
                request,
                tmp,
                video_frame_samples=self.video_frame_samples,
                max_width=self.max_attachment_width,
            )
            parts: list[dict[str, Any]] = [{"text": prompt}]
            for label, image_path in visuals:
                parts.append({"text": f"Visual attachment: {label}"})
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type(image_path),
                            "data": file_to_base64(image_path),
                        }
                    }
                )

            generation_config: dict[str, Any] = {
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": response_model.model_json_schema(),
            }
            generation_config.update(self.extra_generation_config)
            body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
            url = f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"
            headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

        text_parts: list[str] = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
        if not text_parts:
            raise RuntimeError(f"Gemini response contained no text: {data}")
        return response_model.model_validate(parse_json_text("".join(text_parts)))
