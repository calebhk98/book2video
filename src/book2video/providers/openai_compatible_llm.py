from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest
from book2video.providers.lifecycle import run_lifecycle_command
from book2video.providers.provider_utils import (
    build_structured_prompt,
    file_to_data_uri,
    materialize_attachment_images,
    parse_json_text,
    temporary_directory,
)

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLMProvider(LLMProvider):
    """Adapter for vLLM, LM Studio, and other Chat Completions-compatible servers."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_COMPATIBLE_API_KEY",
        provider_name: str = "openai-compatible",
        timeout_sec: float = 600.0,
        temperature: float = 0.0,
        video_frame_samples: int = 6,
        max_attachment_width: int = 960,
        extra_body: dict[str, Any] | None = None,
        prepare_command: list[str] | str | None = None,
        release_command: list[str] | str | None = None,
        lifecycle_timeout_sec: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env, "")
        self._provider_id = provider_name
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.video_frame_samples = video_frame_samples
        self.max_attachment_width = max_attachment_width
        self.extra_body = extra_body or {}
        self.prepare_command = prepare_command
        self.release_command = release_command
        self.lifecycle_timeout_sec = lifecycle_timeout_sec

    async def prepare(self) -> None:
        await run_lifecycle_command(self.prepare_command, timeout_sec=self.lifecycle_timeout_sec)

    async def release(self) -> None:
        await run_lifecycle_command(self.release_command, timeout_sec=self.lifecycle_timeout_sec)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        prompt = build_structured_prompt(request, response_model)
        schema = response_model.model_json_schema()

        with temporary_directory("book2video_llm_") as tmp:
            visuals = materialize_attachment_images(
                request,
                tmp,
                video_frame_samples=self.video_frame_samples,
                max_width=self.max_attachment_width,
            )
            if visuals:
                content: str | list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for label, image_path in visuals:
                    content.append({"type": "text", "text": f"Visual attachment: {label}"})
                    content.append({"type": "image_url", "image_url": {"url": file_to_data_uri(image_path)}})
            else:
                content = prompt

            body: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            body.update(self.extra_body)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

        text = data["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
        return response_model.model_validate(parse_json_text(str(text)))
