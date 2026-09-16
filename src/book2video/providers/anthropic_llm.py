from __future__ import annotations

import os
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest
from book2video.providers.provider_utils import (
    build_structured_prompt,
    file_to_base64,
    materialize_attachment_images,
    mime_type,
    temporary_directory,
)

T = TypeVar("T", bound=BaseModel)


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Messages adapter using a forced schema-bearing tool for structured return values."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        max_tokens: int = 8192,
        timeout_sec: float = 600.0,
        video_frame_samples: int = 6,
        max_attachment_width: int = 960,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.video_frame_samples = video_frame_samples
        self.max_attachment_width = max_attachment_width
        self.extra_body = extra_body or {}

    @property
    def provider_id(self) -> str:
        return "anthropic"

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        if not self.api_key:
            raise RuntimeError("Missing Anthropic API key; set ANTHROPIC_API_KEY or configure api_key.")
        prompt = build_structured_prompt(request, response_model)

        with temporary_directory("book2video_anthropic_") as tmp:
            visuals = materialize_attachment_images(
                request,
                tmp,
                video_frame_samples=self.video_frame_samples,
                max_width=self.max_attachment_width,
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for label, image_path in visuals:
                content.append({"type": "text", "text": f"Visual attachment: {label}"})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type(image_path),
                            "data": file_to_base64(image_path),
                        },
                    }
                )

            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}],
                "tools": [
                    {
                        "name": "return_result",
                        "description": "Return the requested structured result.",
                        "input_schema": response_model.model_json_schema(),
                    }
                ],
                "tool_choice": {"type": "tool", "name": "return_result"},
            }
            body.update(self.extra_body)
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json",
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=30.0)) as client:
                response = await client.post(f"{self.base_url}/messages", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "return_result":
                return response_model.model_validate(block.get("input", {}))
        raise RuntimeError("Anthropic response did not contain the forced return_result tool call.")
