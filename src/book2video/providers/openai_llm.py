from __future__ import annotations

import os
from typing import Any

from book2video.providers.openai_compatible_llm import OpenAICompatibleLLMProvider


class OpenAILLMProvider(OpenAICompatibleLLMProvider):
    """OpenAI adapter using the Chat Completions structured-output shape."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: float = 600.0,
        temperature: float = 0.0,
        video_frame_samples: int = 6,
        max_attachment_width: int = 960,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.getenv(api_key_env, ""),
            provider_name="openai",
            timeout_sec=timeout_sec,
            temperature=temperature,
            video_frame_samples=video_frame_samples,
            max_attachment_width=max_attachment_width,
            extra_body=extra_body,
        )
