from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from book2video.cache import StageCache
from book2video.providers.base import LLMProvider, StructuredLLMRequest

T = TypeVar("T", bound=BaseModel)


async def cached_llm_call(
    *,
    cache: StageCache,
    llm: LLMProvider,
    stage: str,
    instructions: str,
    context: dict[str, Any],
    response_model: type[T],
    version: str = "1",
) -> T:
    key = cache.key(
        stage,
        {
            "provider": llm.provider_id,
            "instructions": instructions,
            "context": context,
        },
        version=version,
    )
    cached = cache.get(stage, key, response_model)
    if cached is not None:
        return cached  # type: ignore[return-value]

    response = await llm.generate_structured(
        StructuredLLMRequest(stage=stage, instructions=instructions, context=context),
        response_model,
    )
    cache.put(stage, key, response)
    return response
