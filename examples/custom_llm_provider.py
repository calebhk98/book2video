"""Skeleton for adding another LLM without modifying the pipeline."""

from typing import TypeVar
from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest

T = TypeVar("T", bound=BaseModel)


class MyLLMProvider(LLMProvider):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @property
    def provider_id(self) -> str:
        return f"my-llm:{self.model}"

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        # One possible adapter shape is to ask the provider for JSON matching
        # response_model.model_json_schema(), then validate it here before returning.
        schema = response_model.model_json_schema()
        raise NotImplementedError("Call your LLM here, then return response_model.model_validate(payload)")
