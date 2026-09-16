from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from book2video.schemas import GenerationIntent

T = TypeVar("T", bound=BaseModel)


class MediaAttachment(BaseModel):
    path: str
    kind: str = "file"
    label: str | None = None


class StructuredLLMRequest(BaseModel):
    stage: str
    instructions: str
    context: dict[str, Any] = Field(default_factory=dict)
    attachments: list[MediaAttachment] = Field(default_factory=list)


class LLMProvider(ABC):
    """A small interface intended to keep orchestration separate from model vendors.

    Lifecycle hooks are intentionally no-ops by default. Local providers can use them to keep a
    model resident through an LLM-heavy phase and release it before video generation.
    """

    async def prepare(self) -> None:
        return None

    async def release(self) -> None:
        return None

    @property
    @abstractmethod
    def provider_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_model: type[T],
    ) -> T:
        """Return data validated against response_model."""
        raise NotImplementedError


class VideoCapabilities(BaseModel):
    max_duration_sec: float = 10.0
    min_duration_sec: float = 1.0
    supports_image_refs: bool = True
    supports_video_continuation: bool = True
    supports_audio_refs: bool = False
    supports_multi_gpu_job: bool = False
    outputs_audio: bool = False
    # This is only a planning hint. Providers/models that do not have a stable estimate can leave it null.
    estimated_vram_gb: float | None = None


class VideoGenerationRequest(BaseModel):
    clip_id: str
    take_number: int
    duration_sec: float
    intent: GenerationIntent
    character_reference_images: list[str] = Field(default_factory=list)
    location_reference_images: list[str] = Field(default_factory=list)
    voice_reference_files: list[str] = Field(default_factory=list)
    continuation_video_path: str | None = None
    continuation_frame_path: str | None = None
    seed: int | None = None
    output_path: str
    extra: dict[str, Any] = Field(default_factory=dict)


class VideoGenerationResult(BaseModel):
    output_path: str
    seed: int | None = None
    generation_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoProvider(ABC):
    """The video backend can map a canonical request into any local or remote API.

    prepare/release make local model residency explicit without requiring cloud adapters to care.
    """

    async def prepare(self, device_groups: list[tuple[str, ...]]) -> None:
        return None

    async def release(self) -> None:
        return None

    @property
    @abstractmethod
    def provider_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> VideoCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        request: VideoGenerationRequest,
        devices: tuple[str, ...],
    ) -> VideoGenerationResult:
        raise NotImplementedError
