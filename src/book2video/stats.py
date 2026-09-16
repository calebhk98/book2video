from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from book2video.providers.base import (
    LLMProvider,
    StructuredLLMRequest,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProvider,
)


class RunStats(BaseModel):
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    elapsed_seconds: float = 0.0
    current_phase: str = "starting"
    phase_durations: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_seconds: float = 0.0
    llm_calls_by_stage: dict[str, int] = Field(default_factory=dict)
    video_calls: int = 0
    video_wall_seconds: float = 0.0
    video_provider_seconds: float = 0.0
    requested_video_seconds: float = 0.0
    video_calls_by_device_group: dict[str, int] = Field(default_factory=dict)
    generated_takes: int = 0
    regeneration_takes: int = 0
    phase_completed: int | None = None
    phase_total: int | None = None
    phase_eta_seconds: float | None = None
    device_groups: list[list[str]] = Field(default_factory=list)
    device_plan_explanation: str | None = None
    lifecycle_events: list[str] = Field(default_factory=list)

    @property
    def video_seconds_per_output_second(self) -> float | None:
        if self.requested_video_seconds <= 0:
            return None
        return self.video_provider_seconds / self.requested_video_seconds


class StatsRecorder:
    def __init__(self) -> None:
        self.stats = RunStats()
        self._phase_started = time.perf_counter()
        self._last_phase = "starting"

    def switch_phase(self, phase: str) -> None:
        now = time.perf_counter()
        self.stats.phase_durations[self._last_phase] = self.stats.phase_durations.get(self._last_phase, 0.0) + (now - self._phase_started)
        self._phase_started = now
        self._last_phase = phase
        self.stats.current_phase = phase
        self.stats.phase_completed = None
        self.stats.phase_total = None
        self.stats.phase_eta_seconds = None

    def set_phase_progress(self, completed: int, total: int, elapsed_for_phase: float | None = None) -> None:
        self.stats.phase_completed = completed
        self.stats.phase_total = total
        if completed > 0 and total > completed:
            elapsed = elapsed_for_phase if elapsed_for_phase is not None else (time.perf_counter() - self._phase_started)
            rate = elapsed / completed
            self.stats.phase_eta_seconds = rate * (total - completed)
        elif total and completed >= total:
            self.stats.phase_eta_seconds = 0.0

    def lifecycle(self, message: str) -> None:
        self.stats.lifecycle_events.append(message)
        self.stats.lifecycle_events = self.stats.lifecycle_events[-40:]

    def snapshot(self) -> dict[str, Any]:
        self.stats.elapsed_seconds = time.time() - self.stats.started_at
        return self.stats.model_dump(mode="json") | {
            "video_seconds_per_output_second": self.stats.video_seconds_per_output_second,
        }

    def finish(self) -> dict[str, Any]:
        self.switch_phase("finished")
        self.stats.finished_at = time.time()
        self.stats.elapsed_seconds = self.stats.finished_at - self.stats.started_at
        return self.snapshot()


class InstrumentedLLMProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, recorder: StatsRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    @property
    def provider_id(self) -> str:
        return self.inner.provider_id

    async def prepare(self) -> None:
        await self.inner.prepare()

    async def release(self) -> None:
        await self.inner.release()

    async def generate_structured(self, request: StructuredLLMRequest, response_model):
        started = time.perf_counter()
        try:
            return await self.inner.generate_structured(request, response_model)
        finally:
            elapsed = time.perf_counter() - started
            self.recorder.stats.llm_calls += 1
            self.recorder.stats.llm_seconds += elapsed
            counts = self.recorder.stats.llm_calls_by_stage
            counts[request.stage] = counts.get(request.stage, 0) + 1


class InstrumentedVideoProvider(VideoProvider):
    def __init__(self, inner: VideoProvider, recorder: StatsRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    @property
    def provider_id(self) -> str:
        return self.inner.provider_id

    @property
    def capabilities(self):
        return self.inner.capabilities

    async def prepare(self, device_groups: list[tuple[str, ...]]) -> None:
        await self.inner.prepare(device_groups)

    async def release(self) -> None:
        await self.inner.release()

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        started = time.perf_counter()
        result: VideoGenerationResult | None = None
        try:
            result = await self.inner.generate(request, devices)
            return result
        finally:
            elapsed = time.perf_counter() - started
            stats = self.recorder.stats
            stats.video_calls += 1
            stats.video_wall_seconds += elapsed
            stats.video_provider_seconds += (
                result.generation_seconds if result is not None and result.generation_seconds is not None else elapsed
            )
            stats.requested_video_seconds += request.duration_sec
            key = "+".join(devices) if devices else "provider-default"
            stats.video_calls_by_device_group[key] = stats.video_calls_by_device_group.get(key, 0) + 1
