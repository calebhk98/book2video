import asyncio
from pathlib import Path

from book2video.config import GenerationConfig, ProjectConfig
from book2video.pipeline import Book2VideoPipeline
from book2video.providers.mock_llm import MockLLMProvider
from book2video.providers.mock_video import MockVideoProvider


class TracedLLM(MockLLMProvider):
    def __init__(self, events):
        super().__init__()
        self.events = events

    async def prepare(self):
        self.events.append("llm_prepare")

    async def release(self):
        self.events.append("llm_release")


class TracedVideo(MockVideoProvider):
    def __init__(self, events):
        super().__init__(max_duration_sec=10.0)
        self.events = events

    async def prepare(self, device_groups):
        self.events.append(("video_prepare", tuple(device_groups)))

    async def release(self):
        self.events.append("video_release")


def test_pipeline_batches_llm_and_video_residency(tmp_path: Path):
    events = []
    config = ProjectConfig(
        output_root=str(tmp_path / "out"),
        generation=GenerationConfig(
            device_mode="manual",
            worker_device_groups=[["cuda:0"]],
            takes_per_clip=1,
            max_regeneration_rounds=0,
            phase_swap_local_models=True,
        ),
    )
    pipeline = Book2VideoPipeline(config, TracedLLM(events), TracedVideo(events))
    asyncio.run(pipeline.run("A child reads a book.", "phase_test"))

    first_release = events.index("llm_release")
    video_prepare = next(i for i, x in enumerate(events) if isinstance(x, tuple) and x[0] == "video_prepare")
    video_release = events.index("video_release")
    second_llm_prepare = events.index("llm_prepare", 1)
    assert first_release < video_prepare < video_release < second_llm_prepare
