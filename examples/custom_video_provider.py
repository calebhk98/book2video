"""Skeleton for adding Wan, SkyReels, an MCP call, or a remote API."""

from book2video.providers.base import (
    VideoCapabilities,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProvider,
)


class MyVideoProvider(VideoProvider):
    @property
    def provider_id(self) -> str:
        return "my-video-provider"

    @property
    def capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(
            max_duration_sec=10.0,
            supports_image_refs=True,
            supports_video_continuation=True,
            supports_audio_refs=True,
            supports_multi_gpu_job=True,
            outputs_audio=True,
        )

    async def generate(self, request: VideoGenerationRequest, devices: tuple[str, ...]) -> VideoGenerationResult:
        # The adapter can interpret devices as one distributed job or ignore them when
        # an external scheduler owns GPU placement. Keeping that choice here avoids
        # baking one model's launch strategy into the chapter pipeline.
        #
        # request.character_reference_images and request.location_reference_images
        # contain the resolved visual references for this clip.
        # request.continuation_frame_path is populated for split continuations when ffmpeg
        # can extract a last frame from the selected predecessor.
        raise NotImplementedError
