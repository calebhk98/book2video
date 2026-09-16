from __future__ import annotations

import math

from book2video.providers.base import VideoCapabilities
from book2video.schemas import ClipSpec, ShotPlan


def partition_shots(shot_plans: list[ShotPlan], capabilities: VideoCapabilities) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for plan in shot_plans:
        for shot in plan.shots:
            total = shot.target_duration_sec
            max_len = capabilities.max_duration_sec
            parts = max(1, math.ceil(total / max_len))
            # Balanced pieces are being tried here so an 11-second shot does not become
            # one 10-second clip plus a nearly useless 1-second tail.
            each = total / parts
            previous: str | None = None
            for idx in range(parts):
                clip_id = f"{shot.shot_id}_c{idx + 1:02d}"
                clips.append(
                    ClipSpec(
                        clip_id=clip_id,
                        shot_id=shot.shot_id,
                        scene_id=shot.scene_id,
                        segment_index=idx,
                        segment_count=parts,
                        target_duration_sec=each,
                        continuation_from_clip_id=previous if parts > 1 else None,
                        transition_after=shot.transition_after if idx == parts - 1 else "continuous_generation",
                    )
                )
                previous = clip_id
    return clips
