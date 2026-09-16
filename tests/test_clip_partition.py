from book2video.providers.base import VideoCapabilities
from book2video.schemas import Shot, ShotPlan
from book2video.stages.clips import partition_shots


def test_balanced_partition():
    plan = ShotPlan(
        scene_id="s1",
        shots=[
            Shot(
                shot_id="s1_sh1",
                scene_id="s1",
                beat_ids=["b1"],
                target_duration_sec=21.0,
                visual_description="test",
            )
        ],
    )
    clips = partition_shots([plan], VideoCapabilities(max_duration_sec=10.0))
    assert len(clips) == 3
    assert all(abs(c.target_duration_sec - 7.0) < 0.01 for c in clips)
    assert clips[1].continuation_from_clip_id == clips[0].clip_id
