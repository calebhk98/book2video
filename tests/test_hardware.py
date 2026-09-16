from book2video.hardware import GPUInfo, plan_video_devices
from book2video.providers.base import VideoCapabilities


def gpu(index: int, gb: float) -> GPUInfo:
    return GPUInfo(index=index, name=f"GPU {index}", memory_total_gb=gb, memory_free_gb=gb)


def test_auto_plan_uses_one_worker_per_gpu_when_requirement_is_unknown():
    plan = plan_video_devices(
        gpus=[gpu(0, 24), gpu(1, 24)],
        capabilities=VideoCapabilities(supports_multi_gpu_job=True),
    )
    assert plan.groups == [("cuda:0",), ("cuda:1",)]


def test_auto_plan_groups_cards_when_model_estimate_does_not_fit_one_card():
    plan = plan_video_devices(
        gpus=[gpu(0, 24), gpu(1, 24)],
        capabilities=VideoCapabilities(supports_multi_gpu_job=True, estimated_vram_gb=36),
        reserve_vram_gb=1,
    )
    assert plan.groups == [("cuda:0", "cuda:1")]


def test_auto_plan_adapts_to_single_larger_replacement_gpu():
    plan = plan_video_devices(
        gpus=[gpu(0, 48)],
        capabilities=VideoCapabilities(supports_multi_gpu_job=True, estimated_vram_gb=36),
        reserve_vram_gb=1,
    )
    assert plan.groups == [("cuda:0",)]
