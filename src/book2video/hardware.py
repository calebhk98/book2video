from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field

from book2video.providers.base import VideoCapabilities


class GPUInfo(BaseModel):
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    utilization_pct: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"


class HardwareSnapshot(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    gpus: list[GPUInfo] = Field(default_factory=list)
    detection_error: str | None = None


@dataclass(slots=True)
class DevicePlan:
    groups: list[tuple[str, ...]]
    explanation: str


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def detect_nvidia_gpus() -> HardwareSnapshot:
    """Read a lightweight GPU snapshot through nvidia-smi when it is available.

    To keep Book2Video usable on machines without NVIDIA tooling, failure here is treated as
    hardware information being unavailable rather than a fatal startup error.
    """

    if shutil.which("nvidia-smi") is None:
        return HardwareSnapshot(detection_error="nvidia-smi was not found")

    query = (
        "index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,"
        "power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:  # pragma: no cover - depends on host drivers
        return HardwareSnapshot(detection_error=f"nvidia-smi failed: {exc}")

    gpus: list[GPUInfo] = []
    try:
        for raw in completed.stdout.splitlines():
            if not raw.strip():
                continue
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) < 8:
                continue
            total_mb = float(parts[2])
            free_mb = float(parts[3])
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_gb=round(total_mb / 1024.0, 2),
                    memory_free_gb=round(free_mb / 1024.0, 2),
                    utilization_pct=_float_or_none(parts[4]),
                    temperature_c=_float_or_none(parts[5]),
                    power_draw_w=_float_or_none(parts[6]),
                    power_limit_w=_float_or_none(parts[7]),
                )
            )
    except Exception as exc:
        return HardwareSnapshot(gpus=gpus, detection_error=f"Could not parse nvidia-smi output: {exc}")

    return HardwareSnapshot(gpus=gpus)


def _usable_memory(gpu: GPUInfo, reserve_gb: float) -> float:
    # Total memory is used for planning because another local model may intentionally be unloaded
    # before video generation. Free memory is still shown live in the Studio for diagnosis.
    return max(0.0, gpu.memory_total_gb - reserve_gb)


def plan_video_devices(
    *,
    gpus: Iterable[GPUInfo],
    capabilities: VideoCapabilities,
    strategy: str = "one_per_gpu",
    reserve_vram_gb: float = 1.0,
    min_gpu_memory_gb: float = 0.0,
) -> DevicePlan:
    """Create a device grouping from the GPUs that are present right now.

    A profile may advertise an estimated VRAM requirement. When it does, the planner attempts a
    single-card layout first and then an aggregate multi-GPU layout if the provider says it can use
    more than one card. Without an estimate, one worker per detected GPU is the conservative default.
    """

    candidates = [
        gpu for gpu in sorted(gpus, key=lambda x: x.index)
        if gpu.memory_total_gb >= min_gpu_memory_gb
    ]
    if not candidates:
        return DevicePlan([], "No eligible NVIDIA GPUs were detected for automatic video planning.")

    required = capabilities.estimated_vram_gb
    if strategy == "all_gpus":
        if len(candidates) == 1:
            return DevicePlan([(candidates[0].device,)], "Automatic strategy requested all GPUs; one GPU is present.")
        if not capabilities.supports_multi_gpu_job:
            return DevicePlan(
                [(gpu.device,) for gpu in candidates],
                "The profile does not advertise multi-GPU jobs, so automatic planning uses one worker per GPU instead.",
            )
        return DevicePlan(
            [tuple(gpu.device for gpu in candidates)],
            "Automatic strategy grouped all detected GPUs into one video worker.",
        )

    if required is not None:
        single = [gpu for gpu in candidates if _usable_memory(gpu, reserve_vram_gb) >= required]
        if single:
            return DevicePlan(
                [(gpu.device,) for gpu in single],
                f"Each selected GPU has enough planned VRAM for the profile's ~{required:.1f} GB estimate.",
            )
        aggregate = sum(_usable_memory(gpu, reserve_vram_gb) for gpu in candidates)
        if capabilities.supports_multi_gpu_job and aggregate >= required:
            return DevicePlan(
                [tuple(gpu.device for gpu in candidates)],
                f"No single GPU meets the ~{required:.1f} GB estimate; aggregate planned VRAM does, so the GPUs are grouped.",
            )
        return DevicePlan(
            [],
            f"Detected GPUs do not satisfy the profile's ~{required:.1f} GB planned VRAM estimate with the current reserve.",
        )

    return DevicePlan(
        [(gpu.device,) for gpu in candidates],
        "No model VRAM estimate is configured, so automatic planning uses one independent worker per detected GPU.",
    )
