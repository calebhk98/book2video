from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ProviderConfig(BaseModel):
    # Labels/descriptions are optional metadata for the browser UI. The provider loader only needs
    # class_path and options, so older configs remain usable.
    class_path: str
    options: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    description: str | None = None


class GenerationConfig(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _legacy_device_groups_imply_manual(cls, value: Any) -> Any:
        # v1-v4 configs only had worker_device_groups. Treating an explicitly supplied legacy list
        # as manual preserves those layouts while new configs can default to discovery.
        if isinstance(value, dict) and "device_mode" not in value and value.get("worker_device_groups"):
            value = dict(value)
            value["device_mode"] = "manual"
        return value

    takes_per_clip: int = Field(default=3, ge=1)
    max_regeneration_rounds: int = Field(default=1, ge=0)
    seed_base: int = 1000

    # Automatic discovery is being tried as the default so replacing, adding, or losing a GPU does
    # not require editing cuda indices. Manual groups remain available for unusual model layouts.
    device_mode: Literal["auto", "manual"] = "auto"
    worker_device_groups: list[list[str]] = Field(default_factory=list)
    auto_device_strategy: Literal["one_per_gpu", "all_gpus"] = "one_per_gpu"
    reserve_vram_gb: float = Field(default=1.0, ge=0.0)
    min_gpu_memory_gb: float = Field(default=0.0, ge=0.0)

    # When true, LLM-heavy and video-heavy work is deliberately grouped into phases. Providers that
    # implement prepare/release can use those boundaries to load one local model family at a time.
    phase_swap_local_models: bool = True


class ReviewThresholds(BaseModel):
    min_identity: float = Field(default=0.72, ge=0.0, le=1.0)
    min_continuity: float = Field(default=0.60, ge=0.0, le=1.0)
    min_story_fidelity: float = Field(default=0.72, ge=0.0, le=1.0)
    min_visual_quality: float = Field(default=0.55, ge=0.0, le=1.0)


class DefaultsConfig(BaseModel):
    # The browser studio is being used as the default path so the CLI can remain available without
    # requiring it for normal operation.
    action: Literal["studio", "run", "viewer", "assemble"] = "studio"
    chapter_source: Literal["config", "path", "text", "picker"] = "config"
    launch_viewer_after_run: bool = True
    open_browser: bool = True
    viewer_host: str = "127.0.0.1"
    viewer_port: int = 8765
    viewer_run_dir: str | None = None
    auto_assemble_on_selection: bool = False
    viewer_preload_adjacent: bool = True


class ProjectConfig(BaseModel):
    chapter_text: str = ""
    chapter_path: str | None = None
    output_root: str = "./book2video_output"
    asset_registry_path: str | None = None
    knowledge_paths: list[str] = Field(default_factory=list)

    # Named profiles let several providers/models remain configured at the same time. The legacy
    # llm/video blocks are retained as a compatibility path for v1-v3 config files.
    active_llm_profile: str | None = None
    active_video_profile: str | None = None
    llm_profiles: dict[str, ProviderConfig] = Field(default_factory=dict)
    video_profiles: dict[str, ProviderConfig] = Field(default_factory=dict)
    llm: ProviderConfig | None = None
    video: ProviderConfig | None = None

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    review: ReviewThresholds = Field(default_factory=ReviewThresholds)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    def available_llm_profiles(self) -> dict[str, ProviderConfig]:
        if self.llm_profiles:
            return self.llm_profiles
        if self.llm is not None:
            return {"default": self.llm}
        return {}

    def available_video_profiles(self) -> dict[str, ProviderConfig]:
        if self.video_profiles:
            return self.video_profiles
        if self.video is not None:
            return {"default": self.video}
        return {}

    def resolve_llm_profile(self, name: str | None = None) -> tuple[str, ProviderConfig]:
        profiles = self.available_llm_profiles()
        if not profiles:
            raise ValueError("No LLM provider/profile is configured.")
        chosen = name or self.active_llm_profile or next(iter(profiles))
        if chosen not in profiles:
            raise ValueError(f"Unknown LLM profile {chosen!r}. Available: {', '.join(profiles)}")
        return chosen, profiles[chosen]

    def resolve_video_profile(self, name: str | None = None) -> tuple[str, ProviderConfig]:
        profiles = self.available_video_profiles()
        if not profiles:
            raise ValueError("No video provider/profile is configured.")
        chosen = name or self.active_video_profile or next(iter(profiles))
        if chosen not in profiles:
            raise ValueError(f"Unknown video profile {chosen!r}. Available: {', '.join(profiles)}")
        return chosen, profiles[chosen]


def load_config(path: str | Path) -> ProjectConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)
