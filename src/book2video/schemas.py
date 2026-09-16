from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class SceneBoundary(BaseModel):
    scene_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    location_hint: str | None = None
    time_hint: str | None = None
    characters_hint: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SceneBoundaryResult(BaseModel):
    scenes: list[SceneBoundary]


class BoundaryCritique(BaseModel):
    verdict: Literal["pass", "revise"] = "pass"
    issues: list[str] = Field(default_factory=list)
    revised_scenes: list[SceneBoundary] = Field(default_factory=list)




class SceneSplitDecision(BaseModel):
    scene_id: str
    should_split: bool = False
    split_after_lines: list[int] = Field(default_factory=list)
    reason: str = ""


class SceneRecord(BaseModel):
    scene_id: str
    start_line: int
    end_line: int
    text: str
    location: str | None = None
    time_hint: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    context_snippets: list[str] = Field(default_factory=list)


class Beat(BaseModel):
    beat_id: str
    action: str
    dialogue: str | None = None
    speaker_id: str | None = None
    characters: list[str] = Field(default_factory=list)
    important_visuals: list[str] = Field(default_factory=list)
    source_excerpt: str = ""
    critical: bool = False


class BeatPlan(BaseModel):
    scene_id: str
    beats: list[Beat]


class BeatCritique(BaseModel):
    verdict: Literal["pass", "revise"] = "pass"
    missing_beats: list[str] = Field(default_factory=list)
    invented_beats: list[str] = Field(default_factory=list)
    chronology_errors: list[str] = Field(default_factory=list)
    character_errors: list[str] = Field(default_factory=list)
    revised_plan: BeatPlan | None = None


class TimedBeat(BaseModel):
    beat_id: str
    duration_sec: float = Field(gt=0)
    dialogue_sec: float = Field(default=0.0, ge=0)
    action_sec: float = Field(default=0.0, ge=0)
    hold_sec: float = Field(default=0.0, ge=0)
    can_overlap_next: bool = False
    reasoning: str = ""


class TimingPlan(BaseModel):
    scene_id: str
    beats: list[TimedBeat]


class Shot(BaseModel):
    shot_id: str
    scene_id: str
    beat_ids: list[str]
    target_duration_sec: float = Field(gt=0)
    framing: str = "medium"
    camera: str = "locked"
    visual_description: str
    characters_visible: list[str] = Field(default_factory=list)
    characters_audible: list[str] = Field(default_factory=list)
    location: str | None = None
    transition_after: Literal[
        "cut", "crossfade", "fade_black", "match_cut", "continuous_generation", "generated_transition"
    ] = "cut"
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)


class ShotPlan(BaseModel):
    scene_id: str
    shots: list[Shot]


class ClipSpec(BaseModel):
    clip_id: str
    shot_id: str
    scene_id: str
    segment_index: int = 0
    segment_count: int = 1
    target_duration_sec: float = Field(gt=0)
    continuation_from_clip_id: str | None = None
    transition_after: str = "cut"


class GenerationIntent(BaseModel):
    clip_id: str
    visual_prompt: str
    motion_prompt: str = ""
    negative_prompt: str = ""
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    dialogue: list[str] = Field(default_factory=list)
    continuity_notes: list[str] = Field(default_factory=list)


class GeneratedTake(BaseModel):
    clip_id: str
    take_number: int
    output_path: str
    seed: int | None = None
    provider_id: str
    generation_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.output_path)


class TakeAssessment(BaseModel):
    take_number: int
    identity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    continuity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    story_fidelity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    critical_failures: list[str] = Field(default_factory=list)
    minor_issues: list[str] = Field(default_factory=list)


class ClipJudgement(BaseModel):
    clip_id: str
    recommended_take: int
    regenerate: bool = False
    reason: str = ""
    assessments: list[TakeAssessment] = Field(default_factory=list)


class SelectedClip(BaseModel):
    clip_id: str
    scene_id: str
    shot_id: str
    selected_take: int
    selected_path: str
    transition_after: str = "cut"
    judgement: ClipJudgement | None = None


class PipelineSummary(BaseModel):
    run_dir: str
    scenes: int
    shots: int
    clips: int
    generated_takes: int
    selected_clips: int
    rough_cut_path: str | None = None
    review_path: str | None = None
    stats_path: str | None = None
    elapsed_seconds: float | None = None
    device_groups: list[list[str]] = Field(default_factory=list)
