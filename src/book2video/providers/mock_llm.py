from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

from book2video.providers.base import LLMProvider, StructuredLLMRequest
from book2video.schemas import (
    BeatCritique,
    BeatPlan,
    BoundaryCritique,
    ClipJudgement,
    GenerationIntent,
    SceneBoundary,
    SceneBoundaryResult,
    SceneSplitDecision,
    Shot,
    ShotPlan,
    TakeAssessment,
    TimedBeat,
    TimingPlan,
    Beat,
)

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """A deterministic-ish planning stand-in so the pipeline can be tested without an API."""

    def __init__(self, provider_name: str = "mock-llm") -> None:
        self._provider_id = provider_name

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def generate_structured(self, request: StructuredLLMRequest, response_model: type[T]) -> T:
        stage = request.stage
        c = request.context

        if stage == "scene_boundary":
            numbered = c["chapter_numbered"].splitlines()
            raw_lines = [re.sub(r"^\d+:\s?", "", x) for x in numbered]
            breaks = [0]
            for i, line in enumerate(raw_lines, start=1):
                stripped = line.strip()
                if re.fullmatch(r"[_\-*=]{5,}", stripped):
                    breaks.append(i)
                if re.match(r"^#{1,3}\s+SCENE\b", stripped, flags=re.I) and i != 1:
                    breaks.append(i - 1)
            breaks = sorted(set(x for x in breaks if 0 <= x < len(raw_lines)))
            spans: list[tuple[int, int]] = []
            start = 1
            for b in breaks[1:]:
                if b >= start:
                    spans.append((start, b - 1 if raw_lines[b - 1].strip() else b))
                    start = b + 1
            spans.append((start, len(raw_lines)))
            scenes = [
                SceneBoundary(scene_id=f"scene_{i:03d}", start_line=s, end_line=e, reason="mock heuristic")
                for i, (s, e) in enumerate(spans, start=1)
                if e >= s
            ]
            return response_model.model_validate(SceneBoundaryResult(scenes=scenes).model_dump())

        if stage == "scene_critic":
            return response_model.model_validate(
                SceneSplitDecision(scene_id=c["scene_id"], should_split=False, reason="mock keeps proposed scene").model_dump()
            )

        if stage == "beat_breakdown":
            scene = c["scene"]
            text = scene["text"]
            parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
            beats = []
            for i, part in enumerate(parts, start=1):
                beats.append(
                    Beat(
                        beat_id=f"{scene['scene_id']}_b{i:03d}",
                        action=part,
                        source_excerpt=part,
                        characters=scene.get("character_ids", []),
                        critical=i <= 2,
                    )
                )
            return response_model.model_validate(BeatPlan(scene_id=scene["scene_id"], beats=beats).model_dump())

        if stage == "beat_critic":
            return response_model.model_validate(BeatCritique(verdict="pass").model_dump())

        if stage == "timing":
            scene = c["scene"]
            timed = []
            for beat in c["beats"]["beats"]:
                words = len(beat["action"].split()) + len((beat.get("dialogue") or "").split())
                duration = max(2.0, words / 2.6 + 0.75)
                timed.append(
                    TimedBeat(
                        beat_id=beat["beat_id"],
                        duration_sec=duration,
                        dialogue_sec=max(0.0, len((beat.get("dialogue") or "").split()) / 2.6),
                        action_sec=duration,
                        reasoning="mock words-per-second estimate",
                    )
                )
            return response_model.model_validate(TimingPlan(scene_id=scene["scene_id"], beats=timed).model_dump())

        if stage == "shot_plan":
            scene = c["scene"]
            beats = c["beats"]["beats"]
            timing = {x["beat_id"]: x for x in c["timing"]["beats"]}
            shots = []
            for i, beat in enumerate(beats, start=1):
                shots.append(
                    Shot(
                        shot_id=f"{scene['scene_id']}_sh{i:03d}",
                        scene_id=scene["scene_id"],
                        beat_ids=[beat["beat_id"]],
                        target_duration_sec=timing[beat["beat_id"]]["duration_sec"],
                        visual_description=beat["action"],
                        characters_visible=beat.get("characters", []),
                        location=scene.get("location"),
                    )
                )
            return response_model.model_validate(ShotPlan(scene_id=scene["scene_id"], shots=shots).model_dump())

        if stage == "generation_intent":
            shot = c["shot"]
            payload = GenerationIntent(
                clip_id=c["clip"]["clip_id"],
                visual_prompt=shot["visual_description"],
                motion_prompt=shot.get("camera", ""),
                character_ids=c.get("character_ids", []),
                location_id=c.get("location_id"),
                continuity_notes=[str(shot.get("state_before", {})), str(shot.get("state_after", {}))],
            )
            return response_model.model_validate(payload.model_dump())

        if stage == "video_judge":
            takes = c.get("takes", [])
            pick = takes[0]["take_number"] if takes else 1
            assessments = [
                TakeAssessment(
                    take_number=t["take_number"],
                    identity_score=0.9,
                    continuity_score=0.9,
                    story_fidelity_score=0.9,
                    visual_quality_score=0.8,
                )
                for t in takes
            ]
            return response_model.model_validate(
                ClipJudgement(
                    clip_id=c["clip"]["clip_id"],
                    recommended_take=pick,
                    regenerate=False,
                    reason="mock judge selects first take",
                    assessments=assessments,
                ).model_dump()
            )

        raise NotImplementedError(f"Mock LLM does not implement stage {stage!r}")
