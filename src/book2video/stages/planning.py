from __future__ import annotations

from book2video.assets import AssetRegistry
from book2video.cache import StageCache
from book2video.knowledge import LocalKnowledgeIndex
from book2video.prompts import (
    BEAT_BREAKDOWN,
    BEAT_CRITIC,
    GENERATION_INTENT,
    SCENE_BOUNDARY,
    SCENE_CRITIC,
    SHOT_PLAN,
    TIMING,
)
from book2video.providers.base import LLMProvider
from book2video.schemas import (
    BeatCritique,
    BeatPlan,
    BoundaryCritique,
    ClipSpec,
    GenerationIntent,
    SceneBoundary,
    SceneBoundaryResult,
    SceneSplitDecision,
    SceneRecord,
    ShotPlan,
    TimingPlan,
)
from book2video.stages.common import cached_llm_call


def numbered_lines(text: str) -> str:
    return "\n".join(f"{i}: {line}" for i, line in enumerate(text.splitlines(), start=1))


async def detect_scenes(llm: LLMProvider, cache: StageCache, chapter: str) -> SceneBoundaryResult:
    return await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="scene_boundary",
        instructions=SCENE_BOUNDARY,
        context={"chapter_numbered": numbered_lines(chapter)},
        response_model=SceneBoundaryResult,
        version="2",
    )


async def critique_scenes(
    llm: LLMProvider,
    cache: StageCache,
    chapter: str,
    proposed: SceneBoundaryResult,
) -> SceneBoundaryResult:
    lines = chapter.splitlines()
    refined: list[SceneBoundary] = []
    for boundary in proposed.scenes:
        start = max(1, boundary.start_line)
        end = min(len(lines), boundary.end_line)
        scene_text = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        decision = await cached_llm_call(
            cache=cache,
            llm=llm,
            stage="scene_critic",
            instructions=SCENE_CRITIC,
            context={
                "scene_id": boundary.scene_id,
                "start_line": start,
                "end_line": end,
                "scene_numbered": scene_text,
            },
            response_model=SceneSplitDecision,
            version="3",
        )

        valid_splits = sorted(
            {x for x in decision.split_after_lines if start <= x < end}
        ) if decision.should_split else []
        if not valid_splits:
            refined.append(boundary)
            continue

        segment_start = start
        split_points = [*valid_splits, end]
        for part_index, segment_end in enumerate(split_points, start=1):
            refined.append(
                SceneBoundary(
                    scene_id=f"{boundary.scene_id}_{part_index:02d}",
                    start_line=segment_start,
                    end_line=segment_end,
                    location_hint=boundary.location_hint,
                    time_hint=boundary.time_hint,
                    characters_hint=boundary.characters_hint,
                    reason=decision.reason or boundary.reason,
                    confidence=boundary.confidence,
                )
            )
            segment_start = segment_end + 1

    return SceneBoundaryResult(scenes=refined)


def materialize_scenes(
    chapter: str,
    boundaries: SceneBoundaryResult,
    assets: AssetRegistry,
    knowledge: LocalKnowledgeIndex,
) -> list[SceneRecord]:
    lines = chapter.splitlines()
    out: list[SceneRecord] = []
    for boundary in boundaries.scenes:
        start = max(1, boundary.start_line)
        end = min(len(lines), boundary.end_line)
        text = "\n".join(lines[start - 1 : end]).strip()
        chars = assets.resolve_characters(text)
        location = assets.resolve_location(text, boundary.location_hint)
        context = knowledge.search(" ".join([text[:1200], *chars, location or ""]), top_k=6)
        out.append(
            SceneRecord(
                scene_id=boundary.scene_id,
                start_line=start,
                end_line=end,
                text=text,
                location=location,
                time_hint=boundary.time_hint,
                character_ids=chars,
                context_snippets=context,
            )
        )
    return out


async def build_beats(
    llm: LLMProvider,
    cache: StageCache,
    scene: SceneRecord,
    assets: AssetRegistry,
) -> BeatPlan:
    context = {
        "scene": scene.model_dump(mode="json"),
        "character_context": assets.character_context(scene.character_ids),
        "location_context": assets.location_context(scene.location),
        "retrieved_context": scene.context_snippets,
    }
    draft = await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="beat_breakdown",
        instructions=BEAT_BREAKDOWN,
        context=context,
        response_model=BeatPlan,
        version="2",
    )
    critique = await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="beat_critic",
        instructions=BEAT_CRITIC,
        context={**context, "proposed": draft.model_dump(mode="json")},
        response_model=BeatCritique,
        version="2",
    )
    return critique.revised_plan if critique.verdict == "revise" and critique.revised_plan else draft


async def time_beats(
    llm: LLMProvider,
    cache: StageCache,
    scene: SceneRecord,
    beats: BeatPlan,
) -> TimingPlan:
    return await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="timing",
        instructions=TIMING,
        context={"scene": scene.model_dump(mode="json"), "beats": beats.model_dump(mode="json")},
        response_model=TimingPlan,
        version="2",
    )


async def plan_shots(
    llm: LLMProvider,
    cache: StageCache,
    scene: SceneRecord,
    beats: BeatPlan,
    timing: TimingPlan,
    assets: AssetRegistry,
) -> ShotPlan:
    return await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="shot_plan",
        instructions=SHOT_PLAN,
        context={
            "scene": scene.model_dump(mode="json"),
            "beats": beats.model_dump(mode="json"),
            "timing": timing.model_dump(mode="json"),
            "character_context": assets.character_context(scene.character_ids),
            "location_context": assets.location_context(scene.location),
        },
        response_model=ShotPlan,
        version="2",
    )


async def compile_generation_intent(
    llm: LLMProvider,
    cache: StageCache,
    scene: SceneRecord,
    beats: BeatPlan,
    shot_plan: ShotPlan,
    clip: ClipSpec,
    assets: AssetRegistry,
) -> GenerationIntent:
    shot = next(x for x in shot_plan.shots if x.shot_id == clip.shot_id)
    related_beats = [b for b in beats.beats if b.beat_id in shot.beat_ids]
    char_ids = list(dict.fromkeys([*shot.characters_visible, *shot.characters_audible]))
    if not char_ids:
        char_ids = assets.resolve_characters(" ".join([shot.visual_description, *(b.action for b in related_beats)]))
    location = assets.resolve_location(shot.visual_description, shot.location or scene.location)
    return await cached_llm_call(
        cache=cache,
        llm=llm,
        stage="generation_intent",
        instructions=GENERATION_INTENT,
        context={
            "scene": scene.model_dump(mode="json"),
            "shot": shot.model_dump(mode="json"),
            "clip": clip.model_dump(mode="json"),
            "beats": [b.model_dump(mode="json") for b in related_beats],
            "character_ids": char_ids,
            "location_id": location,
            "character_context": assets.character_context(char_ids),
            "location_context": assets.location_context(location),
        },
        response_model=GenerationIntent,
        version="2",
    )
