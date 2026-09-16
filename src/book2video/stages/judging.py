from __future__ import annotations

from book2video.assets import AssetRegistry
from book2video.cache import StageCache
from book2video.prompts import VIDEO_JUDGE
from book2video.providers.base import LLMProvider, MediaAttachment, StructuredLLMRequest
from book2video.schemas import ClipJudgement, ClipSpec, GeneratedTake, GenerationIntent, Shot


async def judge_takes(
    *,
    llm: LLMProvider,
    cache: StageCache,
    clip: ClipSpec,
    shot: Shot,
    intent: GenerationIntent,
    takes: list[GeneratedTake],
    assets: AssetRegistry,
) -> ClipJudgement:
    context = {
        "clip": clip.model_dump(mode="json"),
        "shot": shot.model_dump(mode="json"),
        "intent": intent.model_dump(mode="json"),
        "character_context": assets.character_context(intent.character_ids),
        "location_context": assets.location_context(intent.location_id),
        "takes": [x.model_dump(mode="json") for x in takes],
    }
    key = cache.key(
        "video_judge",
        {"provider": llm.provider_id, "context": context, "prompt": VIDEO_JUDGE},
        version="2",
    )
    cached = cache.get("video_judge", key, ClipJudgement)
    if cached:
        return cached  # type: ignore[return-value]

    attachments = [MediaAttachment(path=t.output_path, kind="video", label=f"take {t.take_number}") for t in takes]
    response = await llm.generate_structured(
        StructuredLLMRequest(
            stage="video_judge",
            instructions=VIDEO_JUDGE,
            context=context,
            attachments=attachments,
        ),
        ClipJudgement,
    )
    cache.put("video_judge", key, response)
    return response
