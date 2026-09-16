from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from pathlib import Path
from typing import Callable

from book2video.assets import AssetRegistry
from book2video.cache import StageCache
from book2video.config import GenerationConfig, ReviewThresholds
from book2video.media import extract_last_frame
from book2video.providers.base import LLMProvider, VideoGenerationRequest, VideoProvider
from book2video.schemas import (
    ClipJudgement,
    ClipSpec,
    GeneratedTake,
    GenerationIntent,
    SelectedClip,
    Shot,
)
from book2video.stages.judging import judge_takes

# message, completed, total. Keeping this small makes it usable by the browser, CLI, or a future MCP
# surface without coupling generation code to any one UI.
PhaseProgressCallback = Callable[[str, int | None, int | None], None]


class GenerationOrchestrator:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        video: VideoProvider,
        cache: StageCache,
        assets: AssetRegistry,
        config: GenerationConfig,
        thresholds: ReviewThresholds,
        output_dir: str | Path,
        progress_callback: PhaseProgressCallback | None = None,
    ) -> None:
        self.llm = llm
        self.video = video
        self.cache = cache
        self.assets = assets
        self.config = config
        self.thresholds = thresholds
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

    def _progress(self, message: str, completed: int | None = None, total: int | None = None) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message, completed, total)

    def _needs_regen(self, judgement: ClipJudgement) -> bool:
        if judgement.regenerate:
            return True
        chosen = next((a for a in judgement.assessments if a.take_number == judgement.recommended_take), None)
        if chosen is None:
            return False
        return (
            chosen.identity_score < self.thresholds.min_identity
            or chosen.continuity_score < self.thresholds.min_continuity
            or chosen.story_fidelity_score < self.thresholds.min_story_fidelity
            or chosen.visual_quality_score < self.thresholds.min_visual_quality
            or bool(chosen.critical_failures)
        )

    async def _generate_take(
        self,
        *,
        clip: ClipSpec,
        intent: GenerationIntent,
        take_number: int,
        devices: tuple[str, ...],
        continuation_path: str | None,
        continuation_frame: str | None,
        round_number: int,
    ) -> GeneratedTake:
        take_dir = self.output_dir / clip.clip_id
        take_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"r{round_number}_t{take_number:02d}.mp4"
        output_path = take_dir / suffix
        seed_material = f"{clip.clip_id}|{round_number}|{take_number}".encode("utf-8")
        stable = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        seed = self.config.seed_base + (stable % 2_000_000_000)

        request = VideoGenerationRequest(
            clip_id=clip.clip_id,
            take_number=take_number,
            duration_sec=clip.target_duration_sec,
            intent=intent,
            character_reference_images=self.assets.character_images(intent.character_ids),
            location_reference_images=self.assets.location_images(intent.location_id),
            voice_reference_files=self.assets.voice_refs(intent.character_ids),
            continuation_video_path=continuation_path,
            continuation_frame_path=continuation_frame,
            seed=seed,
            output_path=str(output_path),
        )
        started = time.perf_counter()
        result = await self.video.generate(request, devices)
        elapsed = time.perf_counter() - started
        return GeneratedTake(
            clip_id=clip.clip_id,
            take_number=take_number,
            output_path=result.output_path,
            seed=result.seed if result.seed is not None else seed,
            provider_id=self.video.provider_id,
            generation_seconds=result.generation_seconds if result.generation_seconds is not None else elapsed,
            metadata=result.metadata,
        )

    async def _generate_batch(
        self,
        jobs: list[tuple[ClipSpec, GenerationIntent, int, str | None, str | None, int]],
        *,
        progress_prefix: str,
    ) -> dict[str, list[GeneratedTake]]:
        groups = [tuple(group) for group in self.config.worker_device_groups]
        if not groups:
            raise RuntimeError("No video worker device groups are available. Check automatic GPU detection or manual groups.")

        queue: asyncio.Queue[
            tuple[ClipSpec, GenerationIntent, int, str | None, str | None, int] | None
        ] = asyncio.Queue()
        for job in jobs:
            await queue.put(job)
        results: dict[str, list[GeneratedTake]] = {}
        lock = asyncio.Lock()
        completed = 0
        total_jobs = len(jobs)
        self._progress(f"{progress_prefix}: 0/{total_jobs} takes", 0, total_jobs)

        async def worker(devices: tuple[str, ...]) -> None:
            nonlocal completed
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return
                clip, intent, take_number, continuation_path, continuation_frame, round_number = job
                try:
                    take = await self._generate_take(
                        clip=clip,
                        intent=intent,
                        take_number=take_number,
                        devices=devices,
                        continuation_path=continuation_path,
                        continuation_frame=continuation_frame,
                        round_number=round_number,
                    )
                    async with lock:
                        results.setdefault(clip.clip_id, []).append(take)
                        completed += 1
                        self._progress(
                            f"{progress_prefix}: {completed}/{total_jobs} takes",
                            completed,
                            total_jobs,
                        )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(group)) for group in groups]
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        for takes in results.values():
            takes.sort(key=lambda x: x.take_number)
        return results

    async def generate_initial(
        self,
        *,
        clips: list[ClipSpec],
        intents: dict[str, GenerationIntent],
    ) -> dict[str, list[GeneratedTake]]:
        """Generate all initial takes without loading the judge between clips.

        For split-shot continuations, take N follows take N from the preceding segment. This is being
        tried so a continuation chain can remain internally consistent before an LLM has selected a
        winner, which lets the expensive video phase stay batched.
        """

        pending = deque(clips)
        all_takes: dict[str, list[GeneratedTake]] = {}
        layer = 0
        while pending:
            ready: list[ClipSpec] = []
            waiting: deque[ClipSpec] = deque()
            while pending:
                clip = pending.popleft()
                if clip.continuation_from_clip_id and clip.continuation_from_clip_id not in all_takes:
                    waiting.append(clip)
                else:
                    ready.append(clip)
            pending = waiting
            if not ready:
                unresolved = [x.clip_id for x in pending]
                raise RuntimeError(f"No initial generation jobs are ready; continuation dependency cycle? {unresolved}")

            layer += 1
            jobs: list[tuple[ClipSpec, GenerationIntent, int, str | None, str | None, int]] = []
            for clip in ready:
                predecessor_takes = all_takes.get(clip.continuation_from_clip_id or "", [])
                for take_number in range(1, self.config.takes_per_clip + 1):
                    prev_path = None
                    prev_frame = None
                    if predecessor_takes:
                        source = next((x for x in predecessor_takes if x.take_number == take_number), predecessor_takes[0])
                        prev_path = source.output_path
                        prev_frame = extract_last_frame(
                            prev_path,
                            self.output_dir / clip.clip_id / f"continuation_initial_t{take_number:02d}.png",
                        )
                    jobs.append((clip, intents[clip.clip_id], take_number, prev_path, prev_frame, 0))

            batch = await self._generate_batch(jobs, progress_prefix=f"Initial video layer {layer}")
            for clip_id, takes in batch.items():
                all_takes.setdefault(clip_id, []).extend(takes)
                all_takes[clip_id].sort(key=lambda x: x.take_number)
        return all_takes

    async def judge_all(
        self,
        *,
        clips: list[ClipSpec],
        intents: dict[str, GenerationIntent],
        shots: dict[str, Shot],
        all_takes: dict[str, list[GeneratedTake]],
        existing_selected: dict[str, SelectedClip] | None = None,
    ) -> tuple[dict[str, SelectedClip], list[ClipSpec], dict[str, ClipJudgement]]:
        selected = dict(existing_selected or {})
        rejected: list[ClipSpec] = []
        judgements: dict[str, ClipJudgement] = {}
        total = len(clips)
        for index, clip in enumerate(clips, start=1):
            takes = all_takes.get(clip.clip_id, [])
            if not takes:
                raise RuntimeError(f"No takes exist for {clip.clip_id}")
            self._progress(f"Judging clip {index}/{total}", index - 1, total)
            judgement = await judge_takes(
                llm=self.llm,
                cache=self.cache,
                clip=clip,
                shot=shots[clip.shot_id],
                intent=intents[clip.clip_id],
                takes=takes,
                assets=self.assets,
            )
            judgements[clip.clip_id] = judgement
            chosen = next((x for x in takes if x.take_number == judgement.recommended_take), None) or takes[0]
            selected[clip.clip_id] = SelectedClip(
                clip_id=clip.clip_id,
                scene_id=clip.scene_id,
                shot_id=clip.shot_id,
                selected_take=chosen.take_number,
                selected_path=chosen.output_path,
                transition_after=clip.transition_after,
                judgement=judgement,
            )
            if self._needs_regen(judgement):
                rejected.append(clip)
        self._progress(f"Judged {total}/{total} clips; {len(rejected)} requested another generation round", total, total)
        return selected, rejected, judgements

    async def generate_regeneration_round(
        self,
        *,
        rejected: list[ClipSpec],
        intents: dict[str, GenerationIntent],
        all_takes: dict[str, list[GeneratedTake]],
        selected: dict[str, SelectedClip],
        round_number: int,
    ) -> int:
        jobs: list[tuple[ClipSpec, GenerationIntent, int, str | None, str | None, int]] = []
        for clip in rejected:
            current = all_takes.get(clip.clip_id, [])
            next_take_base = max((x.take_number for x in current), default=0)
            prev_path = None
            prev_frame = None
            if clip.continuation_from_clip_id and clip.continuation_from_clip_id in selected:
                predecessor = selected[clip.continuation_from_clip_id]
                prev_path = predecessor.selected_path
                prev_frame = extract_last_frame(
                    prev_path,
                    self.output_dir / clip.clip_id / f"continuation_regen_r{round_number}.png",
                )
            for offset in range(1, self.config.takes_per_clip + 1):
                jobs.append(
                    (
                        clip,
                        intents[clip.clip_id],
                        next_take_base + offset,
                        prev_path,
                        prev_frame,
                        round_number,
                    )
                )
        if not jobs:
            return 0
        batch = await self._generate_batch(jobs, progress_prefix=f"Regeneration round {round_number}")
        added = 0
        for clip_id, takes in batch.items():
            all_takes.setdefault(clip_id, []).extend(takes)
            all_takes[clip_id].sort(key=lambda x: x.take_number)
            added += len(takes)
        return added
