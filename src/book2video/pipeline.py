from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from book2video.assembly import FFmpegAssembler
from book2video.assets import AssetRegistry
from book2video.cache import StageCache
from book2video.config import ProjectConfig
from book2video.hardware import detect_nvidia_gpus, plan_video_devices
from book2video.io_utils import write_json
from book2video.knowledge import LocalKnowledgeIndex
from book2video.providers.base import LLMProvider, VideoProvider
from book2video.review import build_review_html
from book2video.schemas import PipelineSummary, SceneRecord, Shot
from book2video.stats import InstrumentedLLMProvider, InstrumentedVideoProvider, StatsRecorder
from book2video.stages.clips import partition_shots
from book2video.stages.generation import GenerationOrchestrator
from book2video.stages.planning import (
    build_beats,
    compile_generation_intent,
    critique_scenes,
    detect_scenes,
    materialize_scenes,
    plan_shots,
    time_beats,
)

# stage, message, rough overall fraction, live stats. A dictionary is used for the last field so the
# UI can grow without forcing every non-web caller to learn a new model class.
ProgressCallback = Callable[[str, str, float | None, dict[str, Any]], None]


class Book2VideoPipeline:
    def __init__(
        self,
        config: ProjectConfig,
        llm: LLMProvider,
        video: VideoProvider,
        *,
        progress_callback: ProgressCallback | None = None,
    ):
        self.config = config
        self.recorder = StatsRecorder()
        self.llm = InstrumentedLLMProvider(llm, self.recorder)
        self.video = InstrumentedVideoProvider(video, self.recorder)
        self.progress_callback = progress_callback
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache = StageCache(self.output_root / ".cache")
        self.assets = AssetRegistry.from_path(config.asset_registry_path)
        self.knowledge = LocalKnowledgeIndex(config.knowledge_paths)
        self._stats_path: Path | None = None
        self._llm_prepared = False
        self._video_prepared = False
        self._device_groups: list[tuple[str, ...]] = []

    def _phase(self, name: str) -> None:
        if self.recorder.stats.current_phase != name:
            self.recorder.switch_phase(name)

    def _progress(
        self,
        stage: str,
        message: str,
        fraction: float | None = None,
        *,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if completed is not None and total is not None:
            self.recorder.set_phase_progress(completed, total)
        snapshot = self.recorder.snapshot()
        if self._stats_path is not None:
            write_json(self._stats_path, snapshot)
        if self.progress_callback is not None:
            self.progress_callback(stage, message, fraction, snapshot)

    async def _activate_llm(self, reason: str) -> None:
        if self.config.generation.phase_swap_local_models and self._video_prepared:
            self._progress("lifecycle", "Releasing the video provider before returning to LLM work", None)
            await self.video.release()
            self._video_prepared = False
            self.recorder.lifecycle("video released")
        if not self._llm_prepared:
            self._progress("lifecycle", f"Preparing the LLM provider for {reason}", None)
            await self.llm.prepare()
            self._llm_prepared = True
            self.recorder.lifecycle(f"llm prepared: {reason}")

    async def _activate_video(self, reason: str) -> None:
        if self.config.generation.phase_swap_local_models and self._llm_prepared:
            self._progress("lifecycle", "Releasing the LLM provider before video generation", None)
            await self.llm.release()
            self._llm_prepared = False
            self.recorder.lifecycle("llm released")
        if not self._video_prepared:
            self._progress("lifecycle", f"Preparing the video provider for {reason}", None)
            await self.video.prepare(self._device_groups)
            self._video_prepared = True
            self.recorder.lifecycle(f"video prepared: {reason}")

    async def _release_all(self) -> None:
        # Cleanup is attempted independently so one provider's release failure does not prevent the
        # other local model from getting a chance to free its resources.
        if self._llm_prepared:
            try:
                await self.llm.release()
            finally:
                self._llm_prepared = False
                self.recorder.lifecycle("llm released at end")
        if self._video_prepared:
            try:
                await self.video.release()
            finally:
                self._video_prepared = False
                self.recorder.lifecycle("video released at end")

    def _resolve_device_groups(self) -> None:
        if self.config.generation.device_mode == "manual":
            groups = [tuple(x) for x in self.config.generation.worker_device_groups if x]
            if not groups:
                raise RuntimeError("Manual device mode needs at least one worker_device_groups entry.")
            explanation = "Using the manually configured video worker groups."
        else:
            snapshot = detect_nvidia_gpus()
            plan = plan_video_devices(
                gpus=snapshot.gpus,
                capabilities=self.video.capabilities,
                strategy=self.config.generation.auto_device_strategy,
                reserve_vram_gb=self.config.generation.reserve_vram_gb,
                min_gpu_memory_gb=self.config.generation.min_gpu_memory_gb,
            )
            groups = plan.groups
            explanation = plan.explanation
            if self._stats_path is not None:
                write_json(self._stats_path.with_name("hardware_start.json"), snapshot)
            if not groups:
                raise RuntimeError(f"Automatic GPU planning could not create a video worker: {explanation}")

        self._device_groups = groups
        # GenerationOrchestrator consumes the existing config field, so resolving auto mode into a
        # concrete runtime copy keeps the lower-level queue code simple.
        self.config.generation.worker_device_groups = [list(x) for x in groups]
        self.recorder.stats.device_groups = [list(x) for x in groups]
        self.recorder.stats.device_plan_explanation = explanation

    async def run(self, chapter: str, chapter_name: str = "chapter") -> PipelineSummary:
        if not chapter.strip():
            raise ValueError("Chapter text is empty. Supply a chapter path, inline text, browser upload, or config value.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / f"{Path(chapter_name).stem}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._stats_path = run_dir / "run_stats.json"
        (run_dir / "chapter.txt").write_text(chapter, encoding="utf-8")

        try:
            self._resolve_device_groups()
            await self._activate_llm("chapter planning")
            self._phase("scene planning")
            self._progress("scenes", "Finding candidate scene boundaries", 0.03)

            boundaries = await detect_scenes(self.llm, self.cache, chapter)
            self._progress("scenes", "Checking whether candidate scenes should be split further", 0.08)
            boundaries = await critique_scenes(self.llm, self.cache, chapter, boundaries)
            write_json(run_dir / "01_scenes.json", boundaries)

            scenes = materialize_scenes(chapter, boundaries, self.assets, self.knowledge)
            write_json(run_dir / "02_scene_records.json", [x.model_dump(mode="json") for x in scenes])

            self._phase("beat/timing/shot planning")
            beat_plans = []
            timing_plans = []
            shot_plans = []
            total_scenes = max(1, len(scenes))
            for index, scene in enumerate(scenes, start=1):
                base = 0.10 + ((index - 1) / total_scenes) * 0.32
                self._progress("planning", f"Scene {index}/{len(scenes)}: building beats", base, completed=index - 1, total=total_scenes)
                beats = await build_beats(self.llm, self.cache, scene, self.assets)
                self._progress("planning", f"Scene {index}/{len(scenes)}: estimating timing", base + 0.01)
                timing = await time_beats(self.llm, self.cache, scene, beats)
                self._progress("planning", f"Scene {index}/{len(scenes)}: planning shots", base + 0.02)
                shots = await plan_shots(self.llm, self.cache, scene, beats, timing, self.assets)
                beat_plans.append(beats)
                timing_plans.append(timing)
                shot_plans.append(shots)

            write_json(run_dir / "03_beats.json", [x.model_dump(mode="json") for x in beat_plans])
            write_json(run_dir / "04_timing.json", [x.model_dump(mode="json") for x in timing_plans])
            write_json(run_dir / "05_shots.json", [x.model_dump(mode="json") for x in shot_plans])

            clips = partition_shots(shot_plans, self.video.capabilities)
            write_json(run_dir / "06_clips.json", [x.model_dump(mode="json") for x in clips])

            scene_map: dict[str, SceneRecord] = {x.scene_id: x for x in scenes}
            beat_map = {x.scene_id: x for x in beat_plans}
            shot_plan_map = {x.scene_id: x for x in shot_plans}
            shots: dict[str, Shot] = {shot.shot_id: shot for plan in shot_plans for shot in plan.shots}

            self._phase("generation request compilation")
            intents = {}
            total_clips = max(1, len(clips))
            for index, clip in enumerate(clips, start=1):
                self._progress(
                    "compile",
                    f"Compiling generator request {index}/{len(clips)}",
                    0.43 + (index / total_clips) * 0.10,
                    completed=index - 1,
                    total=total_clips,
                )
                intents[clip.clip_id] = await compile_generation_intent(
                    self.llm,
                    self.cache,
                    scene_map[clip.scene_id],
                    beat_map[clip.scene_id],
                    shot_plan_map[clip.scene_id],
                    clip,
                    self.assets,
                )
            write_json(run_dir / "07_generation_intents.json", {k: v.model_dump(mode="json") for k, v in intents.items()})

            generator = GenerationOrchestrator(
                llm=self.llm,
                video=self.video,
                cache=self.cache,
                assets=self.assets,
                config=self.config.generation,
                thresholds=self.config.review,
                output_dir=run_dir / "takes",
                progress_callback=lambda message, completed, total: self._progress(
                    self.recorder.stats.current_phase,
                    message,
                    None,
                    completed=completed,
                    total=total,
                ),
            )

            # The initial video pass is deliberately kept together. This gives local providers a
            # useful boundary where an LLM can be released before the larger video model is loaded.
            self._phase("initial video generation")
            await self._activate_video("initial take generation")
            self._progress("generation", f"Generating initial takes for {len(clips)} clips", 0.55)
            all_takes = await generator.generate_initial(clips=clips, intents=intents)
            self.recorder.stats.generated_takes = sum(len(x) for x in all_takes.values())

            self._phase("initial video judging")
            await self._activate_llm("video judging")
            self._progress("judging", f"Judging {len(clips)} clips", 0.77)
            selected_map, rejected, _ = await generator.judge_all(
                clips=clips,
                intents=intents,
                shots=shots,
                all_takes=all_takes,
            )

            for round_number in range(1, self.config.generation.max_regeneration_rounds + 1):
                if not rejected:
                    break
                self._phase(f"video regeneration {round_number}")
                await self._activate_video(f"regeneration round {round_number}")
                self._progress(
                    "regeneration",
                    f"Regeneration round {round_number}: {len(rejected)} clips requested new takes",
                    min(0.90, 0.80 + round_number * 0.04),
                )
                added = await generator.generate_regeneration_round(
                    rejected=rejected,
                    intents=intents,
                    all_takes=all_takes,
                    selected=selected_map,
                    round_number=round_number,
                )
                self.recorder.stats.generated_takes += added
                self.recorder.stats.regeneration_takes += added

                self._phase(f"regeneration judging {round_number}")
                await self._activate_llm(f"judging regeneration round {round_number}")
                clips_to_rejudge = list(rejected)
                selected_map, rejected, _ = await generator.judge_all(
                    clips=clips_to_rejudge,
                    intents=intents,
                    shots=shots,
                    all_takes=all_takes,
                    existing_selected=selected_map,
                )

            selected = [selected_map[c.clip_id] for c in clips]
            write_json(run_dir / "08_selected.json", [x.model_dump(mode="json") for x in selected])
            write_json(
                run_dir / "09_all_takes.json",
                {k: [x.model_dump(mode="json") for x in values] for k, values in all_takes.items()},
            )

            self._phase("assembly")
            self._progress("assembly", "Building the first rough cut", 0.96)
            assembler = FFmpegAssembler()
            rough_cut = assembler.assemble(selected, run_dir)
            review = build_review_html(run_dir, selected, all_takes)

            await self._release_all()
            stats = self.recorder.finish()
            write_json(self._stats_path, stats)
            summary = PipelineSummary(
                run_dir=str(run_dir),
                scenes=len(scenes),
                shots=sum(len(x.shots) for x in shot_plans),
                clips=len(clips),
                generated_takes=sum(len(x) for x in all_takes.values()),
                selected_clips=len(selected),
                rough_cut_path=rough_cut,
                review_path=review,
                stats_path=str(self._stats_path),
                elapsed_seconds=stats["elapsed_seconds"],
                device_groups=[list(x) for x in self._device_groups],
            )
            write_json(run_dir / "summary.json", summary)
            self._progress("done", "Chapter first pass is ready", 1.0)
            return summary
        finally:
            # If a model call raises, freeing a local model is more useful than preserving the loaded
            # state. Release hooks are no-ops for providers that do not own resident resources.
            try:
                await self._release_all()
            finally:
                if self._stats_path is not None and self.recorder.stats.finished_at is None:
                    write_json(self._stats_path, self.recorder.snapshot())
