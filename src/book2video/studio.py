from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from book2video.assembly import FFmpegAssembler
from book2video.config import GenerationConfig, ProjectConfig, ProviderConfig, ReviewThresholds, load_config
from book2video.io_utils import read_json, read_text_file, write_json
from book2video.hardware import detect_nvidia_gpus
from book2video.pipeline import Book2VideoPipeline
from book2video.provider_loader import build_provider
from book2video.review import load_selected_and_takes, save_selection_override
from book2video.viewer import ViewerState


class StudioPreferences(BaseModel):
    # These values live beside config.toml rather than rewriting it. That is being tried so the
    # commented TOML can remain useful documentation while the browser still remembers choices.
    active_llm_profile: str | None = None
    active_video_profile: str | None = None
    chapter_source: Literal["config", "path", "text", "upload"] = "config"
    chapter_path: str | None = None
    output_root: str | None = None
    asset_registry_path: str | None = None
    knowledge_paths: list[str] = Field(default_factory=list)
    generation: GenerationConfig | None = None
    review: ReviewThresholds | None = None
    auto_assemble_on_selection: bool | None = None
    viewer_preload_adjacent: bool | None = None
    custom_llm_profiles: dict[str, ProviderConfig] = Field(default_factory=dict)
    custom_video_profiles: dict[str, ProviderConfig] = Field(default_factory=dict)


class RunRequest(BaseModel):
    chapter_source: Literal["config", "path", "text", "upload"] = "config"
    chapter_path: str | None = None
    chapter_text: str = ""
    chapter_name: str | None = None
    llm_profile: str | None = None
    video_profile: str | None = None
    output_root: str | None = None
    asset_registry_path: str | None = None
    knowledge_paths: list[str] = Field(default_factory=list)
    generation: GenerationConfig
    review: ReviewThresholds
    auto_assemble_on_selection: bool = False
    viewer_preload_adjacent: bool = True


class ProfileSaveRequest(BaseModel):
    kind: Literal["llm", "video"]
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    class_path: str = Field(min_length=3)
    label: str | None = None
    description: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class TakeSelectionRequest(BaseModel):
    run: str
    clip_id: str
    take_number: int = Field(ge=1)


class RunNameRequest(BaseModel):
    run: str


class JobState:
    def __init__(self) -> None:
        self.status = "idle"
        self.stage = "idle"
        self.message = "No run is active"
        self.progress: float | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.summary: dict[str, Any] | None = None
        self.error: str | None = None
        self.stats: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary,
            "error": self.error,
            "stats": self.stats,
        }


class StudioState:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        self.preferences_path = self.config_path.with_name(self.config_path.stem + ".studio.json")
        self.preferences = self._load_preferences()
        self.job = JobState()
        self.task: asyncio.Task[Any] | None = None
        self._hardware_cache: dict[str, Any] | None = None
        self._hardware_cache_at = 0.0

    def _load_preferences(self) -> StudioPreferences:
        if self.preferences_path.exists():
            try:
                return StudioPreferences.model_validate(read_json(self.preferences_path))
            except Exception:
                # A broken preference overlay should not make config.toml unusable. The UI can save
                # a fresh overlay after startup if this file was edited by hand incorrectly.
                pass
        return StudioPreferences()

    def save_preferences(self, prefs: StudioPreferences) -> StudioPreferences:
        self.preferences = prefs
        write_json(self.preferences_path, prefs)
        return prefs

    def profiles(self, kind: Literal["llm", "video"]) -> dict[str, ProviderConfig]:
        if kind == "llm":
            base = dict(self.config.available_llm_profiles())
            base.update(self.preferences.custom_llm_profiles)
            return base
        base = dict(self.config.available_video_profiles())
        base.update(self.preferences.custom_video_profiles)
        return base

    def save_profile(self, request: ProfileSaveRequest) -> None:
        profile = ProviderConfig(
            class_path=request.class_path,
            options=request.options,
            label=request.label,
            description=request.description,
        )
        prefs = self.preferences.model_copy(deep=True)
        target = prefs.custom_llm_profiles if request.kind == "llm" else prefs.custom_video_profiles
        target[request.name] = profile
        self.save_preferences(prefs)

    def delete_custom_profile(self, kind: Literal["llm", "video"], name: str) -> None:
        prefs = self.preferences.model_copy(deep=True)
        target = prefs.custom_llm_profiles if kind == "llm" else prefs.custom_video_profiles
        if name not in target:
            raise KeyError(f"{name!r} is not a browser-created {kind} profile.")
        del target[name]
        self.save_preferences(prefs)

    def hardware_snapshot(self, max_age_sec: float = 2.0) -> dict[str, Any]:
        now = time.time()
        if self._hardware_cache is None or (now - self._hardware_cache_at) >= max_age_sec:
            self._hardware_cache = detect_nvidia_gpus().model_dump(mode="json")
            self._hardware_cache_at = now
        return self._hardware_cache

    def list_runs(self) -> list[dict[str, Any]]:
        root = Path(self.preferences.output_root or self.config.output_root).expanduser()
        if not root.is_absolute():
            root = (self.config_path.parent / root).resolve()
        if not root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for run in root.iterdir():
            if not run.is_dir() or not (run / "08_selected.json").exists():
                continue
            summary = read_json(run / "summary.json") if (run / "summary.json").exists() else {}
            rows.append(
                {
                    "name": run.name,
                    "path": str(run.resolve()),
                    "modified": run.stat().st_mtime,
                    "summary": summary,
                }
            )
        rows.sort(key=lambda x: x["modified"], reverse=True)
        return rows

    def resolve_run(self, run_name: str) -> Path:
        matches = {row["name"]: Path(row["path"]) for row in self.list_runs()}
        if run_name not in matches:
            raise KeyError(f"Unknown completed run: {run_name}")
        return matches[run_name]

    def bootstrap(self) -> dict[str, Any]:
        llm_profiles = self.profiles("llm")
        video_profiles = self.profiles("video")
        prefs = self.preferences
        config = self.config
        return {
            "config_path": str(self.config_path),
            "preferences_path": str(self.preferences_path),
            "chapter": {
                "config_has_path": bool(config.chapter_path),
                "config_path": config.chapter_path,
                "config_has_text": bool(config.chapter_text.strip()),
            },
            "llm_profiles": {
                name: {**profile.model_dump(mode="json"), "custom": name in prefs.custom_llm_profiles}
                for name, profile in llm_profiles.items()
            },
            "video_profiles": {
                name: {**profile.model_dump(mode="json"), "custom": name in prefs.custom_video_profiles}
                for name, profile in video_profiles.items()
            },
            "defaults": {
                "llm_profile": prefs.active_llm_profile or config.active_llm_profile or (next(iter(llm_profiles), None)),
                "video_profile": prefs.active_video_profile or config.active_video_profile or (next(iter(video_profiles), None)),
                "chapter_source": prefs.chapter_source,
                "chapter_path": prefs.chapter_path or config.chapter_path or "",
                "output_root": prefs.output_root or config.output_root,
                "asset_registry_path": prefs.asset_registry_path if prefs.asset_registry_path is not None else (config.asset_registry_path or ""),
                "knowledge_paths": prefs.knowledge_paths or config.knowledge_paths,
                "generation": (prefs.generation or config.generation).model_dump(mode="json"),
                "review": (prefs.review or config.review).model_dump(mode="json"),
                "auto_assemble_on_selection": (
                    prefs.auto_assemble_on_selection
                    if prefs.auto_assemble_on_selection is not None
                    else config.defaults.auto_assemble_on_selection
                ),
                "viewer_preload_adjacent": (
                    prefs.viewer_preload_adjacent
                    if prefs.viewer_preload_adjacent is not None
                    else config.defaults.viewer_preload_adjacent
                ),
            },
            "runs": self.list_runs(),
            "job": self.job.snapshot(),
            "hardware": self.hardware_snapshot(),
        }

    def _resolve_chapter(self, request: RunRequest) -> tuple[str, str]:
        if request.chapter_source == "path":
            if not request.chapter_path:
                raise ValueError("Chapter source 'path' needs a path.")
            path = Path(request.chapter_path).expanduser()
            return read_text_file(path), request.chapter_name or path.stem
        if request.chapter_source in {"text", "upload"}:
            if not request.chapter_text.strip():
                raise ValueError(f"Chapter source {request.chapter_source!r} needs chapter text.")
            return request.chapter_text, request.chapter_name or ("uploaded_chapter" if request.chapter_source == "upload" else "inline_chapter")

        # The config source intentionally ignores the browser's path/text fields. This makes it a
        # predictable way to say "use what config.toml says" after experimenting in the UI.
        if self.config.chapter_path and str(self.config.chapter_path).strip():
            path = Path(self.config.chapter_path).expanduser()
            if not path.is_absolute():
                path = (self.config_path.parent / path).resolve()
            return read_text_file(path), path.stem
        if self.config.chapter_text.strip():
            return self.config.chapter_text, "chapter"
        raise ValueError("config.toml does not currently contain a chapter_path or chapter_text.")

    async def start_run(self, request: RunRequest) -> dict[str, Any]:
        if self.task is not None and not self.task.done():
            raise RuntimeError("A Book2Video run is already active in this studio process.")

        llm_profiles = self.profiles("llm")
        video_profiles = self.profiles("video")
        llm_name = request.llm_profile or self.config.active_llm_profile or next(iter(llm_profiles), None)
        video_name = request.video_profile or self.config.active_video_profile or next(iter(video_profiles), None)
        if not llm_name or llm_name not in llm_profiles:
            raise ValueError(f"Unknown LLM profile: {llm_name!r}")
        if not video_name or video_name not in video_profiles:
            raise ValueError(f"Unknown video profile: {video_name!r}")

        chapter, chapter_name = self._resolve_chapter(request)
        runtime_config = self.config.model_copy(deep=True)
        runtime_config.output_root = request.output_root or self.config.output_root
        runtime_config.asset_registry_path = request.asset_registry_path or None
        runtime_config.knowledge_paths = request.knowledge_paths
        runtime_config.generation = request.generation
        runtime_config.review = request.review

        llm_profile = llm_profiles[llm_name]
        video_profile = video_profiles[video_name]

        # Remembering the form settings separately from config.toml lets the next browser session
        # reopen where the user left off without flattening the comments in the TOML file.
        prefs = self.preferences.model_copy(deep=True)
        prefs.active_llm_profile = llm_name
        prefs.active_video_profile = video_name
        prefs.chapter_source = request.chapter_source
        prefs.chapter_path = request.chapter_path
        prefs.output_root = runtime_config.output_root
        prefs.asset_registry_path = request.asset_registry_path
        prefs.knowledge_paths = request.knowledge_paths
        prefs.generation = request.generation
        prefs.review = request.review
        prefs.auto_assemble_on_selection = request.auto_assemble_on_selection
        prefs.viewer_preload_adjacent = request.viewer_preload_adjacent
        self.save_preferences(prefs)

        self.job = JobState()
        self.job.status = "running"
        self.job.stage = "starting"
        self.job.message = f"Starting with LLM '{llm_name}' and video '{video_name}'"
        self.job.progress = 0.0
        self.job.started_at = time.time()

        async def runner() -> None:
            try:
                llm = build_provider(llm_profile.class_path, llm_profile.options)
                video = build_provider(video_profile.class_path, video_profile.options)

                def progress(stage: str, message: str, fraction: float | None, stats: dict[str, Any]) -> None:
                    self.job.stage = stage
                    self.job.message = message
                    self.job.stats = stats
                    if fraction is not None:
                        self.job.progress = fraction

                pipeline = Book2VideoPipeline(runtime_config, llm, video, progress_callback=progress)
                summary = await pipeline.run(chapter, chapter_name)
                self.job.status = "completed"
                self.job.stage = "done"
                self.job.message = "First pass completed"
                self.job.progress = 1.0
                self.job.summary = summary.model_dump(mode="json")
            except Exception as exc:
                self.job.status = "failed"
                self.job.stage = "failed"
                self.job.message = str(exc)
                self.job.error = repr(exc)
            finally:
                self.job.finished_at = time.time()

        self.task = asyncio.create_task(runner())
        return self.job.snapshot()


class StudioViewerState(ViewerState):
    def __init__(self, run_dir: str | Path, run_name: str, auto_assemble_on_selection: bool = False) -> None:
        super().__init__(run_dir, auto_assemble_on_selection=auto_assemble_on_selection)
        self.media_prefix = f"/api/media/{quote(run_name)}"

    def _media_url(self, path: str) -> str:
        source = Path(path).resolve()
        try:
            rel = source.relative_to(self.run_dir)
        except ValueError as exc:
            raise RuntimeError(f"Generated media {source} is outside run directory {self.run_dir}.") from exc
        return self.media_prefix + "/" + "/".join(quote(part) for part in rel.parts)


def _profile_public(profile: ProviderConfig, custom: bool) -> dict[str, Any]:
    data = profile.model_dump(mode="json")
    data["custom"] = custom
    return data


def create_studio_app(config_path: str | Path = "config.toml") -> FastAPI:
    state = StudioState(config_path)
    app = FastAPI(title="Book2Video Studio")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return STUDIO_HTML

    @app.get("/api/bootstrap")
    async def api_bootstrap() -> dict[str, Any]:
        return state.bootstrap()

    @app.get("/api/job")
    async def api_job() -> dict[str, Any]:
        return state.job.snapshot() | {"hardware": state.hardware_snapshot()}

    @app.get("/api/hardware")
    async def api_hardware() -> dict[str, Any]:
        return state.hardware_snapshot(max_age_sec=1.0)

    @app.post("/api/run")
    async def api_run(request: RunRequest) -> dict[str, Any]:
        try:
            return await state.start_run(request)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preferences")
    async def api_preferences(prefs: StudioPreferences) -> dict[str, Any]:
        state.save_preferences(prefs)
        return state.bootstrap()

    @app.post("/api/profile")
    async def api_profile(request: ProfileSaveRequest) -> dict[str, Any]:
        state.save_profile(request)
        return state.bootstrap()

    @app.delete("/api/profile/{kind}/{name}")
    async def api_delete_profile(kind: Literal["llm", "video"], name: str) -> dict[str, Any]:
        try:
            state.delete_custom_profile(kind, name)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return state.bootstrap()

    @app.get("/api/runs")
    async def api_runs() -> list[dict[str, Any]]:
        return state.list_runs()

    @app.get("/api/review/state")
    async def api_review_state(run: str) -> dict[str, Any]:
        try:
            run_dir = state.resolve_run(run)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        viewer = StudioViewerState(
            run_dir,
            run,
            auto_assemble_on_selection=(
                state.preferences.auto_assemble_on_selection
                if state.preferences.auto_assemble_on_selection is not None
                else state.config.defaults.auto_assemble_on_selection
            ),
        )
        return viewer.snapshot()

    @app.post("/api/review/select")
    async def api_review_select(selection: TakeSelectionRequest) -> dict[str, Any]:
        try:
            run_dir = state.resolve_run(selection.run)
            viewer = StudioViewerState(
                run_dir,
                selection.run,
                auto_assemble_on_selection=(
                    state.preferences.auto_assemble_on_selection
                    if state.preferences.auto_assemble_on_selection is not None
                    else state.config.defaults.auto_assemble_on_selection
                ),
            )
            return viewer.select(selection.clip_id, selection.take_number)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/review/assemble")
    async def api_review_assemble(request: RunNameRequest) -> dict[str, Any]:
        try:
            run_dir = state.resolve_run(request.run)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        selected, _ = load_selected_and_takes(run_dir)
        return {"rough_cut": FFmpegAssembler().assemble(selected, run_dir)}

    @app.get("/api/media/{run_name}/{media_path:path}")
    async def api_media(run_name: str, media_path: str) -> FileResponse:
        try:
            run_dir = state.resolve_run(run_name).resolve()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        target = (run_dir / media_path).resolve()
        try:
            target.relative_to(run_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Media path escapes the run directory.") from exc
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Media file not found.")
        return FileResponse(target)

    return app


def serve_studio(
    config_path: str | Path = "config.toml",
    *,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool | None = None,
) -> None:
    import uvicorn

    config = load_config(config_path)
    host = host or config.defaults.viewer_host
    port = port or config.defaults.viewer_port
    if open_browser is None:
        open_browser = config.defaults.open_browser
    app = create_studio_app(config_path)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


STUDIO_HTML = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Book2Video Studio</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;--bg:#0f1012;--panel:#17191d;--panel2:#1e2126;--border:#333842;--muted:#9ba3af;--text:#eef1f5;--accent:#d7dbe2;--danger:#ef8a8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}button,input,select,textarea{font:inherit}button{border:1px solid #4a505c;background:#292d34;color:var(--text);padding:.5rem .75rem;border-radius:7px;cursor:pointer}button:hover{background:#353a43}button.primary{background:#e7e9ed;color:#111;border-color:#e7e9ed;font-weight:700}button.danger{border-color:#825050;color:#ffc7c7}.top{height:58px;display:flex;align-items:center;padding:0 1rem;border-bottom:1px solid var(--border);background:#14161a;position:sticky;top:0;z-index:20}.brand{font-weight:800;margin-right:1.5rem}.nav{display:flex;gap:.4rem}.nav button.active{background:#eceef2;color:#111}.right{margin-left:auto;color:var(--muted);font-size:.82rem}.page{display:none;padding:1rem;max-width:1500px;margin:0 auto}.page.active{display:block}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:1rem}.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:1rem}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}h2,h3{margin:.1rem 0 .8rem}.field{margin:.7rem 0}.field label{display:flex;align-items:center;gap:.35rem;font-size:.88rem;font-weight:650;margin-bottom:.3rem}.hint{display:inline-grid;place-items:center;width:17px;height:17px;border:1px solid #646b78;border-radius:50%;font-size:.72rem;color:#c5cad2;cursor:help}input[type=text],input[type=number],select,textarea{width:100%;background:#101216;color:var(--text);border:1px solid #3b414b;border-radius:6px;padding:.5rem}textarea{min-height:110px;resize:vertical;font-family:ui-monospace,SFMono-Regular,monospace}.row{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.7rem}.muted{color:var(--muted);font-size:.82rem}.profileDesc{min-height:2.8rem;padding:.45rem .6rem;background:#111318;border-radius:6px;color:#b6bdc7;font-size:.82rem;margin-top:.45rem}.actions{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.statusbox{background:#101216;border:1px solid #313640;border-radius:8px;padding:.75rem}.bar{height:10px;background:#252932;border-radius:10px;overflow:hidden;margin-top:.5rem}.bar>div{height:100%;background:#d9dde4;width:0}.error{color:#ffaaaa}.runRow{display:flex;align-items:center;gap:.8rem;padding:.7rem;border-bottom:1px solid #2d3139}.runRow .grow{flex:1}.reviewLayout{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:calc(100vh - 116px);border:1px solid var(--border);border-radius:10px;overflow:hidden}.playerCol{display:flex;flex-direction:column;min-width:0;background:#0a0a0a}.playerWrap{position:relative;flex:1;min-height:0}.videoSlot{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;background:#000}.videoSlot.active{display:block}.transport{background:#181a1f;border-top:1px solid var(--border);padding:.7rem}.sideClips{overflow:auto;background:#15171b;border-left:1px solid var(--border)}.clip{padding:.7rem;border-bottom:1px solid #2b2f36;cursor:pointer}.clip.current{background:#242831;outline:2px solid #777f8d;outline-offset:-2px}.clipHead{display:flex;justify-content:space-between;gap:.4rem;font-size:.85rem}.takeButtons{display:flex;gap:.3rem;flex-wrap:wrap;margin-top:.45rem}.takeButtons button{padding:.25rem .45rem;font-size:.78rem}.takeButtons button.selected{background:#eceef2;color:#111}.hidden{display:none!important}details{border-top:1px solid #30343c;margin-top:1rem;padding-top:.8rem}summary{cursor:pointer;font-weight:700}.check{display:flex;align-items:center;gap:.5rem}.check input{width:auto}.statgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.55rem;margin-top:.7rem}.stat{background:#151820;border:1px solid #303642;border-radius:7px;padding:.55rem}.stat .v{font-size:1.05rem;font-weight:750}.gpuRow{display:grid;grid-template-columns:1.6fr .8fr .8fr .8fr .8fr;gap:.5rem;padding:.4rem .2rem;border-bottom:1px solid #2d323b;font-size:.82rem}.phasebar{height:7px;background:#252932;border-radius:10px;overflow:hidden;margin-top:.35rem}.phasebar>div{height:100%;background:#8e97a6;width:0}@media(max-width:800px){.statgrid{grid-template-columns:1fr 1fr}.gpuRow{grid-template-columns:1.4fr .8fr .8fr}}@media(max-width:1000px){.span4,.span5,.span6,.span7,.span8{grid-column:span 12}.reviewLayout{grid-template-columns:1fr;grid-template-rows:65vh 35vh;height:auto}.sideClips{border-left:0;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<div class="top"><div class="brand">Book2Video Studio</div><div class="nav"><button data-page="setup" class="active">Setup</button><button data-page="review">Review</button><button data-page="runs">Runs</button><button data-page="profiles">Profiles</button></div><div class="right" id="configPath"></div></div>

<section id="page-setup" class="page active">
<div class="grid">
  <div class="card span6"><h3>Chapter</h3>
    <div class="field"><label>Chapter source <span class="hint" title="config uses chapter_path/chapter_text from config.toml. path reads a server-side filesystem path. text uses the text box. upload lets your browser choose a local file and sends its contents to the local studio server.">?</span></label><select id="chapterSource"><option value="config">Config TOML</option><option value="path">Filesystem path</option><option value="text">Paste / inline text</option><option value="upload">Browser file upload</option></select></div>
    <div id="chapterPathWrap" class="field hidden"><label>Chapter path <span class="hint" title="A path visible to the machine running Book2Video, for example /home/me/book/ch01.md.">?</span></label><input id="chapterPath" type="text"></div>
    <div id="chapterTextWrap" class="field hidden"><label>Chapter text <span class="hint" title="The chapter is sent as text directly into the pipeline. This is not saved into the studio preference file.">?</span></label><textarea id="chapterText"></textarea></div>
    <div id="chapterUploadWrap" class="field hidden"><label>Chapter file <span class="hint" title="The browser reads the selected file and sends its text to localhost when the run starts. The original file is not modified.">?</span></label><input id="chapterFile" type="file" accept=".md,.txt,text/plain,text/markdown"><div class="muted" id="uploadName"></div></div>
  </div>

  <div class="card span6"><h3>Models</h3>
    <div class="field"><label>LLM profile <span class="hint" title="Used for scene detection, beats, timing, shot planning, prompt compilation, and automated video judging. A vision-capable model is useful for judging clips.">?</span></label><select id="llmProfile"></select><div class="profileDesc" id="llmDesc"></div></div>
    <div class="field"><label>Video profile <span class="hint" title="The generator used for physical clips. Profiles can point at local command wrappers, ComfyUI, or online APIs.">?</span></label><select id="videoProfile"></select><div class="profileDesc" id="videoDesc"></div></div>
  </div>

  <div class="card span6"><h3>Project paths</h3>
    <div class="field"><label>Output root <span class="hint" title="Run folders, cache files, generated takes, review data, and rough cuts are written here.">?</span></label><input id="outputRoot" type="text"></div>
    <div class="field"><label>Asset registry <span class="hint" title="JSON registry mapping canonical character/location IDs to profile text, aliases, reference images, voice references, and related reusable assets.">?</span></label><input id="assetRegistry" type="text"></div>
    <div class="field"><label>Knowledge paths <span class="hint" title="One file path per line. These can contain visual style rules, world/location bibles, screenplay conventions, or other project context.">?</span></label><textarea id="knowledgePaths" style="min-height:80px"></textarea></div>
  </div>

  <div class="card span6"><h3>Generation</h3>
    <div class="row3"><div class="field"><label>Takes / clip <span class="hint" title="Initial candidate generations for every physical clip before the judge chooses one.">?</span></label><input id="takesPerClip" type="number" min="1"></div><div class="field"><label>Regen rounds <span class="hint" title="Extra batches allowed when the judge rejects the current candidates. Zero disables automatic regeneration.">?</span></label><input id="regenRounds" type="number" min="0"></div><div class="field"><label>Seed base <span class="hint" title="A stable base added to deterministic per-clip seed derivation where the provider supports seeds.">?</span></label><input id="seedBase" type="number"></div></div>
    <div class="row"><div class="field"><label>GPU device mode <span class="hint" title="Auto discovers the NVIDIA GPUs currently present each run. Manual uses the exact worker groups below.">?</span></label><select id="deviceMode"><option value="auto">Automatic discovery</option><option value="manual">Manual groups</option></select></div><div class="field"><label>Automatic strategy <span class="hint" title="one_per_gpu favors throughput when a model fits one card. all_gpus asks a multi-GPU-capable provider to use every detected card for one worker.">?</span></label><select id="autoStrategy"><option value="one_per_gpu">One worker per usable GPU</option><option value="all_gpus">All GPUs in one worker</option></select></div></div>
    <div class="row"><div class="field"><label>VRAM reserve / GPU (GB) <span class="hint" title="Planning headroom subtracted from each card's total VRAM when a video profile provides an estimated requirement. It does not reserve memory at the CUDA level.">?</span></label><input id="reserveVram" type="number" min="0" step="0.25"></div><div class="field"><label>Ignore GPUs smaller than (GB) <span class="hint" title="Optional filter for mixed systems. Zero allows every detected NVIDIA GPU to be considered.">?</span></label><input id="minGpuMemory" type="number" min="0" step="1"></div></div>
    <div id="workerGroupsWrap" class="field"><label>Manual worker device groups <span class="hint" title='JSON. [["cuda:0"],["cuda:1"]] means two independent workers. [["cuda:0","cuda:1"]] hands both cards to one provider call for distributed inference.'>?</span></label><input id="workerGroups" type="text"></div>
    <div class="check"><input id="phaseSwap" type="checkbox"><label for="phaseSwap">Batch LLM/video phases and release one before loading the other</label><span class="hint" title="Intended for local LLM + local video on the same GPUs. Planning/compilation runs together, then video takes run together, then judging runs together. Providers with lifecycle support can unload at those boundaries.">?</span></div>
    <div class="check"><input id="preloadAdjacent" type="checkbox"><label for="preloadAdjacent">Preload previous/current/next review clips</label><span class="hint" title="Keeps three video elements warm in the review page to reduce stalls between cuts. Browser buffering policy still applies.">?</span></div>
    <div class="check"><input id="autoAssemble" type="checkbox"><label for="autoAssemble">Rebuild rough_cut.mp4 after every take change</label><span class="hint" title="Usually slower than necessary. Leaving it off lets the live playlist change immediately and rebuilds the MP4 only when requested.">?</span></div>
  </div>

  <div class="card span6"><h3>Detected hardware <span class="hint" title="Live values come from nvidia-smi when available. Automatic planning re-detects GPUs when a run starts, so replacing or removing a card does not depend on a saved cuda count.">?</span></h3><div id="hardwareList" class="muted">Detecting GPUs…</div><div class="profileDesc" id="devicePlan">The actual device plan appears after a run starts.</div></div>

  <div class="card span12"><h3>Automated review thresholds</h3><div class="row3"><div class="field"><label>Identity <span class="hint" title="Below this score, the selected take is considered a candidate for regeneration.">?</span></label><input id="minIdentity" type="number" min="0" max="1" step=".01"></div><div class="field"><label>Continuity <span class="hint" title="How closely the take preserves the expected scene/shot state and continuation.">?</span></label><input id="minContinuity" type="number" min="0" max="1" step=".01"></div><div class="field"><label>Story fidelity <span class="hint" title="How closely required actions and events match the planned shot.">?</span></label><input id="minStory" type="number" min="0" max="1" step=".01"></div></div><div class="row3"><div class="field"><label>Visual quality <span class="hint" title="Artifact/quality floor. A rough first pass can intentionally use a lower value than a final render.">?</span></label><input id="minVisual" type="number" min="0" max="1" step=".01"></div></div></div>

  <div class="card span12"><div class="actions"><button class="primary" onclick="startRun()">Start chapter first pass</button><button onclick="saveDefaults()">Save these UI defaults</button><span class="muted">The browser defaults are saved beside config.toml; the commented TOML is left intact.</span></div><div class="statusbox" style="margin-top:.8rem"><div><strong id="jobStage">Idle</strong> — <span id="jobMessage">No run is active</span></div><div class="bar"><div id="jobBar"></div></div><div class="muted" id="jobDetail"></div><div class="phasebar"><div id="phaseBar"></div></div><div class="muted" id="phaseDetail"></div><div class="statgrid"><div class="stat"><div class="muted">LLM calls</div><div class="v" id="statLlm">0</div></div><div class="stat"><div class="muted">Video takes</div><div class="v" id="statVideo">0</div></div><div class="stat"><div class="muted">Requested footage</div><div class="v" id="statFootage">0s</div></div><div class="stat"><div class="muted">Video sec / output sec</div><div class="v" id="statRtf">—</div></div></div></div></div>
</div>
</section>

<section id="page-review" class="page">
<div class="actions" style="margin-bottom:.8rem"><label>Run</label><select id="reviewRun" style="max-width:520px"></select><button onclick="loadReview()">Load run</button><button onclick="rebuildCut()">Rebuild rough_cut.mp4</button><span class="muted" id="reviewStatus"></span></div>
<div class="reviewLayout">
  <div class="playerCol"><div class="playerWrap"><video id="rslot0" class="videoSlot" controls preload="auto" playsinline></video><video id="rslot1" class="videoSlot" controls preload="auto" playsinline></video><video id="rslot2" class="videoSlot" controls preload="auto" playsinline></video></div><div class="transport"><div id="reviewTitle"><strong>No run loaded</strong></div><div class="actions" style="margin-top:.5rem"><button onclick="previousClip()">◀ Previous</button><button onclick="nextClip()">Next ▶</button><span>Take:</span><span id="currentTakeButtons"></span></div></div></div>
  <div class="sideClips" id="reviewClips"></div>
</div>
</section>

<section id="page-runs" class="page"><div class="card"><h3>Completed runs</h3><div id="runsList"></div></div></section>

<section id="page-profiles" class="page"><div class="grid"><div class="card span5"><h3>Profiles</h3><div class="field"><label>Kind <span class="hint" title="A profile is a named provider configuration. Browser-created profiles overlay config.toml profiles without deleting them.">?</span></label><select id="profileKind"><option value="llm">LLM</option><option value="video">Video</option></select></div><div class="field"><label>Existing profile</label><select id="profileExisting"></select></div><div class="actions"><button onclick="loadProfileEditor()">Load into editor</button><button class="danger" onclick="deleteProfile()">Delete browser-created profile</button></div><div class="muted" style="margin-top:.8rem">Profiles from config.toml cannot be deleted here. Saving the same name creates a browser overlay, leaving the TOML untouched.</div></div><div class="card span7"><h3>Profile editor</h3><div class="row"><div class="field"><label>Name <span class="hint" title="Stable name shown in model dropdowns, such as wan22_fast or local_qwen.">?</span></label><input id="profileName" type="text"></div><div class="field"><label>Display label</label><input id="profileLabel" type="text"></div></div><div class="field"><label>Python class path <span class="hint" title="module.path:ClassName. Adding a genuinely new backend normally means adding one provider class and pointing a profile here.">?</span></label><input id="profileClass" type="text"></div><div class="field"><label>Description <span class="hint" title="Shown under the model selector so you can remember why this profile exists.">?</span></label><input id="profileDescription" type="text"></div><div class="field"><label>Provider options (JSON) <span class="hint" title="Passed as keyword arguments to the provider class. Prefer environment-variable names for API secrets; anything saved here is plain text in the local studio preference file.">?</span></label><textarea id="profileOptions" style="min-height:240px">{}</textarea></div><button class="primary" onclick="saveProfile()">Save profile overlay</button><span id="profileStatus" class="muted"></span></div></div></section>

<script>
let bootstrap=null, jobTimer=null, uploadedChapter={text:'',name:''};
let reviewState=null, currentIndex=0, activeSlot=null;
const rslots=['rslot0','rslot1','rslot2'].map(id=>({el:document.getElementById(id),index:null,url:null}));
const $=id=>document.getElementById(id);
const esc=s=>(s??'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function showPage(name){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.page===name));$('page-'+name).classList.add('active');}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>showPage(b.dataset.page));

async function init(){const r=await fetch('/api/bootstrap');bootstrap=await r.json();$('configPath').textContent=bootstrap.config_path;applyDefaults();renderProfiles();renderRuns();renderReviewRunOptions();updateChapterFields();updateDeviceFields();renderHardware(bootstrap.hardware);pollJob();}
function fillSelect(el,profiles,selected){el.innerHTML=Object.entries(profiles).map(([name,p])=>`<option value="${esc(name)}" ${name===selected?'selected':''}>${esc(p.label||name)}</option>`).join('');}
function applyDefaults(){const d=bootstrap.defaults;$('chapterSource').value=d.chapter_source;$('chapterPath').value=d.chapter_path||'';$('outputRoot').value=d.output_root||'';$('assetRegistry').value=d.asset_registry_path||'';$('knowledgePaths').value=(d.knowledge_paths||[]).join('\n');$('takesPerClip').value=d.generation.takes_per_clip;$('regenRounds').value=d.generation.max_regeneration_rounds;$('seedBase').value=d.generation.seed_base;$('deviceMode').value=d.generation.device_mode||'auto';$('autoStrategy').value=d.generation.auto_device_strategy||'one_per_gpu';$('reserveVram').value=d.generation.reserve_vram_gb??1;$('minGpuMemory').value=d.generation.min_gpu_memory_gb??0;$('workerGroups').value=JSON.stringify(d.generation.worker_device_groups||[]);$('phaseSwap').checked=d.generation.phase_swap_local_models!==false;$('minIdentity').value=d.review.min_identity;$('minContinuity').value=d.review.min_continuity;$('minStory').value=d.review.min_story_fidelity;$('minVisual').value=d.review.min_visual_quality;$('preloadAdjacent').checked=!!d.viewer_preload_adjacent;$('autoAssemble').checked=!!d.auto_assemble_on_selection;fillSelect($('llmProfile'),bootstrap.llm_profiles,d.llm_profile);fillSelect($('videoProfile'),bootstrap.video_profiles,d.video_profile);updateProfileDescriptions();updateDeviceFields();}
function updateProfileDescriptions(){const lp=bootstrap.llm_profiles[$('llmProfile').value];const vp=bootstrap.video_profiles[$('videoProfile').value];$('llmDesc').textContent=lp?.description||lp?.class_path||'';$('videoDesc').textContent=vp?.description||vp?.class_path||'';}
$('llmProfile').onchange=updateProfileDescriptions;$('videoProfile').onchange=updateProfileDescriptions;$('chapterSource').onchange=updateChapterFields;$('deviceMode').onchange=updateDeviceFields;
function updateChapterFields(){const s=$('chapterSource').value;$('chapterPathWrap').classList.toggle('hidden',s!=='path');$('chapterTextWrap').classList.toggle('hidden',s!=='text');$('chapterUploadWrap').classList.toggle('hidden',s!=='upload');}
function updateDeviceFields(){const manual=$('deviceMode').value==='manual';$('workerGroupsWrap').classList.toggle('hidden',!manual);$('autoStrategy').disabled=manual;$('reserveVram').disabled=manual;$('minGpuMemory').disabled=manual;}
function fmtSeconds(v){if(v===null||v===undefined||!isFinite(v))return '—';v=Math.max(0,Number(v));if(v<60)return `${v.toFixed(v<10?1:0)}s`;const h=Math.floor(v/3600),m=Math.floor((v%3600)/60),sec=Math.floor(v%60);return h?`${h}h ${m}m ${sec}s`:`${m}m ${sec}s`;}
function renderHardware(hw){if(!hw)return;const root=$('hardwareList');if(!hw.gpus?.length){root.innerHTML=`<div>${esc(hw.detection_error||'No NVIDIA GPUs detected')}</div>`;return;}root.innerHTML=`<div class="gpuRow"><strong>GPU</strong><strong>VRAM free</strong><strong>Util</strong><strong>Temp</strong><strong>Power</strong></div>`+hw.gpus.map(g=>`<div class="gpuRow"><span>${esc(`cuda:${g.index} · ${g.name}`)}</span><span>${Number(g.memory_free_gb).toFixed(1)} / ${Number(g.memory_total_gb).toFixed(1)} GB</span><span>${g.utilization_pct??'—'}%</span><span>${g.temperature_c??'—'}°C</span><span>${g.power_draw_w==null?'—':Number(g.power_draw_w).toFixed(0)+' W'}</span></div>`).join('');}
$('chapterFile').onchange=async e=>{const f=e.target.files?.[0];if(!f)return;uploadedChapter={text:await f.text(),name:f.name.replace(/\.[^.]+$/,'')};$('uploadName').textContent=`Loaded ${f.name} (${uploadedChapter.text.length.toLocaleString()} characters)`;};
function formPayload(){let groups=[];if($('deviceMode').value==='manual'){try{groups=JSON.parse($('workerGroups').value||'[]')}catch(e){throw new Error('Manual worker device groups must be valid JSON.')}}return {chapter_source:$('chapterSource').value,chapter_path:$('chapterPath').value||null,chapter_text:$('chapterSource').value==='upload'?uploadedChapter.text:$('chapterText').value,chapter_name:$('chapterSource').value==='upload'?uploadedChapter.name:null,llm_profile:$('llmProfile').value,video_profile:$('videoProfile').value,output_root:$('outputRoot').value||null,asset_registry_path:$('assetRegistry').value||null,knowledge_paths:$('knowledgePaths').value.split('\n').map(x=>x.trim()).filter(Boolean),generation:{takes_per_clip:Number($('takesPerClip').value),max_regeneration_rounds:Number($('regenRounds').value),seed_base:Number($('seedBase').value),device_mode:$('deviceMode').value,worker_device_groups:groups,auto_device_strategy:$('autoStrategy').value,reserve_vram_gb:Number($('reserveVram').value),min_gpu_memory_gb:Number($('minGpuMemory').value),phase_swap_local_models:$('phaseSwap').checked},review:{min_identity:Number($('minIdentity').value),min_continuity:Number($('minContinuity').value),min_story_fidelity:Number($('minStory').value),min_visual_quality:Number($('minVisual').value)},auto_assemble_on_selection:$('autoAssemble').checked,viewer_preload_adjacent:$('preloadAdjacent').checked};}
async function startRun(){try{const payload=formPayload();if(payload.chapter_source==='upload'&&!payload.chapter_text)throw new Error('Choose a chapter file first.');const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not start run');showJob(d);startJobPolling();}catch(e){$('jobStage').textContent='Error';$('jobMessage').textContent=e.message;}}
async function saveDefaults(){try{const p=formPayload();const prefs={active_llm_profile:p.llm_profile,active_video_profile:p.video_profile,chapter_source:p.chapter_source,chapter_path:p.chapter_path,output_root:p.output_root,asset_registry_path:p.asset_registry_path,knowledge_paths:p.knowledge_paths,generation:p.generation,review:p.review,auto_assemble_on_selection:p.auto_assemble_on_selection,viewer_preload_adjacent:p.viewer_preload_adjacent,custom_llm_profiles:bootstrap.llm_profiles?Object.fromEntries(Object.entries(bootstrap.llm_profiles).filter(([n,v])=>v.custom)): {},custom_video_profiles:bootstrap.video_profiles?Object.fromEntries(Object.entries(bootstrap.video_profiles).filter(([n,v])=>v.custom)): {}};const r=await fetch('/api/preferences',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(prefs)});bootstrap=await r.json();$('jobMessage').textContent='UI defaults saved';}catch(e){$('jobMessage').textContent=e.message;}}
function showJob(j){$('jobStage').textContent=j.stage||j.status;$('jobMessage').textContent=j.message||'';$('jobBar').style.width=`${Math.max(0,Math.min(100,(j.progress??0)*100))}%`;const st=j.stats||{};$('jobDetail').textContent=j.error||((j.status==='running'&&j.started_at)?`Running for ${fmtSeconds(Date.now()/1000-j.started_at)}`:j.status);const pc=st.phase_completed,pt=st.phase_total;const phasePct=(pc!=null&&pt)?Math.max(0,Math.min(100,pc/pt*100)):0;$('phaseBar').style.width=`${phasePct}%`;$('phaseDetail').textContent=pt?`${pc??0}/${pt} in current phase${st.phase_eta_seconds!=null?` · ETA ${fmtSeconds(st.phase_eta_seconds)}`:''}`:(st.current_phase||'');$('statLlm').textContent=st.llm_calls??0;$('statVideo').textContent=st.video_calls??0;$('statFootage').textContent=fmtSeconds(st.requested_video_seconds??0);$('statRtf').textContent=st.video_seconds_per_output_second==null?'—':Number(st.video_seconds_per_output_second).toFixed(1)+'×';if(st.device_plan_explanation)$('devicePlan').textContent=st.device_plan_explanation;if(j.hardware)renderHardware(j.hardware);if(j.status==='completed'&&j.summary){$('jobDetail').textContent=`${j.summary.scenes} scenes · ${j.summary.shots} shots · ${j.summary.clips} clips · ${j.summary.generated_takes} takes · ${fmtSeconds(j.summary.elapsed_seconds)}`;}}
async function pollJob(){const r=await fetch('/api/job');const j=await r.json();showJob(j);if(j.status==='running')startJobPolling();if(j.status==='completed'){await refreshBootstrap();if(j.summary?.run_dir){const name=j.summary.run_dir.split('/').pop();$('reviewRun').value=name;await loadReview();showPage('review');}}}
function startJobPolling(){clearInterval(jobTimer);jobTimer=setInterval(async()=>{const r=await fetch('/api/job');const j=await r.json();showJob(j);if(j.status!=='running'){clearInterval(jobTimer);jobTimer=null;if(j.status==='completed'){await refreshBootstrap();const name=j.summary?.run_dir?.split('/').pop();if(name){$('reviewRun').value=name;await loadReview();showPage('review');}}}},1500);}
async function refreshBootstrap(){const r=await fetch('/api/bootstrap');bootstrap=await r.json();renderRuns();renderReviewRunOptions();renderProfiles();renderHardware(bootstrap.hardware);}
function renderRuns(){const root=$('runsList');root.innerHTML=(bootstrap.runs||[]).map(x=>`<div class="runRow"><div class="grow"><strong>${esc(x.name)}</strong><div class="muted">${esc(x.path)}</div><div class="muted">${x.summary?.scenes??'?'} scenes · ${x.summary?.shots??'?'} shots · ${x.summary?.clips??'?'} clips</div></div><button onclick="openRun('${esc(x.name)}')">Review</button></div>`).join('')||'<div class="muted">No completed runs yet.</div>';}
function renderReviewRunOptions(){const current=$('reviewRun').value;$('reviewRun').innerHTML=(bootstrap.runs||[]).map(x=>`<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');if(current&&[...$('reviewRun').options].some(o=>o.value===current))$('reviewRun').value=current;}
async function openRun(name){$('reviewRun').value=name;await loadReview();showPage('review');}

function preloadEnabled(){return $('preloadAdjacent').checked;}
function clipAt(i){return reviewState?.clips?.[i];}function currentPlayer(){return activeSlot?.el||null;}
function desiredIndices(center){if(!preloadEnabled())return[center];return[center-1,center,center+1].filter(i=>i>=0&&i<(reviewState?.clips?.length||0));}
function findSlot(index){return rslots.find(s=>s.index===index)||null;}
function clearSlot(slot){if(!slot)return;slot.el.pause();slot.el.removeAttribute('src');slot.el.load();slot.el.classList.remove('active');slot.index=null;slot.url=null;}
function chooseReusableSlot(index){const existing=findSlot(index);if(existing)return existing;const desired=new Set(desiredIndices(currentIndex));const empty=rslots.find(s=>s.index===null);if(empty)return empty;return rslots.find(s=>!desired.has(s.index))||rslots[0];}
function assignSlot(index,force=false){if(index<0||index>=(reviewState?.clips?.length||0))return null;const c=clipAt(index);let slot=findSlot(index);if(slot&&slot.url===c.selected_url&&!force)return slot;if(!slot)slot=chooseReusableSlot(index);if(slot===activeSlot)slot.el.pause();slot.index=index;slot.url=c.selected_url;slot.el.src=c.selected_url;slot.el.preload='auto';slot.el.load();return slot;}
function primeAround(index){const wanted=new Set(desiredIndices(index));for(const i of wanted)assignSlot(i,false);for(const slot of rslots)if(slot.index!==null&&!wanted.has(slot.index)&&slot!==activeSlot)clearSlot(slot);}
function whenMetadata(slot,fn){if(slot.el.readyState>=1)fn();else slot.el.addEventListener('loadedmetadata',fn,{once:true});}
function activateIndex(index,shouldPlay,restart=true){if(!reviewState?.clips?.length)return;currentIndex=Math.max(0,Math.min(index,reviewState.clips.length-1));primeAround(currentIndex);const slot=assignSlot(currentIndex,false);rslots.forEach(s=>s.el.classList.toggle('active',s===slot));activeSlot=slot;whenMetadata(slot,()=>{if(restart)slot.el.currentTime=0;if(shouldPlay)slot.el.play().catch(()=>{});});renderReview();primeAround(currentIndex);}
function renderReview(){if(!reviewState)return;const c=clipAt(currentIndex);$('reviewTitle').innerHTML=c?`<strong>${currentIndex+1}/${reviewState.clips.length} — ${esc(c.clip_id)} — take ${c.selected_take}</strong>`:'<strong>No clips</strong>';$('currentTakeButtons').innerHTML=c?c.takes.map(t=>`<button class="${t.take_number===c.selected_take?'selected':''}" onclick="selectTake('${esc(c.clip_id)}',${t.take_number},${currentIndex})">${t.take_number}</button>`).join(' '):'';$('reviewClips').innerHTML=reviewState.clips.map((x,i)=>`<div class="clip ${i===currentIndex?'current':''}" onclick="jumpToClip(${i})"><div class="clipHead"><strong>${esc(x.clip_id)}</strong><span>${i+1}/${reviewState.clips.length}</span></div><div class="muted">${esc(x.scene_id)} · ${esc(x.shot_id)}${x.target_duration_sec?` · ${Number(x.target_duration_sec).toFixed(1)}s`:''}</div><div class="takeButtons">${x.takes.map(t=>`<button class="${t.take_number===x.selected_take?'selected':''}" onclick="event.stopPropagation();selectTake('${esc(x.clip_id)}',${t.take_number},${i})">Take ${t.take_number}</button>`).join('')}</div>${x.judgement_reason?`<div class="muted">${esc(x.judgement_reason)}</div>`:''}</div>`).join('');}
async function loadReview(){const run=$('reviewRun').value;if(!run)return;const r=await fetch('/api/review/state?run='+encodeURIComponent(run));if(!r.ok){$('reviewStatus').textContent=await r.text();return;}reviewState=await r.json();currentIndex=0;rslots.forEach(clearSlot);activeSlot=null;renderReview();if(reviewState.clips.length)activateIndex(0,false,true);$('reviewStatus').textContent='';}
function jumpToClip(i){const p=currentPlayer();const play=!!p&&!p.paused&&!p.ended;if(p)p.pause();activateIndex(i,play,true);}function nextClip(){if(currentIndex>=reviewState.clips.length-1)return;const p=currentPlayer(),play=!!p&&!p.paused&&!p.ended;if(p)p.pause();activateIndex(currentIndex+1,play,true);}function previousClip(){if(currentIndex<=0)return;const p=currentPlayer(),play=!!p&&!p.paused&&!p.ended;if(p)p.pause();activateIndex(currentIndex-1,play,true);}
async function selectTake(clipId,takeNumber,index){const old=clipAt(index);if(old?.selected_take===takeNumber)return;const changing=index===currentIndex;const p=currentPlayer();const resume=changing&&!!p&&!p.paused&&!p.ended;const run=$('reviewRun').value;const r=await fetch('/api/review/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run,clip_id:clipId,take_number:takeNumber})});if(!r.ok){$('reviewStatus').textContent=await r.text();return;}reviewState=await r.json();const loaded=findSlot(index);if(loaded)assignSlot(index,true);if(changing)activateIndex(currentIndex,resume,true);else{primeAround(currentIndex);renderReview();}$('reviewStatus').textContent=`${clipId} → take ${takeNumber}`;}
async function rebuildCut(){const run=$('reviewRun').value;if(!run)return;$('reviewStatus').textContent='Rebuilding…';const r=await fetch('/api/review/assemble',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run})});const d=await r.json();$('reviewStatus').textContent=d.rough_cut?'rough_cut.mp4 rebuilt':'Nothing assembled';}
rslots.forEach(slot=>slot.el.addEventListener('ended',()=>{if(slot===activeSlot&&currentIndex<reviewState.clips.length-1)activateIndex(currentIndex+1,true,true);}));
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;if(document.getElementById('page-review').classList.contains('active')){if(e.key===']')nextClip();if(e.key==='[')previousClip();const n=Number(e.key);const c=clipAt(currentIndex);if(n>=1&&n<=9&&c?.takes.some(t=>t.take_number===n))selectTake(c.clip_id,n,currentIndex);}});

function mergedProfiles(kind){return kind==='llm'?bootstrap.llm_profiles:bootstrap.video_profiles;}
function renderProfiles(){const kind=$('profileKind').value;const profiles=mergedProfiles(kind);$('profileExisting').innerHTML=Object.entries(profiles).map(([name,p])=>`<option value="${esc(name)}">${esc(p.label||name)}${p.custom?' (browser)':''}</option>`).join('');}
$('profileKind').onchange=renderProfiles;
function loadProfileEditor(){const kind=$('profileKind').value,name=$('profileExisting').value,p=mergedProfiles(kind)[name];if(!p)return;$('profileName').value=name;$('profileLabel').value=p.label||'';$('profileClass').value=p.class_path||'';$('profileDescription').value=p.description||'';$('profileOptions').value=JSON.stringify(p.options||{},null,2);$('profileStatus').textContent=p.custom?'Browser-created profile loaded':'TOML profile loaded; saving creates an overlay';}
async function saveProfile(){try{const options=JSON.parse($('profileOptions').value||'{}');const body={kind:$('profileKind').value,name:$('profileName').value.trim(),class_path:$('profileClass').value.trim(),label:$('profileLabel').value.trim()||null,description:$('profileDescription').value.trim()||null,options};const r=await fetch('/api/profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not save profile');bootstrap=d;renderProfiles();applyDefaults();$('profileStatus').textContent='Saved';}catch(e){$('profileStatus').textContent=e.message;}}
async function deleteProfile(){const kind=$('profileKind').value,name=$('profileExisting').value,p=mergedProfiles(kind)[name];if(!p?.custom){$('profileStatus').textContent='Only browser-created overlays can be deleted here.';return;}const r=await fetch(`/api/profile/${kind}/${encodeURIComponent(name)}`,{method:'DELETE'});const d=await r.json();if(!r.ok){$('profileStatus').textContent=d.detail||'Delete failed';return;}bootstrap=d;renderProfiles();applyDefaults();$('profileStatus').textContent='Deleted browser overlay';}
init();
</script>
</body>
</html>'''
