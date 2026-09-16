# Book2Video

Book2Video is an experimental, provider-agnostic pipeline for turning a prose chapter into an unattended AI-video rough cut. It separates story analysis, scene/beat/shot planning, video generation, automated take review, and final human review so LLM and video backends can be swapped without redesigning the pipeline.

The normal interface is a localhost browser Studio. The CLI remains available for automation and debugging.

> **Project status:** early/experimental. The orchestration, mock providers, browser Studio, review flow, caching, GPU planning, and tests are intended to be usable now. Real model APIs and local video runtimes change quickly; individual provider profiles may need adjustment for the exact model/checkpoint you choose.

## Highlights

- Browser-first setup, run monitoring, model selection, and take review
- Provider profiles instead of hard-coded model choices
- Local and online LLM/video adapters
- Runtime NVIDIA GPU discovery and configurable multi-GPU planning
- Phase-based local model residency so a local LLM can unload before video generation and vice versa
- Multiple takes per clip, automated judging, bounded regeneration, and nondestructive manual overrides
- Previous/current/next clip preloading in the review playlist
- Content-addressed caching and structured JSON artifacts throughout the pipeline
- Mock providers for testing the complete workflow without API spend or a GPU

## Requirements

- Python 3.11+
- `ffmpeg` available on `PATH` for rough-cut assembly
- `nvidia-smi` is optional; it is used for automatic NVIDIA GPU discovery/telemetry when present
- Local/online provider runtimes and API keys only for the providers you choose to use

## Quick start

```bash
git clone <your-repository-url>
cd book2video
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp config.example.toml config.toml
book2video
```

With the example defaults, `book2video` starts a localhost web server and opens the Studio in your browser. The mock LLM/video profiles allow the orchestration and UI to be exercised before connecting a real backend.

## Browser Studio

The **Setup** page controls the normal run:

- LLM profile and video profile
- chapter source: config TOML, filesystem path, pasted text, or browser upload
- output root, asset registry, and knowledge/style files
- takes per clip, regeneration rounds, and seed base
- automatic or manual GPU layout
- automatic GPU strategy, VRAM planning reserve, and minimum-card-size filter
- batched local-model swapping
- automated review thresholds
- live run progress, ETA-like phase progress, GPU telemetry, and run statistics

Most controls have a `?` hover hint. Browser choices are stored in `config.studio.json` beside `config.toml`; the commented TOML is not rewritten.

## GPU changes are detected at run time

The default is now:

```toml
[generation]
device_mode = "auto"
auto_device_strategy = "one_per_gpu"
```

At the start of each run, Book2Video asks `nvidia-smi` what GPUs are actually present. This means the saved configuration does not assume that today's `cuda:0`/`cuda:1` layout will exist forever.

Examples:

```text
2 x 24 GB cards today  -> two workers when the model fits one card
1 x 48 GB replacement  -> one worker on that card
1 card removed         -> the remaining card is used
3 cards added later    -> three workers when appropriate
```

A video profile can optionally advertise an estimated requirement:

```toml
[video_profiles.wan.options]
estimated_vram_gb = 36.0
supports_multi_gpu_job = true
```

With two 24 GB cards and a 1 GB/card planning reserve, no single card meets that estimate, so the automatic planner can group the cards into one multi-GPU worker if that provider advertises multi-GPU support.

This is still a planning hint rather than magic VRAM detection. If a profile has no useful `estimated_vram_gb`, Book2Video cannot know whether an arbitrary checkpoint will actually fit; `one_per_gpu` then defaults to one worker per detected card. Manual groups remain available for unusual runtimes.

Manual examples:

```json
[["cuda:0"], ["cuda:1"]]
```

or:

```json
[["cuda:0", "cuda:1"]]
```

Old v1-v4 configs that explicitly contain `worker_device_groups` but no `device_mode` are treated as manual layouts for compatibility.

## Local LLM + local video model residency

v0.5 changes the execution order so local LLM and video work can be batched rather than constantly fighting over VRAM.

The default phase order is approximately:

```text
load/prepare LLM
  -> scene detection
  -> scene checking
  -> beats
  -> timing
  -> shot planning
  -> generation-prompt compilation
release LLM

load/prepare video model
  -> generate every initial take
release video model

load/prepare LLM
  -> judge every clip
release LLM

if clips need regeneration:
  load video -> generate rejected clips -> release video
  load LLM   -> judge those clips       -> release LLM

assemble rough cut
```

The setting is:

```toml
phase_swap_local_models = true
```

Provider lifecycle hooks are optional. Cloud providers treat them as no-ops. Local providers can use them to actually load/unload resources.

Ollama uses `keep_alive` while the LLM phase is active and attempts to unload its model on release. The OpenAI-compatible/vLLM adapter and local video adapters also accept optional lifecycle commands, for example:

```toml
[llm_profiles.vllm.options]
prepare_command = ["systemctl", "--user", "start", "book2video-vllm"]
release_command = ["systemctl", "--user", "stop", "book2video-vllm"]

[video_profiles.wan.options]
prepare_command = ["systemctl", "--user", "start", "book2video-wan"]
release_command = ["systemctl", "--user", "stop", "book2video-wan"]
```

A native future provider can override `prepare()` and `release()` directly instead.

### Continuation clips without interleaving the judge

A split shot may need the previous generated segment. During the initial video phase, continuation **take N follows take N** from the previous segment. This is being tried so a whole video phase can remain batched before the LLM judge is loaded again.

After the first judging pass, regeneration rounds can use the currently selected predecessor clip.

## Live progress instead of a two-day mystery box

The Studio polls the job state while a run is active. It shows:

- current phase and current operation
- rough overall progress bar
- current-phase completed/total counts
- estimated remaining time for the current measurable phase
- elapsed wall time
- LLM call count
- video generation call/take count
- requested output-video duration
- observed video-generation seconds per requested output second
- detected GPU name
- total/free VRAM
- GPU utilization
- GPU temperature
- power draw
- the automatic device-plan explanation

Generation updates after each completed take, so a long video phase should continue visibly moving even when the rough overall bar is intentionally conservative.

`run_stats.json` is also written inside the run folder and updated during the run. It records phase durations, LLM call counts by stage, video calls, generated/regeneration take counts, requested footage duration, device groups, lifecycle events, and related timing data.

A completed run now contains roughly:

```text
01_scenes.json
02_scene_records.json
03_beats.json
04_timing.json
05_shots.json
06_clips.json
07_generation_intents.json
08_selected.json
09_all_takes.json
hardware_start.json
run_stats.json
selection_overrides.json
summary.json
takes/
rough_cut.mp4
review.html
```

## Named model profiles

Several models can remain configured at once:

```toml
active_llm_profile = "ollama"
active_video_profile = "wan_command"

[llm_profiles.ollama]
label = "Ollama local"
class_path = "book2video.providers.ollama_llm:OllamaLLMProvider"
[llm_profiles.ollama.options]
model = "my-model"

[video_profiles.wan_command]
label = "Wan local"
class_path = "book2video.providers.command_video:CommandVideoProvider"
[video_profiles.wan_command.options]
max_duration_sec = 10.0
command = ["python", "./wan_wrapper.py", "--request", "{{REQUEST_JSON}}", "--output", "{{OUTPUT}}"]
```

The **Profiles** page can create browser-side profile overlays without deleting older profiles or rewriting the TOML.

## Chapter sources

The browser source dropdown means:

- **Config TOML**: use `chapter_path`, falling back to `chapter_text`, from `config.toml`.
- **Filesystem path**: read a path visible to the Book2Video process.
- **Paste / inline text**: use the browser text box.
- **Browser file upload**: select a `.md`/`.txt` file in the browser and send its text to localhost.

## Review playlist

The **Review** page plays selected individual clips as a continuous playlist rather than using `rough_cut.mp4` as the editing surface.

By default:

```text
previous clip -> loaded
current clip  -> visible
next clip     -> loaded
```

Changing the current clip from Take 2 to Take 3 restarts the changed clip from time zero. If playback was active, the new take starts playing from zero; if paused, it remains paused at zero.

Changing another clip does not interrupt the current clip. The changed clip's preload is refreshed if it is currently in the previous/next slot.

Keyboard shortcuts:

- `[` previous clip
- `]` next clip
- `1`-`9` choose a take for the current clip

When the choices look right, **Rebuild rough_cut.mp4** creates a conventional MP4 from the current selections.

## Pipeline shape

```text
Chapter
  -> scene boundary pass
  -> per-scene split check
  -> entity/location resolution
  -> beat breakdown + critic
  -> timing
  -> shot plan
  -> generator-independent physical clip plan
  -> generation-intent compilation
  -> batched video generation
  -> batched LLM judging
  -> bounded regeneration loops
  -> selected-take playlist
  -> rough_cut.mp4
```

Shots remain separate from physical generator clips. If a planner creates an 18-second shot but the selected model supports 10 seconds, a deterministic partition stage can split that shot without teaching the screenplay representation about today's model limit.

## Adding a model

If the new model uses an already-supported runtime, normally only a new profile is required.

If it is a genuinely new backend, add one provider class and point a profile at it. No central provider switch statement needs editing.

LLM providers implement:

```python
class MyLLMProvider(LLMProvider):
    @property
    def provider_id(self):
        return "my-llm"

    async def prepare(self):
        ...  # optional

    async def release(self):
        ...  # optional

    async def generate_structured(self, request, response_model):
        ...
```

Video providers implement:

```python
class MyVideoProvider(VideoProvider):
    @property
    def provider_id(self):
        return "my-video"

    @property
    def capabilities(self):
        return VideoCapabilities(
            max_duration_sec=10.0,
            supports_multi_gpu_job=True,
            estimated_vram_gb=36.0,  # optional planner hint
        )

    async def prepare(self, device_groups):
        ...  # optional

    async def release(self):
        ...  # optional

    async def generate(self, request, devices):
        ...
```

Starter files are under `examples/`.

## Built-in adapters

LLM:

- Mock
- Ollama
- OpenAI-compatible / vLLM / LM Studio
- OpenAI
- Anthropic
- Gemini

Video:

- Mock
- generic local command wrapper
- ComfyUI API workflow
- Runway
- Replicate
- fal

See `PROVIDERS.md` for provider-specific configuration.

## CLI remains available

```bash
book2video studio
book2video run --chapter-path ./chapter01.md --llm-profile ollama --video-profile wan_command
book2video viewer --run-dir ./book2video_output/chapter01_...
book2video assemble --run-dir ./book2video_output/chapter01_...
```

The browser is intended to make these unnecessary for ordinary use.

## Tests

```bash
pytest -q
```

v0.5 includes tests for config compatibility, hardware/device planning, named profiles, provider construction, clip partitioning, review behavior, Studio controls/profile overlays, and caching.

## Data and API-key notes

Local profiles can keep chapter text, character references, and generated media on the machine. Online profiles send the inputs required by that provider to its service, so check that provider's terms and data-handling rules before using private manuscripts or reference material.

API keys are read from environment variables by the included online adapters where supported. `config.toml`, `config.studio.json`, `.env*`, and generated output are ignored by the repository `.gitignore` by default to reduce accidental commits of machine-specific state or secrets.

## License

MIT. See [`LICENSE`](LICENSE).
