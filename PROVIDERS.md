# Provider adapters

The pipeline keeps provider choice in `config.toml`. These are copy/paste starting points rather than fixed recommendations; model names and model-specific video input fields change more often than the orchestration code should.

## LLM providers

The video judge sends sampled frames from each take to the chosen LLM. For local providers, pick a vision-capable model if you want that stage to actually inspect the generated video.

### Local: Ollama

```toml
[llm]
class_path = "book2video.providers.ollama_llm:OllamaLLMProvider"
[llm.options]
model = "YOUR_VISION_CAPABLE_OLLAMA_MODEL"
base_url = "http://127.0.0.1:11434"
video_frame_samples = 6
```

### Local: vLLM / LM Studio / OpenAI-compatible server

```toml
[llm]
class_path = "book2video.providers.openai_compatible_llm:OpenAICompatibleLLMProvider"
[llm.options]
model = "YOUR_LOCAL_MODEL"
base_url = "http://127.0.0.1:8000/v1"
api_key = ""
video_frame_samples = 6
```

### Online: OpenAI

```toml
[llm]
class_path = "book2video.providers.openai_llm:OpenAILLMProvider"
[llm.options]
model = "YOUR_OPENAI_MODEL"
# OPENAI_API_KEY is read from the environment by default.
video_frame_samples = 6
```

### Online: Anthropic

```toml
[llm]
class_path = "book2video.providers.anthropic_llm:AnthropicLLMProvider"
[llm.options]
model = "YOUR_CLAUDE_MODEL"
# ANTHROPIC_API_KEY is read from the environment by default.
video_frame_samples = 6
```

The Anthropic adapter requests the structured result through a forced tool call whose input schema is the Pydantic response schema. This is being used so the orchestration does not depend on free-form JSON parsing when a schema-bearing tool is available.

### Online: Gemini

```toml
[llm]
class_path = "book2video.providers.gemini_llm:GeminiLLMProvider"
[llm.options]
model = "YOUR_GEMINI_MODEL"
# GEMINI_API_KEY is read from the environment by default.
video_frame_samples = 6
```

## Video providers

### Local: generic command wrapper

This is the least opinionated adapter for native LightX2V, Wan, SkyReels, or a future MCP-facing executable. The command receives a generated request JSON and the requested GPU list.

```toml
[video]
class_path = "book2video.providers.command_video:CommandVideoProvider"
[video.options]
max_duration_sec = 10.0
supports_multi_gpu_job = true
command = [
  "python", "./my_wan_wrapper.py",
  "--request", "{{REQUEST_JSON}}",
  "--output", "{{OUTPUT}}"
]
```

Useful command placeholders include `{{REQUEST_JSON}}`, `{{OUTPUT}}`, `{{PROMPT}}`, `{{SEED}}`, `{{DURATION}}`, and `{{DEVICES_CSV}}`. The sidecar request JSON also contains all reference paths, continuation paths, and canonical generation intent.

### Local: ComfyUI API workflow

Export a workflow in API format and place placeholders in widget values where the workflow expects runtime data.

```toml
[video]
class_path = "book2video.providers.comfyui_video:ComfyUIVideoProvider"
[video.options]
workflow_path = "./workflows/wan22_i2v_api.json"
max_duration_sec = 10.0
server_url = "http://127.0.0.1:8188"

[video.options.server_by_device]
"cuda:0" = "http://127.0.0.1:8188"
"cuda:1" = "http://127.0.0.1:8189"
```

Recognized workflow placeholders include:

`{{PROMPT}}`, `{{MOTION_PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{SEED}}`, `{{DURATION}}`, `{{OUTPUT_PREFIX}}`, `{{CHAR_REF_0}}`, `{{CHAR_REF_1}}`, `{{LOC_REF_0}}`, and `{{CONTINUATION_FRAME}}`.

Running two ComfyUI servers is one possible way to let the pipeline's two workers target the two 3090s independently. A different workflow/server arrangement can be used without changing the pipeline.

### Online: Runway Model Router

```toml
[video]
class_path = "book2video.providers.runway_video:RunwayVideoProvider"
[video.options]
config_id = "preview-fast"
max_duration_sec = 30.0
aspect_ratio = "16:9"
# RUNWAYML_API_SECRET is read from the environment.
```

The built-in router payload is intentionally conservative and uses at most one first-frame reference. `payload_template` can replace it when a specific Runway model exposes multi-character references, audio references, or other fields that differ from the router's generic shape.

### Online: Replicate

Replicate video models have different input schemas, so the adapter accepts a template rather than pretending one input shape covers all of them.

```toml
[video]
class_path = "book2video.providers.replicate_video:ReplicateVideoProvider"
[video.options]
model = "OWNER/MODEL"
max_duration_sec = 10.0
output_path = "output"

[video.options.input_template]
prompt = "{{PROMPT}}"
duration = "{{DURATION}}"
seed = "{{SEED}}"
reference_images = "{{REFERENCE_DATA_URIS}}"
```

### Online: fal

```toml
[video]
class_path = "book2video.providers.fal_video:FalVideoProvider"
[video.options]
model = "MODEL/ENDPOINT"
max_duration_sec = 10.0

[video.options.arguments_template]
prompt = "{{PROMPT}}"
duration = "{{DURATION}}"
seed = "{{SEED}}"
image_urls = "{{REFERENCE_DATA_URIS}}"
```

For Replicate/fal, the exact argument template should be adjusted to the chosen model's published input schema. That keeps a new model addition mostly in configuration rather than forcing a provider rewrite.

## Template values for online video adapters

Whole-value placeholders preserve list/dict types. Common values include:

- `{{PROMPT}}`, `{{MOTION_PROMPT}}`, `{{NEGATIVE_PROMPT}}`
- `{{DURATION}}`, `{{SEED}}`
- `{{CHARACTER_REFERENCE_DATA_URIS}}`
- `{{LOCATION_REFERENCE_DATA_URIS}}`
- `{{REFERENCE_DATA_URIS}}`
- `{{VOICE_REFERENCE_DATA_URIS}}`
- `{{CONTINUATION_FRAME_DATA_URI}}`
- `{{CONTINUATION_VIDEO_DATA_URI}}`

## Named profiles and the Studio

v0.4 can keep several provider/model configurations at the same time under `llm_profiles` and `video_profiles`. The Studio model dropdown reads those profiles directly.

A second model using an existing provider normally only needs another profile:

```toml
[video_profiles.wan_fast]
label = "Wan fast local"
class_path = "book2video.providers.command_video:CommandVideoProvider"
[video_profiles.wan_fast.options]
max_duration_sec = 10.0
command = ["python", "./wan_fast.py", "--request", "{{REQUEST_JSON}}", "--output", "{{OUTPUT}}"]

[video_profiles.wan_quality]
label = "Wan quality local"
class_path = "book2video.providers.command_video:CommandVideoProvider"
[video_profiles.wan_quality.options]
max_duration_sec = 10.0
command = ["python", "./wan_quality.py", "--request", "{{REQUEST_JSON}}", "--output", "{{OUTPUT}}"]
```

The Studio **Profiles** page can create the same sort of profile as a browser-side overlay. Those overlays are stored in `config.studio.json`; `config.toml` is left unchanged so its comments are not lost.

## v0.5: local model lifecycle and VRAM planning

Local providers can now participate in phase-level load/unload behavior through `prepare()` and `release()`.

For an OpenAI-compatible local LLM server, optional commands can own the server lifetime:

```toml
[llm_profiles.local_vllm.options]
model = "YOUR_MODEL"
base_url = "http://127.0.0.1:8000/v1"
prepare_command = ["systemctl", "--user", "start", "book2video-vllm"]
release_command = ["systemctl", "--user", "stop", "book2video-vllm"]
```

For the generic local command video provider:

```toml
[video_profiles.wan_local.options]
command = ["python", "./wan_wrapper.py", "--request", "{{REQUEST_JSON}}", "--output", "{{OUTPUT}}"]
supports_multi_gpu_job = true
estimated_vram_gb = 36.0
prepare_command = ["systemctl", "--user", "start", "book2video-wan"]
release_command = ["systemctl", "--user", "stop", "book2video-wan"]
```

`estimated_vram_gb` is optional and is only used by Book2Video's automatic worker planner. It is not treated as a promise that a particular checkpoint will fit, because actual usage can depend on resolution, quantization, offloading, attention backend, and runtime implementation.

The ComfyUI adapter also accepts optional `prepare_command` / `release_command` values. This makes it possible to use a managed ComfyUI service or wrapper if you want the video phase to actually relinquish VRAM before the LLM judging phase.

Ollama uses `keep_alive` while serving LLM requests and attempts a best-effort model unload when its release hook is called. Set `unload_on_release = false` if that behavior is not wanted.
