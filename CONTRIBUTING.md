# Contributing

Book2Video is structured so new model backends can usually be added without replacing existing ones.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

`ffmpeg` is expected for rough-cut assembly. NVIDIA GPU discovery uses `nvidia-smi` when available, but the mock providers and most orchestration tests do not require a GPU.

## Adding providers

If a model already works through a supported runtime, prefer adding a named profile rather than a new provider class.

For a genuinely different backend, add a provider under `src/book2video/providers/` and point a profile at its import path. `examples/custom_llm_provider.py` and `examples/custom_video_provider.py` are intentionally small starting points.

Provider lifecycle methods (`prepare` / `release`) are optional. They are useful for local backends when LLM and video models share GPU memory and should be resident in separate pipeline phases.

## Pull requests

Please include tests for behavior that can be exercised without paid API access. For provider-specific integrations, document any required environment variables and note which behavior was verified against the real service versus mocked locally.
