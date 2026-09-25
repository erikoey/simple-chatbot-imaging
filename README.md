# simple-chatbot-imaging

Standalone image generation package with pluggable providers, opt-in fallback
orchestration, and a true async API.

## Installation

```bash
uv add simple-chatbot-imaging            # core + OpenRouter provider
uv add "simple-chatbot-imaging[huggingface]"  # + Hugging Face provider
uv add "simple-chatbot-imaging[all]"      # everything
```

The `fallback` extra enables fallback orchestration only; install the provider
extras you want in the chain yourself (e.g. `[huggingface]`).

## Quick start

```python
from simple_chatbot_imaging import create_image_generator

gen = create_image_generator("openrouter", media_path="media")
path = await gen.generate_image_async("a neon cyberpunk city at night")
```

Synchronous use:

```python
path = gen.generate_image("a neon cyberpunk city at night")
```

## Providers

### OpenRouter (`openrouter`)

- Uses the OpenRouter `images/generations` API via `httpx` (native async).
- Requires the `OPENROUTER_API_KEY` environment variable.
- Options: `model` (default `meta/muse-image`), `base_url`, `api_key_env`,
  `request_timeout_seconds`.

### Hugging Face (`huggingface`, extra: `huggingface`)

- Runs a Gradio Space (ZeroGPU) via `gradio-client`.
- Uses `HUGGINGFACE_ACCESS_TOKEN` if set.
- Options: `space_id` (default `hugging-apps/qwen-image-2-1`), `hf_token_env`.

## Fallback (opt-in)

Fallback is never implicit. Build it explicitly with a caller-defined chain:

```python
from simple_chatbot_imaging import FallbackImageGenerator, create_image_generator

providers = [create_image_generator("openrouter"), create_image_generator("huggingface")]
gen = FallbackImageGenerator(generators=providers)
```

Each provider runs its full lifecycle (retries, state, output movement); the
fallback only advances to the next provider on failure.

## API

- `BaseImageGenerator` — abstract provider base; prompt validation, retries with
  backoff, timeouts, state tracking, output movement.
- `generate_image(...)` — synchronous entry point.
- `generate_image_async(...)` — true async entry point (native async providers).
- `ImageGenerationError(retryable=...)` — permanent errors (bad key, no credits,
  403/404) are not retried.
- `ImageGenerationState` — `IDLE`, `GENERATING`, `ERROR`, `SUCCESS`.
- `create_image_generator(provider, **kwargs)` — factory; raises a clear error
  when an optional provider's extra is not installed.
- `register_provider(name, factory)` — plugin hook.

## Environment variables

| Variable | Provider | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | openrouter | yes |
| `HUGGINGFACE_ACCESS_TOKEN` | huggingface | no (recommended for ZeroGPU spaces) |