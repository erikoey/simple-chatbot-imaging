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

The recommended API is the `generate(...)` convenience method. It returns an
`ImageGenerationResult` with the output path and execution metadata:

```python
from simple_chatbot_imaging import create_image_generator

gen = create_image_generator("openrouter", media_path="media")
result = await gen.generate("a neon cyberpunk city at night")
print(result.path)          # Path to the saved image
print(result.attempts)      # attempts made (including retries)
print(result.duration_seconds)
```

For typed/batch callers, use the request-first canonical API:

```python
from simple_chatbot_imaging import ImageGenerationRequest

request = ImageGenerationRequest(
    prompt="a neon cyberpunk city at night",
    negative_prompt="blurry, low quality",
    resolution=1024,
    aspect_ratio="16:9",
    filename="city.png",
)
result = await gen.generate_image_async(request)
```

Synchronous use:

```python
request = ImageGenerationRequest(prompt="a neon cyberpunk city at night")
result = gen.generate_image(request)
print(result.path)
```

### Lifecycle events (optional)

Pass `on_event` to observe the request lifecycle (`QUEUED`, `GENERATING`,
`RETRYING`, `SAVING`, `SUCCESS`, `ERROR`, `CANCELLED`). Sync and async
callbacks are both supported; callback failures are logged, never raised:

```python
def handle_event(event):
    print(event.status, event.provider_name, event.attempt)

result = await gen.generate("a neon cyberpunk city at night", on_event=handle_event)
```

## Provider configuration file

Provider defaults live in [src/simple_chatbot_imaging/providers.json](src/simple_chatbot_imaging/providers.json). You can edit that file to change API endpoints, model IDs, or add your own providers without changing library code.

```json
{
  "openrouter": {
    "kwargs": {
      "base_url": "https://openrouter.ai/api/v1/images",
      "model": "meta/muse-image",
      "api_key_env": "OPENROUTER_API_KEY"
    }
  },
  "custom_provider": {
    "factory": "your_package.providers:YourCustomProvider",
    "kwargs": {
      "base_url": "https://example.com/api/v1/images",
      "api_key_env": "CUSTOM_API_KEY"
    }
  }
}
```

You can also pass a custom config path explicitly:

```python
from simple_chatbot_imaging import create_image_generator

gen = create_image_generator(
    "openrouter",
    config_path="./providers.json",
    base_url="https://example.com/api/v1/images",
)
```

### OpenRouter (`openrouter`)

- Uses the OpenRouter `images/generations` API via `httpx` (native async).
- Requires the `OPENROUTER_API_KEY` environment variable.
- Options: `model` (default `meta/muse-image`), `base_url`, `api_key_env`,
  `request_timeout_seconds`.

### Hugging Face (`huggingface`, extra: `huggingface`)

- Runs a Gradio Space (ZeroGPU) via `gradio-client`.
- Uses `HUGGINGFACE_ACCESS_TOKEN` if set.
- Options: `space_id` (default `hugging-apps/qwen-image-2-1`), `hf_token_env`.

## Custom HTTP providers

OpenRouter and Qwen share a common base class,
`BaseHTTPImageGenerator` (exported from the package root). It owns the full
JSON-over-HTTP pipeline — Bearer auth from an environment variable
(`SecretStr`), POST request, HTTP status mapping (401/402/403/404/429/5xx with
appropriate retryability), and temp-file handling.

The base also decides **how to obtain the image** from a response, in this
order:

1. `image/*` response content-type — the body itself is the image.
2. A base64 field (`b64_json`, `b64`, `base64`, `image_base64`) anywhere in the
   JSON — decoded (data-URL prefixes are stripped).
3. A URL field (`url`, `image_url`, `imageUrl`, `image`, `output_url`) anywhere
   in the JSON — downloaded with redirects.

Field names are class attributes (`BASE64_FIELDS` / `URL_FIELDS`), so a custom
provider only needs to implement `_build_payload` and set `provider_label`,
endpoint, and defaults — or extend the field tuples for exotic API shapes:

```python
from simple_chatbot_imaging import BaseHTTPImageGenerator

class MyProviderImageGenerator(BaseHTTPImageGenerator):
    provider_label = "MyProvider"
    DEFAULT_BASE_URL = "https://api.example.com/v1/images"
    DEFAULT_MODEL = "example-image"
    DEFAULT_API_KEY_ENV = "EXAMPLE_API_KEY"

    def _build_payload(self, request):
        return {"model": self.model, "prompt": request.prompt}
```

Then register it: `register_provider("example", MyProviderImageGenerator)`.

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
  backoff, timeouts, output movement into the media folder.
- `generate(prompt, **kwargs)` — async convenience method; builds an
  `ImageGenerationRequest` and returns an `ImageGenerationResult`.
- `generate_image_async(request, *, on_event=None)` — canonical async entry
  point; takes an `ImageGenerationRequest`, returns an `ImageGenerationResult`,
  raises `ImageGenerationError` on failure and propagates
  `asyncio.CancelledError` when the calling task is cancelled.
- `generate_image(request, *, on_event=None)` — synchronous bridge over
  `generate_image_async`; raises `RuntimeError` if called from a running event
  loop.
- `ImageGenerationRequest` — frozen dataclass: `prompt`, `negative_prompt`,
  `resolution`, `aspect_ratio`, `steps`, `seed`, `filename`, `overwrite`.
- `ImageGenerationResult` — frozen dataclass: `request_id`, `path`,
  `provider_name`, `model`, `request`, `attempts`, `started_at`,
  `completed_at`, `duration_seconds`.
- `ImageGenerationEvent` / `ImageGenerationStatus` — request-local lifecycle
  events (`QUEUED`, `GENERATING`, `RETRYING`, `SAVING`, `SUCCESS`, `ERROR`,
  `CANCELLED`) delivered via the optional `on_event` callback.
- `ImageGenerationError(retryable=...)` — permanent errors (bad key, no credits,
  403/404) are not retried.
- `create_image_generator(provider, **kwargs)` — factory; raises a clear error
  when an optional provider's extra is not installed.
- `register_provider(name, factory)` — plugin hook.
