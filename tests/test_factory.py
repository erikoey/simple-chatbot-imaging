"""Tests for the factory and opt-in fallback orchestration."""

import asyncio

import pytest

from conftest import FakeImageGenerator
from simple_chatbot_imaging.config import load_provider_config
from simple_chatbot_imaging.factory import (
    FallbackImageGenerator,
    create_image_generator,
    register_provider,
)
from simple_chatbot_imaging.models import ImageGenerationError


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown image provider"):
        create_image_generator("does-not-exist")


def test_no_implicit_fallback_default():
    """create_image_generator must not silently build a fallback."""
    with pytest.raises(ValueError, match="Unknown image provider"):
        create_image_generator("fallback")


def test_register_provider_plugin():
    register_provider("fake", FakeImageGenerator)
    gen = create_image_generator("fake", media_path="media_test_plugin")
    assert isinstance(gen, FakeImageGenerator)


def test_qwen_provider_downloads_image_url(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, *, status_code=200, payload=None, content=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.content = content or b"\x89PNG\r\n\x1a\n" + b"0" * 16
            self.headers = headers or {"content-type": "image/png"}
            self.text = "ok"

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            assert url.endswith("/generation")
            assert json["model"] == "qwen-image-3.0"
            params = json["parameters"]
            assert params["prompt_extend"] is True
            assert params["watermark"] is False
            assert params["size"] == "1024*1024"
            assert params["n"] == 1
            # negative_prompt goes in parameters, not folded into the prompt.
            assert "negative_prompt" not in json["input"]["messages"][0]["content"][0]["text"]
            return FakeResponse(payload={
                "output": {
                    "choices": [{
                        "message": {
                            "content": [{
                                "image": "https://example.com/image.png"
                            }]
                        }
                    }]
                }
            })

        async def get(self, url, follow_redirects=True):
            assert url == "https://example.com/image.png"
            return FakeResponse(content=b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("simple_chatbot_imaging.providers.http_base.httpx.AsyncClient", FakeAsyncClient)

    gen = create_image_generator("qwen", media_path=str(tmp_path))
    assert gen.base_url.endswith("/generation")

    path = asyncio.run(gen.generate(
        "A neon fox in the rain",
        negative_prompt="blurry, low quality",
    )).path
    assert path.is_file()


def test_qwen_provider_passes_negative_prompt_in_parameters(tmp_path, monkeypatch):
    """negative_prompt must go to parameters.negative_prompt per Qwen docs."""

    captured = {}

    class FakeResponse:
        def __init__(self, *, status_code=200, payload=None, content=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.content = content or b"\x89PNG\r\n\x1a\n" + b"0" * 16
            self.headers = headers or {"content-type": "image/png"}
            self.text = "ok"

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured.update(json)
            return FakeResponse(payload={
                "output": {
                    "choices": [{
                        "message": {
                            "content": [{
                                "image": "https://example.com/image.png"
                            }]
                        }
                    }]
                }
            })

        async def get(self, url, follow_redirects=True):
            return FakeResponse(content=b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("simple_chatbot_imaging.providers.http_base.httpx.AsyncClient", FakeAsyncClient)

    gen = create_image_generator("qwen", media_path=str(tmp_path))
    asyncio.run(gen.generate(
        "A red sports car",
        negative_prompt="low resolution, distorted limbs",
        resolution=2048,
        aspect_ratio="16:9",
    ))

    params = captured["parameters"]
    assert params["negative_prompt"] == "low resolution, distorted limbs"
    assert params["size"] == "2048*1152"  # 16:9 mapped into width*height
    text = captured["input"]["messages"][0]["content"][0]["text"]
    assert text == "A red sports car"


def test_provider_config_overrides_openrouter_base_url(tmp_path):
    config_path = tmp_path / "providers.json"
    config_path.write_text(
        '{"openrouter": {"kwargs": {"base_url": "https://example.com/api/v1/images"}}}',
        encoding="utf-8",
    )

    config = load_provider_config(config_path)
    assert config["openrouter"]["kwargs"]["base_url"] == "https://example.com/api/v1/images"

    gen = create_image_generator("openrouter", media_path=str(tmp_path), config_path=config_path)
    assert gen.base_url == "https://example.com/api/v1/images"


def test_fallback_requires_chain(tmp_media):
    with pytest.raises(ValueError, match="non-empty"):
        FallbackImageGenerator(generators=[], media_path=str(tmp_media))


async def test_fallback_first_success_wins(tmp_media, sample_png):
    first = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    second = FakeImageGenerator(media_path=str(tmp_media))
    fb = FallbackImageGenerator(generators=[first, second], media_path=str(tmp_media))
    result = await fb.generate("a nice prompt", filename="fb.png")
    assert result.path.is_file()
    assert fb.last_backend == first.provider_name
    assert second.calls == 0


async def test_fallback_advances_on_failure(tmp_media, sample_png):
    err = ImageGenerationError("transient", retryable=True)
    first = FakeImageGenerator(results=[err, err, err], media_path=str(tmp_media),
                               retry_attempts=2, retry_delay_seconds=0)
    second = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    fb = FallbackImageGenerator(generators=[first, second], media_path=str(tmp_media))
    result = await fb.generate("a nice prompt", filename="fb.png")
    assert result.path.is_file()
    assert fb.last_backend == second.provider_name
    assert first.calls == 3  # full retry lifecycle ran inside the provider


async def test_fallback_all_fail(tmp_media):
    err = ImageGenerationError("boom", retryable=True)
    providers = [
        FakeImageGenerator(results=[err], media_path=str(tmp_media), retry_attempts=0),
        FakeImageGenerator(results=[err], media_path=str(tmp_media), retry_attempts=0),
    ]
    fb = FallbackImageGenerator(generators=providers, media_path=str(tmp_media))
    with pytest.raises(ImageGenerationError, match="All image providers failed"):
        await fb.generate("a nice prompt")