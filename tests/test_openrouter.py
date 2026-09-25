"""Tests for the OpenRouter provider with a mocked httpx transport."""

import base64
import os
from pathlib import Path

import httpx
import pytest

from simple_chatbot_imaging.base import ImageGenerationError
from simple_chatbot_imaging.openrouter import OpenRouterImageGenerator


def _b64_png() -> bytes:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def _make_client(monkeypatch, handler, **gen_kwargs):
    gen = OpenRouterImageGenerator(**gen_kwargs)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient(transport=transport)

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url, **kwargs)
        return await real_client.send(request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return gen


async def test_missing_api_key(tmp_media, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gen = OpenRouterImageGenerator(media_path=str(tmp_media))
    with pytest.raises(ImageGenerationError, match="Missing API key"):
        await gen.generate_image_async("a nice prompt")


async def test_success_png(tmp_media, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/generations")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"data": [{"b64_json": _b64_png().decode()}]})

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media))
    result = await gen.generate_image_async("a nice prompt", filename="or.png")
    p = Path(result)
    assert p.is_file()
    assert p.read_bytes().startswith(b"\x89PNG")
    assert p.parent.resolve() == tmp_media.resolve()


async def test_data_url_prefix_stripped(tmp_media, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"image": f"data:image/png;base64,{_b64_png().decode()}"}]}
        )

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media))
    result = await gen.generate_image_async("a nice prompt")
    assert Path(result).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("status,match", [
    (401, "authentication failed"),
    (402, "credit limit"),
    (403, "forbidden"),
    (404, "model not found"),
])
async def test_permanent_errors_no_retry(tmp_media, monkeypatch, status, match):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="denied")

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media),
                       retry_attempts=3, retry_delay_seconds=0)
    with pytest.raises(ImageGenerationError, match=match):
        await gen.generate_image_async("a nice prompt")


async def test_retryable_429_retries_then_fails(tmp_media, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="slow down")

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media),
                       retry_attempts=2, retry_delay_seconds=0)
    with pytest.raises(ImageGenerationError, match="rate/quota"):
        await gen.generate_image_async("a nice prompt")
    assert calls["n"] == 3


async def test_bad_response_format(tmp_media, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media))
    with pytest.raises(ImageGenerationError, match="Unexpected OpenRouter response"):
        await gen.generate_image_async("a nice prompt")


async def test_invalid_base64(tmp_media, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": "!!!not-base64!!!"}]})

    gen = _make_client(monkeypatch, handler, media_path=str(tmp_media))
    with pytest.raises(ImageGenerationError, match="invalid base64"):
        await gen.generate_image_async("a nice prompt")