"""Tests for the shared HTTP provider base class."""

import base64

import httpx
import pytest

from simple_chatbot_imaging.models import ImageGenerationError, ImageGenerationRequest
from simple_chatbot_imaging.providers.http_base import BaseHTTPImageGenerator


class SimpleHTTPGenerator(BaseHTTPImageGenerator):
    """Minimal concrete subclass for exercising the shared machinery."""

    provider_label = "TestHTTP"

    def _build_payload(self, request: ImageGenerationRequest) -> dict:
        return {"model": self.model, "prompt": request.prompt}


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def _b64_png() -> str:
    return base64.b64encode(PNG_BYTES).decode()


def _response(*, json_body=None, content=None, headers=None, status_code=200) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers=headers or {},
        json=json_body,
        content=content,
        request=httpx.Request("POST", "https://api.example.com/v1/images"),
    )


def test_api_key_missing_raises():
    gen = SimpleHTTPGenerator(api_key_env="TEST_HTTP_MISSING_KEY", media_path="media_test_http")
    with pytest.raises(ImageGenerationError, match="Missing API key"):
        _ = gen.api_key


def test_api_key_is_secretstr(monkeypatch):
    monkeypatch.setenv("TEST_HTTP_MISSING_KEY", "secret-value")
    gen = SimpleHTTPGenerator(api_key_env="TEST_HTTP_MISSING_KEY", media_path="media_test_http")
    key = gen.api_key
    assert key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(key)


def test_model_name_reflects_model():
    gen = SimpleHTTPGenerator(model="my-model", media_path="media_test_http")
    assert gen.model_name == "my-model"


@pytest.mark.parametrize("status,match,retryable", [
    (401, "authentication failed", False),
    (402, "credit limit", False),
    (403, "forbidden", False),
    (404, "model not found", False),
    (429, "rate/quota", True),
    (500, "server error", True),
    (503, "server error", True),
    (418, "request failed", False),
])
def test_raise_for_status_mapping(status, match, retryable):
    gen = SimpleHTTPGenerator(model="my-model", media_path="media_test_http")
    response = _response(status_code=status, json_body={})
    with pytest.raises(ImageGenerationError, match=match) as excinfo:
        gen._raise_for_status(response)
    assert excinfo.value.retryable is retryable


def test_404_includes_endpoint_and_provider_response():
    gen = SimpleHTTPGenerator(model="my-model", media_path="media_test_http")
    request = httpx.Request("POST", "https://api.example.com/wrong-route")
    response = httpx.Response(404, text="route does not exist", request=request)

    with pytest.raises(ImageGenerationError, match="model not found or endpoint unavailable") as excinfo:
        gen._raise_for_status(response)

    assert "https://api.example.com/wrong-route" in str(excinfo.value)
    assert "route does not exist" in str(excinfo.value)


def test_sniff_suffix_from_content_type():
    assert BaseHTTPImageGenerator._sniff_suffix(b"", "image/jpeg; charset=binary") == ".jpg"
    assert BaseHTTPImageGenerator._sniff_suffix(b"", "image/png") == ".png"
    assert BaseHTTPImageGenerator._sniff_suffix(b"", "image/webp") == ".webp"


def test_sniff_suffix_from_signature():
    assert BaseHTTPImageGenerator._sniff_suffix(b"\xff\xd8\xffrest") == ".jpg"
    assert BaseHTTPImageGenerator._sniff_suffix(PNG_BYTES) == ".png"
    riff = b"RIFF" + b"0" * 4 + b"WEBP"
    assert BaseHTTPImageGenerator._sniff_suffix(riff) == ".webp"
    assert BaseHTTPImageGenerator._sniff_suffix(b"unknown") == ".png"


async def test_extract_image_bytes_inline_body():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    response = _response(content=PNG_BYTES, headers={"content-type": "image/png"})
    assert await gen._extract_image_bytes({}, response) == PNG_BYTES


async def test_extract_image_bytes_b64_field():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    response = _response(json_body={"data": [{"b64_json": _b64_png()}]})
    assert await gen._extract_image_bytes(response.json(), response) == PNG_BYTES


async def test_extract_image_bytes_data_url_stripped():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    response = _response(
        json_body={"data": [{"image": f"data:image/png;base64,{_b64_png()}"}]}
    )
    assert await gen._extract_image_bytes(response.json(), response) == PNG_BYTES


async def test_extract_image_bytes_invalid_base64():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    response = _response(json_body={"data": [{"b64_json": "!!!not-base64!!!"}]})
    with pytest.raises(ImageGenerationError, match="invalid base64"):
        await gen._extract_image_bytes(response.json(), response)


async def test_extract_image_bytes_url_field_downloads(monkeypatch):
    gen = SimpleHTTPGenerator(media_path="media_test_http")

    async def fake_download(url):
        assert url == "https://cdn.example.com/image.png"
        return PNG_BYTES, "image/png"

    async def fail_download(url):
        raise AssertionError("should not be called")

    monkeypatch.setattr(gen, "_download_image", fake_download)
    response = _response(
        json_body={"output": {"choices": [{"message": {"content": [
            {"image": "https://cdn.example.com/image.png"}
        ]}}]}}
    )
    assert await gen._extract_image_bytes(response.json(), response) == PNG_BYTES

    # Base64 wins over URL when both appear (resolution order).
    monkeypatch.setattr(gen, "_download_image", fail_download)
    response = _response(
        json_body={"data": [{"b64_json": _b64_png(), "url": "https://ignored.com"}]}
    )
    assert await gen._extract_image_bytes(response.json(), response) == PNG_BYTES


async def test_extract_image_bytes_nothing_usable():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    response = _response(json_body={"unexpected": True})
    with pytest.raises(ImageGenerationError, match="Unexpected TestHTTP response"):
        await gen._extract_image_bytes(response.json(), response)


def test_find_first_walks_nested_structures():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    payload = {
        "output": {
            "choices": [
                {"message": {"content": [{"text": "hi"}, {"image": "https://x/i.png"}]}}
            ]
        }
    }
    assert gen._find_first(payload, gen.URL_FIELDS) == "https://x/i.png"
    b64 = base64.b64encode(PNG_BYTES).decode()
    assert gen._find_first({"a": {"b64": b64}}, ("b64",)) == b64
    assert gen._find_first({"a": {"url": "notaurl"}}, gen.URL_FIELDS) is None


def test_write_temp_image_rejects_empty():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    with pytest.raises(ImageGenerationError, match="no image data"):
        gen._write_temp_image(b"", ".png", "test_")


def test_write_temp_image_roundtrip():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    path = gen._write_temp_image(PNG_BYTES, ".png", "test_")
    try:
        assert path.is_file()
        assert path.name.startswith("test_")
        assert path.read_bytes() == PNG_BYTES
    finally:
        path.unlink(missing_ok=True)


def test_extend_field_tuples(monkeypatch):
    class CustomGenerator(SimpleHTTPGenerator):
        BASE64_FIELDS = ("my_b64",) + SimpleHTTPGenerator.BASE64_FIELDS

    gen = CustomGenerator(media_path="media_test_http")
    response = _response(json_body={"data": [{"my_b64": _b64_png()}]})
    import asyncio
    assert asyncio.run(gen._extract_image_bytes(response.json(), response)) == PNG_BYTES
