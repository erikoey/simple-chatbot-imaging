"""Tests for cost reporting: models, extraction, provider wiring, fallback."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from conftest import FakeImageGenerator
from simple_chatbot_imaging.models import (
    ImageGenerationAttempt,
    ImageGenerationCost,
    ImageGenerationRequest,
    ImageGenerationStatus,
)
from simple_chatbot_imaging.providers.http_base import BaseHTTPImageGenerator
from simple_chatbot_imaging.providers.huggingface import HuggingFaceImageGenerator
from simple_chatbot_imaging.providers.openrouter import OpenRouterImageGenerator

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def _b64_png() -> str:
    import base64
    return base64.b64encode(PNG_BYTES).decode()


class SimpleHTTPGenerator(BaseHTTPImageGenerator):
    provider_label = "TestHTTP"

    def _build_payload(self, request: ImageGenerationRequest) -> dict:
        return {"model": self.model, "prompt": request.prompt}


# --------------------------------------------------------------------- #
# Models                                                                 #
# --------------------------------------------------------------------- #


def test_cost_defaults_all_none():
    cost = ImageGenerationCost()
    assert cost.total_credits is None
    assert cost.total_amount is None
    assert cost.currency is None
    assert cost.total_tokens is None
    assert cost.image_count is None
    assert cost.per_image_credits is None
    assert cost.per_image_amounts is None
    assert cost.per_image_tokens is None
    assert cost.raw_usage is None
    assert cost.is_zero is False


def test_cost_zero_factory():
    cost = ImageGenerationCost.zero()
    assert cost.total_credits == 0.0
    assert cost.total_amount == 0.0
    assert cost.total_tokens == 0
    assert cost.is_zero is True
    with_count = ImageGenerationCost.zero(image_count=2)
    assert with_count.image_count == 2
    assert with_count.is_zero is True


def test_cost_is_zero_requires_reported_values():
    assert ImageGenerationCost(image_count=3).is_zero is False
    assert ImageGenerationCost(total_credits=0.5).is_zero is False
    assert ImageGenerationCost(total_credits=0.0, total_amount=0.0).is_zero is True


def test_result_and_event_accept_cost_fields():
    request = ImageGenerationRequest(prompt="a nice prompt")
    now = datetime.now(UTC)
    cost = ImageGenerationCost(total_amount=0.01, currency="USD")

    from simple_chatbot_imaging.models import ImageGenerationEvent, ImageGenerationResult

    result = ImageGenerationResult(
        request_id="r1", path=Path("x.png"), provider_name="T", model="m",
        request=request, attempts=1, started_at=now, completed_at=now,
        duration_seconds=1.0, cost=cost, raw_response_json='{"usage": {}}',
    )
    assert result.cost is cost
    assert result.raw_response_json == '{"usage": {}}'

    event = ImageGenerationEvent(
        request_id="r1", status=ImageGenerationStatus.SUCCESS, provider_name="T",
        request=request, occurred_at=now, path=Path("x.png"),
        cost=cost, raw_response_json='{"usage": {}}',
    )
    assert event.cost is cost
    assert event.raw_response_json == '{"usage": {}}'


# --------------------------------------------------------------------- #
# Base class normalization                                               #
# --------------------------------------------------------------------- #


async def test_bare_path_result_has_no_cost(tmp_media, sample_png):
    gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    result = await gen.generate_image_async(ImageGenerationRequest(prompt="a nice prompt"))
    assert result.cost is None
    assert result.raw_response_json is None


async def test_attempt_result_carries_cost_and_raw_json(tmp_media, sample_png):
    cost = ImageGenerationCost(total_credits=3.5, image_count=1)
    scripted = ImageGenerationAttempt(path=sample_png, cost=cost, raw_response_json='{"a": 1}')
    gen = FakeImageGenerator(results=[scripted], media_path=str(tmp_media))
    result = await gen.generate_image_async(ImageGenerationRequest(prompt="a nice prompt"))
    assert result.cost is cost
    assert result.raw_response_json == '{"a": 1}'


async def test_success_event_carries_cost(tmp_media, sample_png):
    cost = ImageGenerationCost(total_credits=3.5)
    scripted = ImageGenerationAttempt(path=sample_png, cost=cost)
    gen = FakeImageGenerator(results=[scripted], media_path=str(tmp_media))
    events = []
    await gen.generate_image_async(
        ImageGenerationRequest(prompt="a nice prompt"),
        on_event=events.append,
    )
    success = [e for e in events if e.status == ImageGenerationStatus.SUCCESS]
    assert len(success) == 1
    assert success[0].cost is cost


# --------------------------------------------------------------------- #
# Generic HTTP cost extraction                                           #
# --------------------------------------------------------------------- #


def test_extract_cost_openrouter_style_usage():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {
        "data": [{"b64_json": _b64_png()}],
        "usage": {"cost": 0.04, "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    cost = gen._extract_cost(parsed)
    assert cost is not None
    assert cost.total_amount == 0.04
    assert cost.total_tokens == 11
    assert cost.total_credits is None
    assert cost.raw_usage["cost"] == 0.04


def test_extract_cost_credits_and_currency():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"usage": {"credits_consumed": 2.5, "amount": 0.1, "currency": "EUR"}}
    cost = gen._extract_cost(parsed)
    assert cost.total_credits == 2.5
    assert cost.total_amount == 0.1
    assert cost.currency == "EUR"


def test_extract_cost_image_count_only():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"usage": {"image_count": 2}}
    cost = gen._extract_cost(parsed)
    assert cost is not None
    assert cost.image_count == 2
    assert cost.total_amount is None
    assert cost.total_credits is None


def test_extract_cost_no_usage_returns_none():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    assert gen._extract_cost({"data": [{"b64_json": _b64_png()}]}) is None


def test_extract_cost_usage_without_cost_keys_returns_none():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    assert gen._extract_cost({"usage": {"prompt_tokens": 10, "completion_tokens": 2}}) is None


def test_extract_cost_finds_nested_usage():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"output": {"meta": {"usage": {"cost": 0.5}}}}
    cost = gen._extract_cost(parsed)
    assert cost is not None
    assert cost.total_amount == 0.5


def test_extract_per_image_parallel_lists():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"usage": {
        "cost": 0.06,
        "per_image_costs": [0.03, 0.03],
        "per_image_tokens": [5, 6],
    }}
    cost = gen._extract_cost(parsed)
    assert cost.per_image_amounts == (0.03, 0.03)
    assert cost.per_image_tokens == (5, 6)
    assert cost.per_image_credits is None


def test_extract_per_image_entry_list():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"usage": {
        "credits": 4.0,
        "images": [{"credits": 2.0}, {"credits": 2.0}],
    }}
    cost = gen._extract_cost(parsed)
    assert cost.total_credits == 4.0
    assert cost.per_image_credits == (2.0, 2.0)


def test_extract_per_image_mixed_entries_skip_none():
    gen = SimpleHTTPGenerator(media_path="media_test_http")
    parsed = {"usage": {
        "tokens": 11,
        "data": [{"total_tokens": 5}, {"other": 1}, {"total_tokens": 6}],
    }}
    cost = gen._extract_cost(parsed)
    assert cost.total_tokens == 11
    assert cost.per_image_tokens == (5, None, 6)


async def test_http_pipeline_returns_attempt_with_cost(tmp_media, monkeypatch):
    monkeypatch.setenv("TEST_HTTP_MISSING_KEY_X", "unused")

    class NoAuthGenerator(SimpleHTTPGenerator):
        def _build_headers(self) -> dict[str, str]:
            return {"Content-Type": "application/json"}

    gen = NoAuthGenerator(media_path=str(tmp_media))
    body = {
        "data": [{"b64_json": _b64_png()}],
        "usage": {"cost": 0.02, "total_tokens": 7, "image_count": 1},
    }

    async def fake_post(url, payload):
        return httpx.Response(
            200, json=body,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(gen, "_post_json", fake_post)
    attempt = await gen._generate_once_async(
        ImageGenerationRequest(prompt="a nice prompt")
    )
    assert attempt.path.is_file()
    assert attempt.cost.total_amount == 0.02
    assert attempt.cost.total_tokens == 7
    assert attempt.cost.image_count == 1
    assert json.loads(attempt.raw_response_json) == body


# --------------------------------------------------------------------- #
# OpenRouter end-to-end                                                  #
# --------------------------------------------------------------------- #


async def _run_openrouter(monkeypatch, tmp_media, body):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gen = OpenRouterImageGenerator(media_path=str(tmp_media))

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    real_client = httpx.AsyncClient(transport=transport)

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url, **kwargs)
        return await real_client.send(request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = await gen.generate_image_async(
        ImageGenerationRequest(prompt="a nice prompt")
    )
    await real_client.aclose()
    return result


async def test_openrouter_cost_in_usd_from_usage(tmp_media, monkeypatch):
    body = {
        "data": [{"b64_json": _b64_png()}],
        "usage": {"cost": 0.04, "total_tokens": 11},
    }
    result = await _run_openrouter(monkeypatch, tmp_media, body)
    assert result.cost is not None
    assert result.cost.total_amount == 0.04
    assert result.cost.currency == "USD"
    assert result.cost.total_tokens == 11
    assert json.loads(result.raw_response_json) == body


async def test_openrouter_explicit_currency_not_overridden(tmp_media, monkeypatch):
    body = {"data": [{"b64_json": _b64_png()}], "usage": {"cost": 1.0, "currency": "EUR"}}
    result = await _run_openrouter(monkeypatch, tmp_media, body)
    assert result.cost.currency == "EUR"


async def test_openrouter_no_usage_cost_is_none(tmp_media, monkeypatch):
    body = {"data": [{"b64_json": _b64_png()}]}
    result = await _run_openrouter(monkeypatch, tmp_media, body)
    assert result.cost is None
    assert json.loads(result.raw_response_json) == body


# --------------------------------------------------------------------- #
# Hugging Face: always zero                                              #
# --------------------------------------------------------------------- #


class _FakeJob:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def done(self):
        return True

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True


class _FakeGradioClient:
    def __init__(self, result):
        self._result = result

    def submit(self, **kwargs):
        return _FakeJob(self._result)


async def test_huggingface_always_reports_zero_cost(tmp_path):
    image_file = tmp_path / "out.png"
    image_file.write_bytes(PNG_BYTES)

    gen = HuggingFaceImageGenerator(media_path=str(tmp_path / "media"))
    gen._client = _FakeGradioClient(str(image_file))

    attempt = await gen._generate_once_async(
        ImageGenerationRequest(prompt="a nice prompt")
    )
    assert attempt.path == image_file
    assert attempt.cost is not None
    assert attempt.cost.is_zero is True
    assert attempt.cost.image_count == 1


# --------------------------------------------------------------------- #
# Fallback propagates cost                                               #
# --------------------------------------------------------------------- #


async def test_fallback_propagates_cost(tmp_media, sample_png):
    from simple_chatbot_imaging.factory import FallbackImageGenerator

    cost = ImageGenerationCost(total_credits=1.5, image_count=1)
    failing = FakeImageGenerator(
        results=[RuntimeError("boom")], media_path=str(tmp_media)
    )
    winning = FakeImageGenerator(
        results=[ImageGenerationAttempt(path=sample_png, cost=cost)],
        media_path=str(tmp_media),
    )
    fb = FallbackImageGenerator(generators=[failing, winning], media_path=str(tmp_media))
    result = await fb.generate_image_async(
        ImageGenerationRequest(prompt="a nice prompt")
    )
    assert fb.last_backend == winning.provider_name
    assert result.cost is cost
