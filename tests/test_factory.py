"""Tests for the factory and opt-in fallback orchestration."""

from pathlib import Path

import pytest

from simple_chatbot_imaging.base import ImageGenerationError
from simple_chatbot_imaging.factory import (
    FallbackImageGenerator,
    create_image_generator,
    register_provider,
)

from conftest import FakeImageGenerator


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


def test_fallback_requires_chain(tmp_media):
    with pytest.raises(ValueError, match="non-empty"):
        FallbackImageGenerator(generators=[], media_path=str(tmp_media))


async def test_fallback_first_success_wins(tmp_media, sample_png):
    first = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    second = FakeImageGenerator(media_path=str(tmp_media))
    fb = FallbackImageGenerator(generators=[first, second], media_path=str(tmp_media))
    result = await fb.generate_image_async("a nice prompt", filename="fb.png")
    assert Path(result).is_file()
    assert fb.last_backend == first.provider_name
    assert second.calls == 0


async def test_fallback_advances_on_failure(tmp_media, sample_png):
    err = ImageGenerationError("transient", retryable=True)
    first = FakeImageGenerator(results=[err, err, err], media_path=str(tmp_media),
                               retry_attempts=2, retry_delay_seconds=0)
    second = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    fb = FallbackImageGenerator(generators=[first, second], media_path=str(tmp_media))
    result = await fb.generate_image_async("a nice prompt", filename="fb.png")
    assert Path(result).is_file()
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
        await fb.generate_image_async("a nice prompt")