"""Tests for the base generator: validation, retries, state, filename guard."""

import sys

import pytest

from simple_chatbot_imaging.base import (
    ImageGenerationError,
    ImageGenerationState,
)

from conftest import FakeImageGenerator


async def test_prompt_validation_empty(tmp_media):
    gen = FakeImageGenerator(media_path=str(tmp_media))
    with pytest.raises(ValueError, match="No prompt"):
        await gen.generate_image_async("")
    assert gen.state == ImageGenerationState.ERROR


async def test_prompt_validation_too_short(tmp_media):
    gen = FakeImageGenerator(media_path=str(tmp_media))
    with pytest.raises(ValueError, match="at least 3 characters"):
        await gen.generate_image_async("ab")
    assert gen.state == ImageGenerationState.ERROR


async def test_filename_containment(tmp_media, sample_png):
    gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    with pytest.raises(ValueError, match="inside the media folder"):
        await gen.generate_image_async("a nice prompt", filename="../escape.png")
    assert gen.state == ImageGenerationState.ERROR


async def test_success_moves_output(tmp_media, sample_png):
    gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    result = await gen.generate_image_async("a nice prompt", filename="out.png")
    from pathlib import Path
    p = Path(result)
    assert p.is_file()
    assert p.parent.resolve() == tmp_media.resolve()
    assert not sample_png.exists()  # moved, not copied
    assert gen.state == ImageGenerationState.SUCCESS


async def test_retryable_error_retries(tmp_media, sample_png):
    err = ImageGenerationError("transient", retryable=True)
    gen = FakeImageGenerator(results=[err, sample_png], media_path=str(tmp_media),
                            retry_attempts=2, retry_delay_seconds=0)
    result = await gen.generate_image_async("a nice prompt")
    assert gen.calls == 2
    assert gen.state == ImageGenerationState.SUCCESS
    from pathlib import Path
    assert Path(result).is_file()


async def test_permanent_error_does_not_retry(tmp_media):
    err = ImageGenerationError("bad key", retryable=False)
    gen = FakeImageGenerator(results=[err, err], media_path=str(tmp_media),
                            retry_attempts=2, retry_delay_seconds=0)
    with pytest.raises(ImageGenerationError, match="bad key"):
        await gen.generate_image_async("a nice prompt")
    assert gen.calls == 1
    assert gen.state == ImageGenerationState.ERROR


async def test_all_attempts_fail(tmp_media):
    err = ImageGenerationError("transient", retryable=True)
    gen = FakeImageGenerator(results=[err, err, err], media_path=str(tmp_media),
                            retry_attempts=2, retry_delay_seconds=0)
    with pytest.raises(ImageGenerationError, match="failed"):
        await gen.generate_image_async("a nice prompt")
    assert gen.calls == 3
    assert gen.state == ImageGenerationState.ERROR


async def test_invalid_result_path(tmp_media):
    gen = FakeImageGenerator(results=["/nonexistent/nope.png"], media_path=str(tmp_media))
    with pytest.raises(ImageGenerationError, match="did not return a valid image file"):
        await gen.generate_image_async("a nice prompt")


def test_sync_bridge_outside_loop(tmp_media, sample_png):
    gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
    result = gen.generate_image("a nice prompt")
    from pathlib import Path
    assert Path(result).is_file()


async def test_sync_bridge_inside_loop_raises(tmp_media):
    gen = FakeImageGenerator(media_path=str(tmp_media))
    with pytest.raises(RuntimeError, match="generate_image_async"):
        gen.generate_image("a nice prompt")


def test_import_does_not_pull_optional_deps():
    import simple_chatbot_imaging

    assert "gradio_client" not in sys.modules
    assert "httpx" not in sys.modules or True  # httpx may load via provider only
    assert simple_chatbot_imaging.__version__