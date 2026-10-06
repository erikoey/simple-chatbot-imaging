"""Tests for the local SDXL provider: dimension math, lazy import, cost."""

import sys
import types

import pytest

from simple_chatbot_imaging.factory import PROVIDERS, create_image_generator
from simple_chatbot_imaging.models import (
    ImageGenerationError,
    ImageGenerationRequest,
)
from simple_chatbot_imaging.providers.sdxl import (
    SdxlImageGenerator,
    _resolution_to_dimensions,
)


class _FakeImage:
    """Duck-typed PIL image recording where it was saved."""

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save(self, path) -> None:
        self.saved_to = str(path)
        with open(path, "wb") as fh:
            fh.write(b"png")


@pytest.fixture
def fake_sdxl(monkeypatch):
    """Install a stub sdxl_generator module that captures generate() calls."""
    calls: dict = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return _FakeImage()

    stub = types.ModuleType("sdxl_generator")
    stub.generate = fake_generate
    monkeypatch.setitem(sys.modules, "sdxl_generator", stub)
    return calls


@pytest.mark.parametrize(
    ("resolution", "aspect_ratio", "expected"),
    [
        (1024, "1:1", (1024, 1024)),
        (1024, "2:3", (680, 1024)),
        (1024, "16:9", (1024, 576)),
        (512, "5:4", (512, 408)),
    ],
)
def test_resolution_to_dimensions(resolution, aspect_ratio, expected):
    width, height = _resolution_to_dimensions(resolution, aspect_ratio)
    assert (width, height) == expected
    assert width % 8 == 0 and height % 8 == 0


def test_resolution_to_dimensions_rejects_bad_ratio():
    with pytest.raises(ValueError, match="aspect_ratio"):
        _resolution_to_dimensions(1024, "not-a-ratio")
    with pytest.raises(ValueError, match="aspect_ratio"):
        _resolution_to_dimensions(1024, "0:1")


async def test_generate_local_runs_stub_and_reports_free(tmp_path, fake_sdxl):
    gen = SdxlImageGenerator(media_path=str(tmp_path))
    result = await gen.generate_image_async(
        ImageGenerationRequest(prompt="a fox resting in the snow", seed=42, steps=5)
    )

    assert result.path.is_file()
    assert result.path.is_relative_to(tmp_path.resolve())
    assert result.cost is not None
    assert result.cost.total_credits == 0
    assert result.cost.total_amount == 0
    assert result.cost.image_count == 1

    assert fake_sdxl["seed"] == 42
    assert fake_sdxl["steps"] == 5
    assert fake_sdxl["width"] == 1024
    assert fake_sdxl["height"] == 1024
    assert fake_sdxl["model_id"] == SdxlImageGenerator.DEFAULT_MODEL_ID
    assert fake_sdxl["vae_id"] == SdxlImageGenerator.DEFAULT_VAE_ID


async def test_missing_extra_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sdxl_generator", None)
    gen = SdxlImageGenerator(media_path=str(tmp_path))
    with pytest.raises(ImageGenerationError, match="'sdxl' extra"):
        await gen.generate_image_async(
            ImageGenerationRequest(prompt="a fox resting in the snow")
        )


def test_factory_registers_sdxl():
    assert "sdxl" in PROVIDERS


def test_factory_constructs_without_gpu_stack(tmp_path):
    """Construction imports no torch/sdxl_generator; use triggers that."""
    assert "torch" not in sys.modules
    assert "sdxl_generator" not in sys.modules

    gen = create_image_generator("sdxl", media_path=str(tmp_path))
    assert gen.model_id == SdxlImageGenerator.DEFAULT_MODEL_ID
    assert "torch" not in sys.modules


def test_factory_applies_bundled_defaults(tmp_path):
    """providers.json / built-in defaults reach the constructor kwargs."""
    gen = create_image_generator("sdxl", media_path=str(tmp_path))
    assert gen.device == "cuda:0"
    assert gen.steps == 7
    assert gen.guidance_scale == 1.5


def test_constructor_defaults_and_vae_none(tmp_path):
    gen = SdxlImageGenerator(media_path=str(tmp_path))
    assert gen.model_id == SdxlImageGenerator.DEFAULT_MODEL_ID
    assert gen.vae_id == SdxlImageGenerator.DEFAULT_VAE_ID
    assert gen.steps == 7
    assert gen.guidance_scale == 1.5

    bundled = SdxlImageGenerator(media_path=str(tmp_path), vae=None)
    assert bundled.vae_id is None


async def test_generate_local_forwards_vae_none(tmp_path, fake_sdxl):
    """vae=None means the checkpoint's baked-in VAE, not the default."""
    gen = SdxlImageGenerator(media_path=str(tmp_path), vae=None)
    result = await gen.generate_image_async(
        ImageGenerationRequest(prompt="a fox resting in the snow")
    )
    assert result.path.is_file()
    assert fake_sdxl["vae_id"] is None


def test_prewarm_uses_tiny_image_and_cleans_tmp(tmp_path, fake_sdxl):
    gen = SdxlImageGenerator(media_path=str(tmp_path))
    assert gen.prewarm() is True
    assert fake_sdxl["width"] == 8
    assert fake_sdxl["height"] == 8
    assert fake_sdxl["steps"] == 1
    assert fake_sdxl["seed"] == 0
    assert list((tmp_path / ".sdxl_tmp").iterdir()) == []


def test_prewarm_failure_is_swallowed(tmp_path, monkeypatch):
    stub = types.ModuleType("sdxl_generator")

    def boom(**kwargs):
        raise RuntimeError("no GPU")

    stub.generate = boom
    monkeypatch.setitem(sys.modules, "sdxl_generator", stub)

    gen = SdxlImageGenerator(media_path=str(tmp_path))
    assert gen.prewarm() is False


def test_release_delegates_to_engine_cache(tmp_path, monkeypatch):
    released = []
    stub = types.ModuleType("sdxl_generator")
    stub.generate = lambda **kwargs: _FakeImage()
    stub.release_engine = lambda: released.append(True)
    monkeypatch.setitem(sys.modules, "sdxl_generator", stub)

    gen = SdxlImageGenerator(media_path=str(tmp_path))
    gen.release()
    from simple_chatbot_imaging.providers.sdxl import release_engine
    release_engine()
    assert released == [True, True]
