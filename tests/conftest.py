"""Shared test helpers: a fake provider and fixtures."""

from pathlib import Path

import pytest

from simple_chatbot_imaging.base import (
    BaseImageGenerator,
    ImageGenerationError,
    ImageGenerationState,
)


class FakeImageGenerator(BaseImageGenerator):
    """Scriptable provider for tests."""

    def __init__(self, results=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.results = list(results or [])
        self.calls = 0

    async def _generate_once_async(self, prompt, negative_prompt, resolution,
                                   aspect_ratio, steps, seed) -> Path:
        self.calls += 1
        if self.results:
            item = self.results.pop(0)
        else:
            raise ImageGenerationError("no more scripted results")
        if isinstance(item, Exception):
            raise item
        return Path(item)


@pytest.fixture
def tmp_media(tmp_path: Path) -> Path:
    media = tmp_path / "media"
    media.mkdir()
    return media


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """A minimal valid PNG file."""
    png = tmp_path / "sample.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return png