"""Shared test helpers: a fake provider and fixtures."""

import asyncio
from pathlib import Path

import pytest

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.models import (
    ImageGenerationAttempt,
    ImageGenerationError,
    ImageGenerationRequest,
)


class FakeImageGenerator(BaseImageGenerator):
    """Scriptable provider for tests."""

    def __init__(
        self,
        results=None,
        barrier: asyncio.Event | None = None,
        concurrent_barriers: list[asyncio.Event] | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.results = list(results or [])
        self.calls = 0
        self.barrier = barrier
        self.concurrent_barriers = concurrent_barriers or []

    async def _generate_once_async(
        self,
        request: ImageGenerationRequest,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        resolution: int | None = None,
        aspect_ratio: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        filename: str | None = None,
    ) -> Path:
        """Generate an image from the request.

        Accepts both positional (ImageGenerationRequest) and keyword arguments
        to support calls from FallbackImageGenerator which passes kwargs directly.
        """
        self.calls += 1

        # Support concurrency testing with barriers
        if self.barrier:
            await self.barrier.wait()

        # For concurrent tests, use per-call barriers
        if self.concurrent_barriers and self.calls <= len(self.concurrent_barriers):
            barrier = self.concurrent_barriers[self.calls - 1]
            await barrier.wait()

        if self.results:
            item = self.results.pop(0)
        else:
            raise ImageGenerationError("no more scripted results")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ImageGenerationAttempt):
            return item
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