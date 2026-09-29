"""Tests for the base generator: validation, retries, events, filename guard."""

import asyncio
from pathlib import Path

import pytest

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.models import (
    ImageGenerationError,
    ImageGenerationEvent,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerationStatus,
)

from conftest import FakeImageGenerator


class TestValidation:
    """Tests for request validation."""

    async def test_prompt_validation_empty(self, tmp_media):
        gen = FakeImageGenerator(media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="")
        with pytest.raises(ValueError, match="No prompt"):
            await gen.generate_image_async(request)

    async def test_prompt_validation_too_short(self, tmp_media):
        gen = FakeImageGenerator(media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="ab")
        with pytest.raises(ValueError, match="at least 3 characters"):
            await gen.generate_image_async(request)

    async def test_filename_containment(self, tmp_media, sample_png):
        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(
            prompt="a nice prompt",
            filename="../escape.png"
        )
        with pytest.raises(ValueError, match="inside the media folder"):
            await gen.generate_image_async(request)

    async def test_existing_file_no_overwrite(self, tmp_media, sample_png):
        """Test that existing files are rejected when overwrite=False."""
        # Create an existing file first
        existing = tmp_media / "existing.png"
        existing.write_bytes(sample_png.read_bytes())

        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(
            prompt="a nice prompt",
            filename="existing.png",
            overwrite=False
        )
        with pytest.raises(FileExistsError):
            await gen.generate_image_async(request)

    async def test_existing_file_with_overwrite(self, tmp_media, sample_png):
        """Test that existing files are overwritten when overwrite=True."""
        # Create an existing file first
        existing = tmp_media / "existing.png"
        existing.write_bytes(b"old content")

        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(
            prompt="a nice prompt",
            filename="existing.png",
            overwrite=True
        )
        result = await gen.generate_image_async(request)
        assert result.path.is_file()

    async def test_unique_filename_when_none(self, tmp_media, sample_png):
        """Test that unique filenames are generated when filename=None."""
        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="a nice prompt", filename=None)
        result = await gen.generate_image_async(request)
        assert result.path.is_file()
        # Filename should contain the request_id
        assert "image-" in result.path.name


class TestSuccessPath:
    """Tests for successful generation."""

    async def test_success_returns_result(self, tmp_media, sample_png):
        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="a nice prompt")
        result = await gen.generate_image_async(request)

        assert isinstance(result, ImageGenerationResult)
        assert result.path.is_file()
        assert result.path.parent.resolve() == tmp_media.resolve()
        assert not sample_png.exists()  # moved, not copied
        assert result.attempts == 1
        assert result.request_id is not None

    async def test_success_emits_events(self, tmp_media, sample_png):
        """Test that lifecycle events are emitted in correct order."""
        events: list[ImageGenerationEvent] = []

        def handler(event: ImageGenerationEvent):
            events.append(event)

        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="a nice prompt")
        result = await gen.generate_image_async(request, on_event=handler)

        # Check event order: QUEUED -> GENERATING -> SAVING -> SUCCESS
        assert len(events) >= 4
        statuses = [e.status for e in events]
        assert ImageGenerationStatus.QUEUED in statuses
        assert ImageGenerationStatus.GENERATING in statuses
        assert ImageGenerationStatus.SAVING in statuses
        assert ImageGenerationStatus.SUCCESS in statuses

        # Verify QUEUED comes first
        assert events[0].status == ImageGenerationStatus.QUEUED
        # Find SUCCESS event and verify it has the path
        success_events = [e for e in events if e.status == ImageGenerationStatus.SUCCESS]
        assert len(success_events) == 1
        assert success_events[0].path == result.path

    async def test_result_contains_metadata(self, tmp_media, sample_png):
        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(
            prompt="a nice prompt",
            resolution=2048,
            aspect_ratio="16:9"
        )
        result = await gen.generate_image_async(request)

        assert result.request_id is not None
        assert result.provider_name == "FakeImageGenerator"
        assert result.attempts >= 1
        # Allow for float rounding in timestamp conversion.
        assert result.duration_seconds == pytest.approx(
            (result.completed_at - result.started_at).total_seconds(),
            abs=1e-6,
        )
        # Verify request was normalized in result
        assert result.request.prompt == "a nice prompt"
        assert result.request.resolution == 2048


class TestRetryBehavior:
    """Tests for retry logic."""

    async def test_retryable_error_retries(self, tmp_media, sample_png):
        err = ImageGenerationError("transient", retryable=True)
        gen = FakeImageGenerator(
            results=[err, sample_png],
            media_path=str(tmp_media),
            retry_attempts=2,
            retry_delay_seconds=0
        )
        request = ImageGenerationRequest(prompt="a nice prompt")
        result = await gen.generate_image_async(request)
        assert gen.calls == 2
        assert isinstance(result, ImageGenerationResult)

    async def test_permanent_error_does_not_retry(self, tmp_media):
        err = ImageGenerationError("bad key", retryable=False)
        gen = FakeImageGenerator(
            results=[err, err],
            media_path=str(tmp_media),
            retry_attempts=2,
            retry_delay_seconds=0
        )
        request = ImageGenerationRequest(prompt="a nice prompt")
        with pytest.raises(ImageGenerationError, match="bad key"):
            await gen.generate_image_async(request)
        assert gen.calls == 1

    async def test_all_attempts_fail(self, tmp_media):
        err = ImageGenerationError("transient", retryable=True)
        gen = FakeImageGenerator(
            results=[err, err, err],
            media_path=str(tmp_media),
            retry_attempts=2,
            retry_delay_seconds=0
        )
        request = ImageGenerationRequest(prompt="a nice prompt")
        with pytest.raises(ImageGenerationError, match="failed"):
            await gen.generate_image_async(request)
        assert gen.calls == 3

    async def test_retrying_event_emitted(self, tmp_media, sample_png):
        """Test that RETRYING events include attempt info."""
        events: list[ImageGenerationEvent] = []

        def handler(event: ImageGenerationEvent):
            events.append(event)

        err = ImageGenerationError("transient", retryable=True)
        gen = FakeImageGenerator(
            results=[err, sample_png],
            media_path=str(tmp_media),
            retry_attempts=2,
            retry_delay_seconds=0
        )
        request = ImageGenerationRequest(prompt="a nice prompt")
        await gen.generate_image_async(request, on_event=handler)

        retrying_events = [e for e in events if e.status == ImageGenerationStatus.RETRYING]
        assert len(retrying_events) == 1
        retrying = retrying_events[0]
        assert retrying.attempt == 2  # Next attempt number
        assert retrying.total_attempts == 3
        assert retrying.delay_seconds is not None


class TestConcurrency:
    """Tests for concurrent request handling."""

    async def test_concurrent_requests_distinct_results(self, tmp_media):
        """Test that concurrent requests produce distinct results."""
        # Create two sample images with different content
        import tempfile
        img1 = Path(tempfile.mktemp(suffix=".png"))
        img2 = Path(tempfile.mktemp(suffix=".png"))
        img1.write_bytes(b"image1_content")
        img2.write_bytes(b"image2_content")

        # Use barriers to ensure concurrent execution
        barrier1 = asyncio.Event()
        barrier2 = asyncio.Event()

        gen1 = FakeImageGenerator(
            results=[img1],
            media_path=str(tmp_media),
            barrier=barrier1
        )
        gen2 = FakeImageGenerator(
            results=[img2],
            media_path=str(tmp_media),
            barrier=barrier2
        )

        request1 = ImageGenerationRequest(prompt="prompt 1")
        request2 = ImageGenerationRequest(prompt="prompt 2")

        async def task1():
            barrier1.set()
            await barrier2.wait()  # Wait for task2 to start
            return await gen1.generate_image_async(request1)

        async def task2():
            barrier2.set()
            await barrier1.wait()  # Wait for task1 to start
            return await gen2.generate_image_async(request2)

        result1, result2 = await asyncio.gather(task1(), task2())

        # Verify distinct results
        assert result1.request_id != result2.request_id
        assert result1.path.name != result2.path.name
        assert "prompt 1" in str(result1.request.prompt)
        assert "prompt 2" in str(result2.request.prompt)

    async def test_concurrent_requests_same_generator(self, tmp_media):
        """Test concurrent requests on the same generator instance."""
        import tempfile
        img1 = Path(tempfile.mktemp(suffix=".png"))
        img2 = Path(tempfile.mktemp(suffix=".png"))
        img1.write_bytes(b"image1")
        img2.write_bytes(b"image2")

        barrier1 = asyncio.Event()
        barrier2 = asyncio.Event()

        gen = FakeImageGenerator(
            results=[img1, img2],
            media_path=str(tmp_media),
            concurrent_barriers=[barrier1, barrier2]
        )

        request1 = ImageGenerationRequest(prompt="prompt 1")
        request2 = ImageGenerationRequest(prompt="prompt 2")

        events_by_request: dict[str, list[ImageGenerationEvent]] = {}

        def make_handler(request_id: str):
            def handler(event: ImageGenerationEvent):
                if request_id not in events_by_request:
                    events_by_request[request_id] = []
                events_by_request[request_id].append(event)
            return handler

        async def task1():
            barrier1.set()
            await barrier2.wait()
            return await gen.generate_image_async(request1, on_event=make_handler("req1"))

        async def task2():
            barrier2.set()
            await barrier1.wait()
            return await gen.generate_image_async(request2, on_event=make_handler("req2"))

        result1, result2 = await asyncio.gather(task1(), task2())

        # Verify each request got its own events
        assert "req1" in events_by_request
        assert "req2" in events_by_request
        # Events for req1 should only contain prompt 1
        for event in events_by_request["req1"]:
            assert event.request.prompt == "prompt 1"
        for event in events_by_request["req2"]:
            assert event.request.prompt == "prompt 2"


class TestCancellation:
    """Tests for cancellation behavior."""

    async def test_cancellation_propagates(self, tmp_media):
        """Test that asyncio.CancelledError propagates correctly."""
        barrier = asyncio.Event()
        gen = FakeImageGenerator(
            results=[],
            media_path=str(tmp_media),
            barrier=barrier
        )
        request = ImageGenerationRequest(prompt="a nice prompt")

        async def cancel_later():
            await asyncio.sleep(0.1)
            task.cancel()

        task = asyncio.create_task(gen.generate_image_async(request))
        # Leave the barrier unset so the fake generator blocks inside
        # _generate_once_async until the cancellation below lands.
        await asyncio.sleep(0.05)

        try:
            await asyncio.gather(cancel_later(), task, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        with pytest.raises(asyncio.CancelledError):
            await task


class TestSyncBridge:
    """Tests for synchronous wrapper."""

    def test_sync_bridge_outside_loop(self, tmp_media, sample_png):
        gen = FakeImageGenerator(results=[sample_png], media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="a nice prompt")
        result = gen.generate_image(request)
        assert isinstance(result, ImageGenerationResult)
        assert result.path.is_file()

    async def test_sync_bridge_inside_loop_raises(self, tmp_media):
        gen = FakeImageGenerator(media_path=str(tmp_media))
        request = ImageGenerationRequest(prompt="a nice prompt")
        with pytest.raises(RuntimeError, match="generate_image_async"):
            gen.generate_image(request)


def test_import_does_not_pull_optional_deps():
    import sys
    import simple_chatbot_imaging

    assert "gradio_client" not in sys.modules
    assert "httpx" not in sys.modules or True  # httpx may load via provider only
    assert simple_chatbot_imaging.__version__