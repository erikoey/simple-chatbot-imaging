"""Base classes for image generation providers.

This module only depends on the standard library plus ``httpx``-free
utilities; it defines the true async provider contract, retry semantics,
lifecycle events, output movement, and a synchronous bridge.
"""

import asyncio
import inspect
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Awaitable, Callable

global_logger = logging.getLogger(__name__)

from simple_chatbot_imaging.models import (
    ImageGenerationError,
    ImageGenerationEvent,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerationStatus,
    generate_unique_filename,
)


class BaseImageGenerator(ABC):
    """Abstract base class for image generation providers.

    Subclasses implement :meth:`_generate_once_async`, which performs a
    single provider call and returns the path to a locally available image
    file. The base class handles request validation, lifecycle events,
    retries with backoff, timeouts, and moving the result into the media folder.
    
    This generator is stateless: each generate_image_async() invocation owns
    its own request lifecycle and returns an ImageGenerationResult containing
    output and execution metadata. Operational failures raise ImageGenerationError.
    """

    def __init__(self,
                 media_path: str = "media",
                 timeout_seconds: int = 600,
                 retry_attempts: int = 2,
                 retry_delay_seconds: float = 2.0) -> None:
        self.media_path = Path(media_path)
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout_seconds
        self.retry_attempts = max(0, retry_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)


    @property
    def provider_name(self) -> str:
        """Human-readable provider name, used for logging."""
        return type(self).__name__

    @property
    def model_name(self) -> str | None:
        """Model name if known at the base class level; subclasses may override."""
        return None

    def _validate_request(self, request: ImageGenerationRequest) -> ImageGenerationRequest:
        """Validate and normalize a request; raises ValueError when invalid.

        Returns a normalized replacement using dataclasses.replace.
        """
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("No prompt given to generate an image")

        prompt = request.prompt.strip()
        if len(prompt) < 3:
            raise ValueError("Provide an image prompt with at least 3 characters.")

        # Normalize negative_prompt
        if request.negative_prompt is None:
            negative_prompt = ""
        elif not isinstance(request.negative_prompt, str):
            raise ValueError("negative_prompt must be a string or None")
        else:
            negative_prompt = request.negative_prompt.strip()

        # Validate numeric parameters
        if request.resolution <= 0:
            raise ValueError("resolution must be positive")
        if request.steps is not None and request.steps <= 0:
            raise ValueError("steps must be positive when provided")
        if request.seed is not None and request.seed < 0:
            raise ValueError("seed must be non-negative when provided")

        # Build normalized request
        normalized = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=request.resolution,
            aspect_ratio=request.aspect_ratio,
            steps=request.steps,
            seed=request.seed,
            filename=request.filename,
            overwrite=request.overwrite,
        )
        return normalized

    def _target_path(self, filename: str) -> Path:
        """Resolve the target path and ensure it stays inside the media folder."""
        target = (self.media_path / filename).resolve()
        if not target.is_relative_to(self.media_path.resolve()):
            raise ValueError(
                f"Filename {filename!r} must stay inside the media folder."
            )
        return target

    def _build_target_path(self, request: ImageGenerationRequest, request_id: str) -> Path:
        """Resolve the target path for a request.

        If filename is None, generate a unique name using the request_id.
        Validates containment and checks overwrite policy.
        """
        if request.filename is None:
            filename = generate_unique_filename(request_id)
        else:
            filename = request.filename

        target = self._target_path(filename)

        # Check for existing file unless overwrite is allowed
        if target.exists() and not request.overwrite:
            raise FileExistsError(f"Target file already exists: {target}")

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    async def _emit_event(
        self,
        event: ImageGenerationEvent,
        handler: Callable[[ImageGenerationEvent], None | Awaitable[None]] | None
    ) -> None:
        """Emit a lifecycle event to the optional callback.

        Callback failures are logged but do not affect generation outcome.
        Handles both sync and async callbacks safely.
        """
        if handler is None:
            return

        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            # Log callback failure but don't propagate it
            global_logger.exception(
                "Lifecycle event handler raised an exception: %s",
                exc,
                extra={"request_id": event.request_id}
            )

    async def generate_image_async(
        self,
        request: ImageGenerationRequest,
        *,
        on_event: Callable[[ImageGenerationEvent], None | Awaitable[None]] | None = None,
    ) -> ImageGenerationResult:
        """Generate an image and return a result object.

        This is the canonical async API. Returns ImageGenerationResult on success,
        raises ImageGenerationError on failure, or asyncio.CancelledError if cancelled.

        Args:
            request: The generation request parameters.
            on_event: Optional callback for lifecycle events.

        Returns:
            ImageGenerationResult with output path and metadata.

        Raises:
            ValueError: If the prompt is invalid or filename escapes media folder.
            FileExistsError: If target exists and overwrite=False.
            ImageGenerationError: If generation fails after retries.
            asyncio.CancelledError: If the calling task is cancelled.
        """
        import uuid
        from datetime import datetime, timezone
        import time as time_module

        request_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc)
        started_monotonic = time_module.monotonic()

        # Validate and normalize request before any network work
        request = self._validate_request(request)
        target = self._build_target_path(request, request_id)

        total_attempts = self.retry_attempts + 1

        # Emit QUEUED event
        queued_event = ImageGenerationEvent(
            request_id=request_id,
            status=ImageGenerationStatus.QUEUED,
            provider_name=self.provider_name,
            request=request,
            occurred_at=datetime.now(timezone.utc),
        )
        await self._emit_event(queued_event, on_event)

        last_error: Exception | None = None
        result_path: Path | None = None
        attempts_made = 0

        try:
            for attempt in range(1, total_attempts + 1):
                attempts_made = attempt

                # Emit GENERATING event
                generating_event = ImageGenerationEvent(
                    request_id=request_id,
                    status=ImageGenerationStatus.GENERATING,
                    provider_name=self.provider_name,
                    request=request,
                    occurred_at=datetime.now(timezone.utc),
                    attempt=attempt,
                    total_attempts=total_attempts,
                )
                await self._emit_event(generating_event, on_event)

                try:
                    result_path = await asyncio.wait_for(
                        self._generate_once_async(request=request),
                        timeout=self.timeout,
                    )
                    last_error = None
                    break

                except asyncio.CancelledError:
                    # Emit CANCELLED event and re-raise
                    cancelled_event = ImageGenerationEvent(
                        request_id=request_id,
                        status=ImageGenerationStatus.CANCELLED,
                        provider_name=self.provider_name,
                        request=request,
                        occurred_at=datetime.now(timezone.utc),
                        attempt=attempt,
                    )
                    await self._emit_event(cancelled_event, on_event)
                    raise

                except ImageGenerationError as exc:
                    last_error = exc
                    retryable = exc.retryable

                except (TimeoutError, ConnectionError) as exc:
                    last_error = exc
                    retryable = True

                except OSError as exc:
                    # Disk/permission problems typically won't fix themselves
                    last_error = exc
                    retryable = not isinstance(
                        exc, (PermissionError, FileNotFoundError, IsADirectoryError)
                    )

                except Exception as exc:
                    # Unexpected provider bug: keep the public error contract
                    raise ImageGenerationError(
                        f"{self.provider_name} raised an unexpected "
                        f"{type(exc).__name__}: {exc}",
                        retryable=False,
                    ) from exc

                if not retryable or attempt >= total_attempts:
                    break

                # Emit RETRYING event
                delay = self.retry_delay_seconds * attempt
                retrying_event = ImageGenerationEvent(
                    request_id=request_id,
                    status=ImageGenerationStatus.RETRYING,
                    provider_name=self.provider_name,
                    request=request,
                    occurred_at=datetime.now(timezone.utc),
                    attempt=attempt + 1,
                    total_attempts=total_attempts,
                    delay_seconds=delay,
                )
                await self._emit_event(retrying_event, on_event)

                global_logger.warning(
                    "[%s] attempt %d/%d failed; retrying in %gs: %s",
                    self.provider_name, attempt, total_attempts, delay, last_error,
                )
                await asyncio.sleep(delay)

            # Check if generation succeeded
            if last_error is not None:
                error_event = ImageGenerationEvent(
                    request_id=request_id,
                    status=ImageGenerationStatus.ERROR,
                    provider_name=self.provider_name,
                    request=request,
                    occurred_at=datetime.now(timezone.utc),
                    attempt=attempts_made,
                    error=last_error,
                )
                await self._emit_event(error_event, on_event)

                raise ImageGenerationError(
                    f"{self.provider_name} image generation failed: {last_error}",
                    retryable=False,
                ) from last_error

            # Validate result path is a local file
            if result_path is None or not Path(result_path).is_file():
                raise ImageGenerationError(
                    f"{self.provider_name} did not return a valid image file: "
                    f"{result_path!r}",
                    retryable=False,
                )

            # Emit SAVING event
            saving_event = ImageGenerationEvent(
                request_id=request_id,
                status=ImageGenerationStatus.SAVING,
                provider_name=self.provider_name,
                request=request,
                occurred_at=datetime.now(timezone.utc),
                path=target,
            )
            await self._emit_event(saving_event, on_event)

            # Move file to target location
            try:
                moved = await asyncio.to_thread(
                    shutil.move, str(result_path), str(target)
                )
            except OSError as exc:
                raise ImageGenerationError(
                    f"Failed to move generated image to {target}: {exc}",
                    retryable=False,
                ) from exc

            img_path = Path(moved)
            if not img_path.is_file():
                raise ImageGenerationError(
                    f"Generated image is missing after move: {target}",
                    retryable=False,
                )

            # Calculate duration and build result
            completed_at = datetime.now(timezone.utc)
            duration_seconds = (
                completed_at.timestamp() - started_at.timestamp()
            )

            result = ImageGenerationResult(
                request_id=request_id,
                path=img_path,
                provider_name=self.provider_name,
                model=self.model_name,
                request=request,
                attempts=attempts_made,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration_seconds,
            )

            # Emit SUCCESS event
            success_event = ImageGenerationEvent(
                request_id=request_id,
                status=ImageGenerationStatus.SUCCESS,
                provider_name=self.provider_name,
                request=request,
                occurred_at=completed_at,
                path=img_path,
            )
            await self._emit_event(success_event, on_event)

            global_logger.info("[%s] saved image to %s", self.provider_name, img_path)
            return result

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Wrap unexpected exceptions in ImageGenerationError
            if not isinstance(exc, ImageGenerationError):
                error = ImageGenerationError(
                    f"{self.provider_name} image generation failed: {exc}",
                    retryable=False,
                )
            else:
                error = exc

            error_event = ImageGenerationEvent(
                request_id=request_id,
                status=ImageGenerationStatus.ERROR,
                provider_name=self.provider_name,
                request=request,
                occurred_at=datetime.now(timezone.utc),
                error=error,
            )
            await self._emit_event(error_event, on_event)
            raise error

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        resolution: int = 1024,
        aspect_ratio: str = "1:1",
        steps: int | None = None,
        seed: int | None = None,
        filename: str | None = None,
        overwrite: bool = False,
        on_event: Callable[[ImageGenerationEvent], None | Awaitable[None]] | None = None,
    ) -> ImageGenerationResult:
        """Convenience method with keyword arguments.

        Creates an ImageGenerationRequest and calls generate_image_async().
        This is the recommended API for most callers.
        """
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            steps=steps,
            seed=seed,
            filename=filename,
            overwrite=overwrite,
        )
        return await self.generate_image_async(request, on_event=on_event)

    def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        on_event: Callable[[ImageGenerationEvent], None | Awaitable[None]] | None = None,
    ) -> ImageGenerationResult:
        """Synchronous bridge over the true async implementation.

        Runs generate_image_async() in a fresh event loop when none is running;
        if called from inside a running loop it raises, to avoid silent
        executor-based blocking.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.generate_image_async(request, on_event=on_event)
            )
        raise RuntimeError(
            "generate_image() cannot be called from a running event loop; "
            "use await generate_image_async(...) instead."
        )

    @abstractmethod
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
    ) -> Path:
        """Perform exactly one provider attempt and return a local image file.

        The returned path must be a regular file. The provider must clean up its
        own partial temporary files if it raises. It must not move a result into
        the configured media path; the base class performs the final placement.

        Args:
            request: The generation request parameters.
            prompt: Optional override for the prompt text.
            negative_prompt: Optional override for negative prompt guidance.
            resolution: Optional override for output resolution.
            aspect_ratio: Optional override for aspect ratio.
            steps: Optional override for inference steps.
            seed: Optional override for random seed.

        Returns:
            Path to the generated image file on disk.

        Raises:
            ImageGenerationError: If generation fails.
        """
        pass