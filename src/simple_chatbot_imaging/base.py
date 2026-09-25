"""Base classes for image generation providers.

This module only depends on the standard library plus ``httpx``-free
utilities; it defines the true async provider contract, retry semantics,
state tracking, output movement, and a synchronous bridge.
"""

import asyncio
import shutil
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path


class ImageGenerationError(RuntimeError):
    """Raised when remote image generation cannot produce an output file.

    Args:
        message: Human-readable error description.
        retryable: False for permanent errors (bad API key, no credits,
            age confirmation required, unknown model) that must not be
            retried by the base class.
    """

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ImageGenerationState(StrEnum):
    """Represent the current state of image generation."""

    IDLE = "IDLE"
    GENERATING = "GENERATING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class BaseImageGenerator(ABC):
    """Abstract base class for image generation providers.

    Subclasses implement :meth:`_generate_once_async`, which performs a
    single provider call and returns the path to a locally available image
    file. The base class handles prompt validation, state tracking, retries
    with backoff, timeouts, and moving the result into the media folder.
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

        self._state = ImageGenerationState.IDLE

    @property
    def state(self) -> ImageGenerationState:
        """Return the current image generation state."""
        return self._state

    @property
    def provider_name(self) -> str:
        """Human-readable provider name, used for logging."""
        return type(self).__name__

    def _validate_prompt(self, prompt: str) -> str:
        """Validate and normalize the prompt; raises ValueError when invalid."""
        if not prompt:
            raise ValueError("No prompt given to generate an image")

        prompt = prompt.strip()
        if len(prompt) < 3:
            raise ValueError("Provide an image prompt with at least 3 characters.")
        return prompt

    def _target_path(self, filename: str) -> Path:
        """Resolve the target path and ensure it stays inside the media folder."""
        target = (self.media_path / filename).resolve()
        if not target.is_relative_to(self.media_path.resolve()):
            raise ValueError(
                f"Filename {filename!r} must stay inside the media folder."
            )
        return target

    async def generate_image_async(self,
                                   prompt: str,
                                   negative_prompt: str = "",
                                   resolution: int = 1024,
                                   aspect_ratio: str = "1:1",
                                   steps: int = 40,
                                   seed: int = 0,
                                   filename: str = "image.png"
                                   ) -> str:
        """Generate an image (true async) and return the path to the saved file.

        Raises:
            ValueError: If the prompt is missing/too short or the filename
                would escape the media folder.
            ImageGenerationError: If generation fails after all retries.
        """
        self._state = ImageGenerationState.GENERATING

        try:
            prompt = self._validate_prompt(prompt)
            target = self._target_path(filename)
        except ValueError:
            self._state = ImageGenerationState.ERROR
            raise

        last_error: Exception | None = None
        result_path: Path | None = None
        for attempt in range(self.retry_attempts + 1):
            temp_path: Path | None = None
            try:
                result_path = await asyncio.wait_for(
                    self._generate_once_async(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        steps=steps,
                        seed=seed,
                    ),
                    timeout=self.timeout,
                )
                last_error = None
                break
            except (ImageGenerationError, TimeoutError, OSError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                if isinstance(exc, ImageGenerationError) and not exc.retryable:
                    break  # permanent error: retrying another provider call won't help

                delay = self.retry_delay_seconds * (attempt + 1)
                print(
                    f"[{self.provider_name}] attempt {attempt + 1} failed; "
                    f"retrying in {delay:g}s: {exc}"
                )
                await asyncio.sleep(delay)
            finally:
                # Clean up a partial temp file from a failed attempt.
                temp_path = None  # providers clean their own temp files

        if last_error is not None:
            self._state = ImageGenerationState.ERROR
            raise ImageGenerationError(
                f"{self.provider_name} image generation failed: {last_error}"
            ) from last_error

        if not result_path or not Path(result_path).is_file():
            self._state = ImageGenerationState.ERROR
            raise ImageGenerationError(
                f"{self.provider_name} did not return a valid image file: {result_path!r}"
            )

        img_path = Path(shutil.move(str(result_path), str(target)))

        if not img_path.is_file():
            self._state = ImageGenerationState.ERROR
            raise ImageGenerationError(f"Failed to move generated image to {target}")

        print(img_path)
        self._state = ImageGenerationState.SUCCESS
        return str(img_path)

    def generate_image(self,
                       prompt: str,
                       negative_prompt: str = "",
                       resolution: int = 1024,
                       aspect_ratio: str = "1:1",
                       steps: int = 40,
                       seed: int = 0,
                       filename: str = "image.png"
                       ) -> str:
        """Synchronous bridge over the true async implementation.

        Runs :meth:`generate_image_async` in a fresh event loop when none is
        running; if called from inside a running loop it raises, to avoid
        silent executor-based blocking.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.generate_image_async(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    steps=steps,
                    seed=seed,
                    filename=filename,
                )
            )
        raise RuntimeError(
            "generate_image() cannot be called from a running event loop; "
            "use await generate_image_async(...) instead."
        )

    @abstractmethod
    def _generate_once_async(self,
                             prompt: str,
                             negative_prompt: str,
                             resolution: int,
                             aspect_ratio: str,
                             steps: int,
                             seed: int) -> Path:
        """Perform a single generation call and return a local image path.

        Implementations must raise :class:`ImageGenerationError` (or a
        subclass of OSError/TimeoutError) on failure; the base class wraps
        these into a final :class:`ImageGenerationError` after retries.
        Implementations should clean up their own temporary files on failure.
        """