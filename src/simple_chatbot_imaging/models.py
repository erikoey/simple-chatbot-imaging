"""Data models for stateless, concurrent image generation."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable


class ImageGenerationStatus(StrEnum):
    """Lifecycle state for one image-generation request."""

    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    RETRYING = "RETRYING"
    SAVING = "SAVING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class ImageGenerationState(StrEnum):
    """Deprecated: use :class:`ImageGenerationStatus` instead.

    Kept for backward compatibility with existing code that may still reference
    this enum directly. New code should use :class:`ImageGenerationStatus`.
    """

    IDLE = "IDLE"
    GENERATING = "GENERATING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    """Inputs and output options for a single image-generation operation."""

    prompt: str
    negative_prompt: str = ""
    resolution: int = 1024
    aspect_ratio: str = "1:1"
    steps: int | None = None
    seed: int | None = None
    filename: str | None = None
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ImageGenerationCost:
    """Cost information reported by a provider for one generation request.

    Semantics:

    - ``None`` on any field means the provider did not report that
      information (unknown, not free).
    - ``0`` means the provider reported or the package asserts a free
      generation (e.g. the Hugging Face provider always reports zero).
    - A response may carry several unit types at once (credits, actual
      currency, tokens).
    - ``image_count`` is the number of images the provider says the request
      produced, when reported (some APIs bill per produced image).
    - ``raw_usage`` holds the provider's usage/cost object verbatim for
      callers that want to interpret provider-specific fields themselves.

    ``per_image_*`` tuples hold one entry per generated image, in the same
    unit as the matching total; ``None`` when the response does not break
    the cost down per image. Individual entries are ``None`` when the
    response omits the value for that image.
    """

    total_credits: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    total_tokens: int | None = None
    image_count: int | None = None
    per_image_credits: tuple[float | None, ...] | None = None
    per_image_amounts: tuple[float | None, ...] | None = None
    per_image_tokens: tuple[int | None, ...] | None = None
    raw_usage: Mapping[str, Any] | None = None

    @classmethod
    def zero(cls, *, image_count: int | None = None) -> ImageGenerationCost:
        """Explicitly free generation: every monetary/token field is 0."""
        return cls(
            total_credits=0.0,
            total_amount=0.0,
            total_tokens=0,
            image_count=image_count,
        )

    @property
    def is_zero(self) -> bool:
        """True when every reported cost quantity is exactly zero."""
        reported = [
            value
            for value in (
                self.total_credits,
                self.total_amount,
                self.total_tokens,
            )
            if value is not None
        ]
        return bool(reported) and all(value == 0 for value in reported)


@dataclass(frozen=True, slots=True)
class ImageGenerationAttempt:
    """Outcome of a single provider attempt.

    Providers may return this instead of a bare ``Path`` from
    :meth:`BaseImageGenerator._generate_once_async` to surface cost
    information extracted from the provider response. Returning a bare
    ``Path`` is still supported and means "no cost information".

    ``raw_response_json`` is the provider's JSON response body as a string
    (or ``None`` for non-JSON responses), so callers can inspect
    provider-specific fields themselves.
    """

    path: Path
    cost: ImageGenerationCost | None = None
    raw_response_json: str | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    """Successful result of one image-generation operation."""

    request_id: str
    path: Path
    provider_name: str
    model: str | None
    request: ImageGenerationRequest
    attempts: int
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    cost: ImageGenerationCost | None = None
    raw_response_json: str | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerationEvent:
    """A request-local lifecycle event emitted during generation."""

    request_id: str
    status: ImageGenerationStatus
    provider_name: str
    request: ImageGenerationRequest
    occurred_at: datetime
    attempt: int | None = None
    total_attempts: int | None = None
    delay_seconds: float | None = None
    path: Path | None = None
    message: str | None = None
    error: Exception | None = None
    cost: ImageGenerationCost | None = None
    raw_response_json: str | None = None


class ImageGenerationEventHandler:
    """Type alias for lifecycle event handlers."""
    pass

ImageGenerationEventHandler = Callable[[ImageGenerationEvent], None | Awaitable[None]]


def generate_unique_filename(request_id: str) -> str:
    """Generate a collision-resistant default filename."""
    return f"image-{request_id}.png"


class ImageGenerationError(Exception):
    """Raised when image generation fails.

    Attributes:
        message: Human-readable error description.
        retryable: Whether the failure might succeed on retry (transient errors).
        cause: The underlying exception that caused this error.
    """

    def __init__(self, message: str, *, retryable: bool = False, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.cause = cause
