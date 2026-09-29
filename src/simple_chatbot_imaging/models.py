"""Data models for stateless, concurrent image generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Awaitable, Callable


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
