"""Standalone image generation package.

Providers implement :class:`BaseImageGenerator`; use
:func:`create_image_generator` to build one by name, or compose several
providers with :class:`FallbackImageGenerator` (opt-in, caller-defined chain).
"""

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.config import load_provider_config
from simple_chatbot_imaging.factory import (
    FallbackImageGenerator,
    create_image_generator,
    register_provider,
)
from simple_chatbot_imaging.models import (
    ImageGenerationError,
    ImageGenerationEvent,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerationState,
    ImageGenerationStatus,
)
from simple_chatbot_imaging.providers.http_base import BaseHTTPImageGenerator

__all__ = [
    "BaseHTTPImageGenerator",
    "BaseImageGenerator",
    "FallbackImageGenerator",
    "ImageGenerationError",
    "ImageGenerationEvent",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageGenerationState",
    "ImageGenerationStatus",
    "create_image_generator",
    "load_provider_config",
    "register_provider",
]

__version__ = "1.0.0-beta.5"