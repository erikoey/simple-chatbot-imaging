"""Standalone image generation package.

Providers implement :class:`BaseImageGenerator`; use
:func:`create_image_generator` to build one by name, or compose several
providers with :class:`FallbackImageGenerator` (opt-in, caller-defined chain).
"""

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.config import load_provider_config, merge_provider_kwargs
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
    ImageGenerationStatus,
    ImageGenerationState,
)

__all__ = [
    "BaseImageGenerator",
    "FallbackImageGenerator",
    "ImageGenerationError",
    "ImageGenerationEvent",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageGenerationStatus",
    "ImageGenerationState",
    "create_image_generator",
    "load_provider_config",
    "merge_provider_kwargs",
    "register_provider",
]

__version__ = "1.0.0-beta.4"