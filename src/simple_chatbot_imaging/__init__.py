"""Standalone image generation package.

Providers implement :class:`BaseImageGenerator`; use
:func:`create_image_generator` to build one by name, or compose several
providers with :class:`FallbackImageGenerator` (opt-in, caller-defined chain).
"""

from simple_chatbot_imaging.base import (
    BaseImageGenerator,
    ImageGenerationError,
    ImageGenerationState,
)
from simple_chatbot_imaging.factory import (
    FallbackImageGenerator,
    create_image_generator,
    register_provider,
)

__all__ = [
    "BaseImageGenerator",
    "FallbackImageGenerator",
    "ImageGenerationError",
    "ImageGenerationState",
    "create_image_generator",
    "register_provider",
]

__version__ = "0.1.0"