"""Local SDXL image generation provider (this machine's GPU).

Optional provider: requires the ``sdxl`` extra (``sdxl-generator`` plus a
GPU-enabled torch build — AMD ROCm wheels on Windows, CUDA build on Linux).
Generation runs through ``sdxl_generator.generate()``, which is synchronous
and caches one model pipeline in-process; a module-level lock serializes
calls because that package does not serialize concurrent callers itself.
"""

import asyncio
import logging
import threading
import uuid
from pathlib import Path

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.models import (
    ImageGenerationAttempt,
    ImageGenerationCost,
    ImageGenerationError,
    ImageGenerationRequest,
)

logger = logging.getLogger(__name__)

# One GPU pipeline per process: sdxl_generator holds a single-entry model
# cache and documents that concurrent calls are not serialized.
_GENERATION_LOCK = threading.Lock()


def release_engine() -> None:
    """Drop the process-global GPU engine so its memory can be reclaimed.

    Safe no-op when the ``sdxl`` extra is not installed.
    """
    try:
        from sdxl_generator import release_engine as _release
    except ImportError:
        return
    _release()


def _resolution_to_dimensions(resolution: int, aspect_ratio: str) -> tuple[int, int]:
    """Map resolution and aspect ratio ("W:H") to SDXL pixel dimensions.

    The longer side becomes ``resolution``; both sides are truncated to
    multiples of 8 (required by the SDXL pipeline) with a floor of 8.
    """
    try:
        w_ratio, h_ratio = (float(part) for part in aspect_ratio.split(":", 1))
        if w_ratio <= 0 or h_ratio <= 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"aspect_ratio must look like 'W:H' with positive numbers, "
            f"got {aspect_ratio!r}"
        ) from exc

    def snap(value: float) -> int:
        return max(8, (int(value) // 8) * 8)

    ratio = h_ratio / w_ratio
    if ratio >= 1:
        return snap(round(resolution / ratio)), snap(resolution)
    return snap(resolution), snap(round(resolution * ratio))


class SdxlImageGenerator(BaseImageGenerator):
    """Generate images with a local SDXL pipeline instead of a hosted API.

    Has no endpoint or API key; the tunables are the pipeline's own:
    model and VAE repo ids, inference steps, guidance scale, and device.
    ``vae=None`` uses the VAE baked into the model checkpoint's
    safetensors instead of loading a separate one.
    """

    DEFAULT_MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"
    DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

    def __init__(self,
                 model: str = DEFAULT_MODEL_ID,
                 vae: str | None = DEFAULT_VAE_ID,
                 device: str = "cuda:0",
                 steps: int = 7,
                 guidance_scale: float = 1.5,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_id = model
        self.vae_id = vae
        self.device = device
        self.steps = steps
        self.guidance_scale = guidance_scale
        self._tmp_dir = self.media_path / ".sdxl_tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str | None:
        return self.model_id

    def prewarm(self) -> bool:
        """Load the pipeline and tune GPU kernels with a tiny 8x8 image.

        Intended for app startup, off the request path: the first real
        generation then reuses the warmed engine instead of paying the
        pipeline load and (on ROCm) MIOpen kernel tuning inline. Failures
        are logged, never raised, so startup cannot be blocked; returns
        True when the engine is ready.
        """
        try:
            attempt = self._generate_local(ImageGenerationRequest(
                prompt="prewarm",
                resolution=8,
                aspect_ratio="1:1",
                steps=1,
                seed=0,
            ))
        except Exception as exc:
            logger.warning("[%s] prewarm failed: %s", self.provider_name, exc)
            return False
        Path(attempt.path).unlink(missing_ok=True)
        return True

    def release(self) -> None:
        """Drop the cached engine to free its GPU memory.

        The engine cache is process-global: releasing affects every
        SdxlImageGenerator in this process, and the next generation reloads
        the pipeline.
        """
        release_engine()

    def _generate_local(self, request: ImageGenerationRequest) -> ImageGenerationAttempt:
        """Synchronous single attempt; runs in a worker thread via to_thread."""
        try:
            from sdxl_generator import generate
        except ImportError as exc:
            raise ImageGenerationError(
                "The SDXL provider requires the 'sdxl' extra. "
                "Install it with: uv add 'simple-chatbot-imaging[sdxl]'",
                retryable=False,
            ) from exc

        width, height = _resolution_to_dimensions(request.resolution, request.aspect_ratio)
        tmp_path = self._tmp_dir / f"{uuid.uuid4().hex}.png"
        try:
            with _GENERATION_LOCK:
                image = generate(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=width,
                    height=height,
                    steps=request.steps if request.steps is not None else self.steps,
                    guidance_scale=self.guidance_scale,
                    seed=request.seed,
                    device=self.device,
                    model_id=self.model_id,
                    vae_id=self.vae_id,
                )
                image.save(tmp_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        # Local generation consumes no metered API: report an explicitly
        # free generation for the single image produced.
        return ImageGenerationAttempt(
            path=tmp_path,
            cost=ImageGenerationCost.zero(image_count=1),
        )

    async def _generate_once_async(self, request: ImageGenerationRequest) -> Path:
        """Run a single local generation without blocking the event loop."""
        return await asyncio.to_thread(self._generate_local, request)
