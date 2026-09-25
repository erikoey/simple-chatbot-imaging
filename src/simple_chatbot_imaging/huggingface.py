"""Hugging Face Spaces (Gradio / ZeroGPU) image generation provider.

Optional provider: requires the ``huggingface`` extra (``gradio-client``).
The Gradio client is synchronous, so calls are bridged with
``asyncio.to_thread`` to keep the async contract non-blocking.
"""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simple_chatbot_imaging.base import BaseImageGenerator, ImageGenerationError

if TYPE_CHECKING:
    from gradio_client import Client
    from gradio_client.exceptions import AppError
    from gradio_client.utils import QueueError

try:
    from gradio_client import Client as _Client
    from gradio_client.exceptions import AppError as _AppError
    from gradio_client.utils import QueueError as _QueueError

    HAS_GRADIO_CLIENT = True
except ImportError:  # pragma: no cover - exercised via lazy factory error
    _Client = None  # type: ignore[assignment]
    _AppError = None  # type: ignore[assignment]
    _QueueError = None  # type: ignore[assignment]
    HAS_GRADIO_CLIENT = False


class HuggingFaceImageGenerator(BaseImageGenerator):
    """Generate images through a Hugging Face Gradio space (ZeroGPU)."""

    DEFAULT_SPACE_ID = "hugging-apps/qwen-image-2-1"

    def __init__(self,
                 space_id: str = DEFAULT_SPACE_ID,
                 hf_token_env: str = "HUGGINGFACE_ACCESS_TOKEN",
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.space_id = space_id
        self.hf_token_env = hf_token_env

        self._client: Any | None = None

    @property
    def token(self) -> str | None:
        """Return the Hugging Face access token, if configured."""
        return os.getenv(self.hf_token_env)

    def _build_client(self) -> Any:
        """Create the Gradio client; raises a clear error when the extra is missing."""
        if not HAS_GRADIO_CLIENT:
            raise ImageGenerationError(
                "The Hugging Face provider requires the 'huggingface' extra. "
                "Install it with: uv add 'simple-chatbot-imaging[huggingface]'",
                retryable=False,
            )
        return _Client(
            src=self.space_id,
            token=self.token,
            download_files=str(self.media_path / ".gradio_tmp"),
        )

    @property
    def client(self) -> Any:
        """Return (and lazily create) the Gradio client for the space."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def view_api(self) -> None:
        """Print the space API, for debug purposes only."""
        self.client.view_api()

    def _submit_and_wait(self,
                         prompt: str,
                         negative_prompt: str,
                         resolution: int,
                         aspect_ratio: str,
                         steps: int,
                         seed: int) -> Path:
        """Synchronous Gradio call; runs in a worker thread via to_thread."""
        # predict(prompt, input_images, negative_prompt, true_cfg_scale,
        #         num_inference_steps, seed, resolution, aspect_ratio, api_name="/generate")
        job = self.client.submit(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            seed=seed if seed > 0 else random.randint(1, 999999),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            api_name="/generate",
        )

        deadline = time.monotonic() + self.timeout

        try:
            while not job.done():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"HF image generation timeout hit: {self.timeout}s.")
                time.sleep(2)

            result = job.result()

            if not result:
                raise ImageGenerationError("HF API did not produce an image")

            # For now the space only returns a single local download path.
            if not isinstance(result, str):
                raise ImageGenerationError(
                    f"Unexpected HF API result type: {type(result).__name__}: {result!r}"
                )

            img_path = Path(result)
            if not img_path.is_file():
                raise ImageGenerationError(
                    f"HF API did not return a local downloaded image file: {result!r}"
                )

            return img_path

        except TimeoutError as exc:
            raise ImageGenerationError(f"HF image generation failed: {exc}") from exc

        except _QueueError as exc:  # type: ignore[misc]
            raise ImageGenerationError(
                "The Hugging Face image queue is currently full. Please try again shortly."
            ) from exc

        except _AppError as exc:  # type: ignore[misc]
            message = str(exc)

            if "expired zerogpu proxy token" in message.lower():
                raise ImageGenerationError(
                    "The Hugging Face shared-GPU session expired. Please retry."
                ) from exc

            if "upstream gradio app has raised an exception" in message.lower():
                raise ImageGenerationError(
                    "The public Qwen image service had an internal error. Try again shortly."
                ) from exc

            raise ImageGenerationError(f"HF image generation failed: {message}") from exc

        finally:
            if job and not job.done():
                job.cancel()

    async def _generate_once_async(self,
                                   prompt: str,
                                   negative_prompt: str,
                                   resolution: int,
                                   aspect_ratio: str,
                                   steps: int,
                                   seed: int) -> Path:
        """Run a single generation job on the HF space without blocking the loop."""
        return await asyncio.to_thread(
            self._submit_and_wait,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            steps=steps,
            seed=seed,
        )