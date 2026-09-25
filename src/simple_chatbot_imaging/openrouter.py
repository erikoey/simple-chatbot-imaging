"""OpenRouter image generation provider.

Uses the OpenRouter image generation endpoint via ``httpx`` (native async),
which returns a base64 data URL that is decoded and saved to a temporary file.
"""

import base64
import os
import tempfile
from pathlib import Path

import httpx

from simple_chatbot_imaging.base import BaseImageGenerator, ImageGenerationError


class OpenRouterImageGenerator(BaseImageGenerator):
    """Generate images through the OpenRouter images/generations API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/images"
    DEFAULT_MODEL = "meta/muse-image"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = "OPENROUTER_API_KEY",
                 request_timeout_seconds: int = 120,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.request_timeout_seconds = request_timeout_seconds

    @property
    def api_key(self) -> str:
        """Return the OpenRouter API key from the environment."""
        key = os.getenv(self.api_key_env)
        if not key:
            raise ImageGenerationError(
                f"Missing API key: set the {self.api_key_env} environment variable.",
                retryable=False,
            )
        return key

    async def _generate_once_async(self,
                                   prompt: str,
                                   negative_prompt: str,
                                   resolution: int,
                                   aspect_ratio: str,
                                   steps: int,
                                   seed: int) -> Path:
        """Call the OpenRouter image API once and save the result to a temp file."""
        # OpenRouter's image API does not support negative prompts, steps,
        # seeds, or aspect ratios; fold the negative prompt into the prompt
        # as guidance and ignore the remaining parameters.
        if negative_prompt:
            prompt = f"{prompt}. Avoid: {negative_prompt}"

        url = f"{self.base_url}/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
        }

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code == 401:
            raise ImageGenerationError(
                "OpenRouter authentication failed (401). Check the API key.",
                retryable=False,
            )
        if response.status_code == 402:
            raise ImageGenerationError(
                "OpenRouter credit limit reached (402).",
                retryable=False,
            )
        if response.status_code == 403:
            # 403 covers permanent account/model restrictions: age
            # confirmation required, model not allowed for the account,
            # moderation blocks, etc. Retrying cannot fix these.
            raise ImageGenerationError(
                f"OpenRouter request forbidden (403): {response.text[:300]}",
                retryable=False,
            )
        if response.status_code == 404:
            raise ImageGenerationError(
                f"OpenRouter model not found (404): {self.model}",
                retryable=False,
            )
        if response.status_code == 429:
            # Rate limit or quota exhausted; retrying after a delay may help.
            raise ImageGenerationError(
                "OpenRouter rate/quota limit hit (429). Try again shortly."
            )
        if response.status_code >= 500:
            raise ImageGenerationError(
                f"OpenRouter server error ({response.status_code}). Try again shortly."
            )
        if response.status_code != 200:
            raise ImageGenerationError(
                f"OpenRouter request failed ({response.status_code}): {response.text[:300]}"
            )

        try:
            data = response.json()
            image_entry = data["data"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ImageGenerationError(
                f"Unexpected OpenRouter response format: {response.text[:300]}"
            ) from exc

        # Responses contain raw base64 in "b64_json" (optionally as a data
        # URL with a mime prefix) or a data URL in "image" — accept both.
        image_b64 = image_entry.get("b64_json") or image_entry.get("image")
        if not image_b64 or not isinstance(image_b64, str):
            raise ImageGenerationError(
                f"OpenRouter response contains no image data: {response.text[:300]}"
            )

        # Strip a data URL prefix like "data:image/jpeg;base64," if present.
        _, sep, payload_b64 = image_b64.partition(",")
        if not sep and not image_b64.startswith("data:"):
            payload_b64 = image_b64
        try:
            image_bytes = base64.b64decode(payload_b64)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError("OpenRouter returned invalid base64 image data.") from exc

        # Sniff the actual image type from the decoded bytes rather than
        # trusting the (often missing) mime prefix.
        if image_bytes[:3] == b"\xff\xd8\xff":
            suffix = ".jpg"
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            suffix = ".webp"
        else:
            suffix = ".png"

        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="openrouter_", delete=False
        )
        try:
            with tmp as f:
                f.write(image_bytes)
        except OSError as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise ImageGenerationError(f"Failed to save OpenRouter image: {exc}") from exc

        img_path = Path(tmp.name)
        if not img_path.is_file() or img_path.stat().st_size == 0:
            img_path.unlink(missing_ok=True)
            raise ImageGenerationError("OpenRouter image download produced an empty file.")

        return img_path