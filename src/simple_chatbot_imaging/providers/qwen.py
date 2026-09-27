"""Qwen image generation provider.

This provider follows the Qwen MaaS image-generation API shape where the
successful response contains an image URL instead of inline binary data. The URL
is downloaded to a temporary local file before it is returned to the base class.
"""

import os
import tempfile
from pathlib import Path

import httpx

from simple_chatbot_imaging.base import BaseImageGenerator, ImageGenerationError


class QwenImageGenerator(BaseImageGenerator):
    """Generate images via the Qwen MaaS multimodal generation endpoint."""

    DEFAULT_BASE_URL = (
        "https://maas.qwencloudapi.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    DEFAULT_MODEL = "qwen-image-3.0"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = "DASHSCOPE_API_KEY",
                 request_timeout_seconds: int = 60,
                 prompt_extend: bool = True,
                 watermark: bool = False,
                 n: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.request_timeout_seconds = request_timeout_seconds
        self.prompt_extend = prompt_extend
        self.watermark = watermark
        self.n = n


    @property
    def api_key(self) -> str:
        """Return the Qwen API key from the environment."""
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ImageGenerationError(
                f"Missing API key: set the {self.api_key_env} environment variable.",
                retryable=False,
            )
        return key

    def _extract_image_url(self, payload: object) -> str:
        """Extract the first downloadable image URL from a Qwen payload."""
        if isinstance(payload, dict):
            for key in ("url", "image_url", "imageUrl"):
                value = payload.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value

            # Qwen-Image responses put the URL under "image" inside
            # output.choices[].message.content[] entries.
            value = payload.get("image")
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value

            for key in ("output", "result", "data", "images", "image"):
                value = payload.get(key)
                if value is not None and not isinstance(value, str):
                    try:
                        return self._extract_image_url(value)
                    except ValueError:
                        pass

            if "choices" in payload and isinstance(payload["choices"], list):
                for choice in payload["choices"]:
                    try:
                        return self._extract_image_url(choice)
                    except ValueError:
                        continue

            if "message" in payload:
                try:
                    return self._extract_image_url(payload["message"])
                except ValueError:
                    pass

            if "content" in payload:
                try:
                    return self._extract_image_url(payload["content"])
                except ValueError:
                    pass

        elif isinstance(payload, list):
            for item in payload:
                try:
                    return self._extract_image_url(item)
                except ValueError:
                    continue

        raise ValueError("Qwen response did not contain a usable image URL.")

    @staticmethod
    def _infer_suffix(content_type: str | None, image_bytes: bytes) -> str:
        """Infer a file suffix from headers or signature bytes."""
        mime = (content_type or "").split(";", 1)[0].strip().lower()
        if mime == "image/jpeg":
            return ".jpg"
        if mime == "image/png":
            return ".png"
        if mime == "image/webp":
            return ".webp"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".png"

    @staticmethod
    def _size_parameter(resolution: int, aspect_ratio: str) -> str:
        """Map resolution + aspect ratio into the Qwen "width*height" size format."""
        width, _, height = aspect_ratio.partition(":")
        try:
            ratio_w = int(width)
            ratio_h = int(height)
        except ValueError:
            return f"{resolution}*{resolution}"

        if ratio_w <= 0 or ratio_h <= 0:
            return f"{resolution}*{resolution}"

        # Map the aspect ratio so the LONG edge equals the requested
        # resolution (e.g. 16:9 @ 2048 -> "2048*1152"), keeping sides as
        # multiples of 8 within the 512-2048 range for Qwen-Image models.
        def clamp(value: int) -> int:
            return max(512, min(2048, value))

        def scale_to(edge: int, numerator: int, denominator: int) -> int:
            return int(round(edge * numerator / denominator / 8) * 8)

        if ratio_w >= ratio_h:
            out_w = clamp(resolution)
            out_h = clamp(scale_to(resolution, ratio_h, ratio_w))
        else:
            out_h = clamp(resolution)
            out_w = clamp(scale_to(resolution, ratio_w, ratio_h))
        return f"{out_w}*{out_h}"

    async def _generate_once_async(self,
                                   prompt: str,
                                   negative_prompt: str,
                                   resolution: int,
                                   aspect_ratio: str,
                                   steps: int,
                                   seed: int) -> Path:
        """Call the Qwen generation endpoint and download the resulting image."""
        # Qwen-Image accepts negative_prompt via parameters.negative_prompt
        # (docs: "Set via parameters.negative_prompt"); do not fold it into
        # the prompt text.
        parameters: dict[str, object] = {
            "prompt_extend": self.prompt_extend,
            "watermark": self.watermark,
            "size": self._size_parameter(resolution, aspect_ratio),
            "n": self.n,
        }
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "text": prompt,
                        }],
                    }
                ]
            },
            "parameters": parameters,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(f"Qwen request failed: {exc}") from exc

        if response.status_code == 401:
            raise ImageGenerationError(
                "Qwen authentication failed (401). Check the API key.",
                retryable=False,
            )
        if response.status_code == 402:
            raise ImageGenerationError(
                "Qwen credit limit reached (402).",
                retryable=False,
            )
        if response.status_code == 403:
            raise ImageGenerationError(
                f"Qwen request forbidden (403): {response.text[:300]}",
                retryable=False,
            )
        if response.status_code == 404:
            raise ImageGenerationError(
                f"Qwen model not found (404): {self.model}",
                retryable=False,
            )
        if response.status_code == 429:
            raise ImageGenerationError("Qwen rate/quota limit hit (429). Try again shortly.")
        if response.status_code >= 500:
            raise ImageGenerationError(
                f"Qwen server error ({response.status_code}). Try again shortly."
            )
        if response.status_code != 200:
            raise ImageGenerationError(
                f"Qwen request failed ({response.status_code}): {response.text[:300]}"
            )

        try:
            result_json = response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                f"Unexpected Qwen response format: {response.text[:300]}"
            ) from exc

        try:
            image_url = self._extract_image_url(result_json)
        except ValueError as exc:
            raise ImageGenerationError(
                f"Qwen response contains no image URL: {response.text[:300]}"
            ) from exc

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                image_response = await client.get(image_url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(f"Qwen image download failed: {exc}") from exc

        if image_response.status_code != 200:
            raise ImageGenerationError(
                f"Qwen image download failed ({image_response.status_code}): {image_response.text[:300]}"
            )

        image_bytes = image_response.content
        if not image_bytes:
            raise ImageGenerationError("Qwen image download returned empty content.")

        suffix = self._infer_suffix(
            image_response.headers.get("content-type"),
            image_bytes,
        )

        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="qwen_",
            delete=False,
        )
        try:
            with tmp as f:
                f.write(image_bytes)
        except OSError as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise ImageGenerationError(f"Failed to save Qwen image: {exc}") from exc

        img_path = Path(tmp.name)
        if not img_path.is_file() or img_path.stat().st_size == 0:
            img_path.unlink(missing_ok=True)
            raise ImageGenerationError("Qwen image download produced an empty file.")

        return img_path
