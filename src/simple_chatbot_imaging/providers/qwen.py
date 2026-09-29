"""Qwen image generation provider.

This provider follows the Qwen MaaS image-generation API shape where the
successful response contains an image URL instead of inline binary data. All
HTTP, status-mapping, response-resolution (including URL download), and
temp-file handling lives in :class:`BaseHTTPImageGenerator`.
"""

from typing import Any

from simple_chatbot_imaging.models import ImageGenerationRequest
from simple_chatbot_imaging.providers.http_base import BaseHTTPImageGenerator


class QwenImageGenerator(BaseHTTPImageGenerator):
    """Generate images via the Qwen MaaS multimodal generation endpoint."""

    provider_label = "Qwen"

    DEFAULT_BASE_URL = (
        "https://maas.qwencloudapi.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    DEFAULT_MODEL = "qwen-image-3.0"
    DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 request_timeout_seconds: int = 60,
                 prompt_extend: bool = True,
                 watermark: bool = False,
                 n: int = 1,
                 **kwargs) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            request_timeout_seconds=request_timeout_seconds,
            **kwargs,
        )
        self.prompt_extend = prompt_extend
        self.watermark = watermark
        self.n = n


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


    def _build_payload(self, request: ImageGenerationRequest) -> dict[str, Any]:
        # Qwen-Image accepts negative_prompt via parameters.negative_prompt
        # (docs: "Set via parameters.negative_prompt"); do not fold it into
        # the prompt text.
        parameters: dict[str, object] = {
            "prompt_extend": self.prompt_extend,
            "watermark": self.watermark,
            "size": self._size_parameter(request.resolution, request.aspect_ratio),
            "n": self.n,
        }
        if request.negative_prompt:
            parameters["negative_prompt"] = request.negative_prompt

        return {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "text": request.prompt,
                        }],
                    }
                ]
            },
            "parameters": parameters,
        }
