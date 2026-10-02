"""OpenRouter image generation provider.

Uses the OpenRouter image generation endpoint via ``httpx`` (native async),
which returns a base64 data URL that is decoded and saved to a temporary file.
All HTTP, status-mapping, response-resolution, and temp-file handling lives in
:class:`BaseHTTPImageGenerator`.
"""

from typing import Any

from simple_chatbot_imaging.models import (
    ImageGenerationAttempt,
    ImageGenerationCost,
    ImageGenerationError,
    ImageGenerationRequest,
)
from simple_chatbot_imaging.providers.http_base import BaseHTTPImageGenerator


class OpenRouterImageGenerator(BaseHTTPImageGenerator):
    """Generate images through the OpenRouter images/generations API."""

    provider_label = "OpenRouter"

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/images"
    DEFAULT_MODEL = "meta/muse-image"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 request_timeout_seconds: int = 120,
                 **kwargs) -> None:
        
        super().__init__(model=model,
                         base_url=base_url,
                         api_key_env=api_key_env,
                         request_timeout_seconds=request_timeout_seconds,
                         **kwargs,
                         )

    @property
    def _endpoint(self) -> str:
        """Return the generations endpoint for either a base URL or full URL."""
        suffix = "/generations"
        if self.base_url.endswith(suffix):
            return self.base_url
        return f"{self.base_url}{suffix}"

    def _build_payload(self, request: ImageGenerationRequest) -> dict[str, Any]:
        prompt = request.prompt
        # OpenRouter's image API does not support negative prompts, steps,
        # seeds, or aspect ratios; fold the negative prompt into the prompt
        # as guidance and ignore the remaining parameters.
        if request.negative_prompt:
            prompt = f"{prompt}. Avoid: {request.negative_prompt}"
        return {
            "model": self.model,
            "prompt": prompt,
        }

    async def _generate_once_async(self, request: ImageGenerationRequest) -> ImageGenerationAttempt:
        """Call the OpenRouter image API once and save the result to a temp file."""
        payload = self._build_payload(request)
        response = await self._post_json(self._endpoint, payload)

        # Provider-specific override for 400: prompt filtered / inappropriate.
        if response.status_code == 400:
            raise ImageGenerationError(
                "Bad Request or the prompt was filtered, inappropiate content (400)",
                retryable=False,
            )

        # OpenRouter reports usage.cost in USD plus token counts in `usage`.
        attempt = await self._resolve_and_save(request, response)
        if attempt.cost is not None and attempt.cost.total_amount is not None:
            cost = ImageGenerationCost(
                total_credits=attempt.cost.total_credits,
                total_amount=attempt.cost.total_amount,
                currency=attempt.cost.currency or "USD",
                total_tokens=attempt.cost.total_tokens,
                image_count=attempt.cost.image_count,
                per_image_credits=attempt.cost.per_image_credits,
                per_image_amounts=attempt.cost.per_image_amounts,
                per_image_tokens=attempt.cost.per_image_tokens,
                raw_usage=attempt.cost.raw_usage,
            )
            attempt = ImageGenerationAttempt(
                path=attempt.path,
                cost=cost,
                raw_response_json=attempt.raw_response_json,
            )
        return attempt
