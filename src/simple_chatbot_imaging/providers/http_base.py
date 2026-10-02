"""Shared base class for HTTP-based image generation providers.

Providers that talk to a plain JSON-over-HTTP image API (OpenRouter, Qwen MaaS,
and similar services) can subclass :class:`BaseHTTPImageGenerator` and only
describe their request payload. The base class owns the full pipeline:

build payload -> POST -> status mapping -> response resolution (inline bytes,
base64 field, or image-URL download) -> suffix sniffing -> temp-file save.

The response resolution is generic: the base class inspects the JSON payload
and decides whether the image is embedded (``image/*`` content or a base64
field) or must be downloaded from a URL field. Field-name tuples are class
attributes so subclasses can extend or reorder them for exotic API shapes.
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.models import (
    ImageGenerationAttempt,
    ImageGenerationCost,
    ImageGenerationError,
    ImageGenerationRequest,
)


class BaseHTTPImageGenerator(BaseImageGenerator):
    """Template base for JSON-over-HTTP image generation providers.

    Subclasses implement :meth:`_build_payload` and set :attr:`provider_label`
    plus the endpoint/defaults. Everything else (auth header, POST, HTTP status
    mapping, response parsing, image extraction, temp-file handling) is shared.
    """

    #: Human-readable label used in error messages ("OpenRouter", "Qwen", ...).
    provider_label: str = "HTTP provider"

    #: JSON keys that may hold raw base64 image data in a response entry.
    BASE64_FIELDS: tuple[str, ...] = ("b64_json", "b64", "base64", "image_base64")

    #: JSON keys that may hold a downloadable image URL in a response entry.
    URL_FIELDS: tuple[str, ...] = ("url", "image_url", "imageUrl", "image", "output_url")

    #: JSON keys under which a usage/cost object may appear anywhere in the
    #: response. The first dict found wins.
    USAGE_FIELDS: tuple[str, ...] = ("usage", "usage_metadata", "cost_info", "billing")

    #: Keys within a usage object that hold a total monetary amount.
    AMOUNT_KEYS: tuple[str, ...] = ("cost", "total_cost", "amount", "total_amount", "price")

    #: Keys within a usage object that hold a total credits consumption.
    CREDITS_KEYS: tuple[str, ...] = (
        "credits", "credits_consumed", "credits_used", "image_credits", "credit_cost",
        "total_credits",
    )

    #: Keys within a usage object that hold a total token count.
    TOKEN_KEYS: tuple[str, ...] = ("total_tokens", "tokens_used", "tokens")

    #: Keys within a usage object that hold the number of produced images.
    IMAGE_COUNT_KEYS: tuple[str, ...] = ("image_count", "images_count", "num_images", "count")

    #: Keys within a usage object that hold a currency code.
    CURRENCY_KEYS: tuple[str, ...] = ("currency", "currency_code")

    DEFAULT_BASE_URL = ""
    DEFAULT_MODEL = ""
    DEFAULT_API_KEY_ENV = ""

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 request_timeout_seconds: int = 120,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.request_timeout_seconds = request_timeout_seconds

    @property
    def model_name(self) -> str | None:
        """Model identifier used for the result metadata."""
        return self.model

    @property
    def api_key(self) -> SecretStr:
        """Return the provider API key from the environment."""
        key = SecretStr(os.getenv(self.api_key_env, ""))
        if not key.get_secret_value():
            raise ImageGenerationError(
                f"Missing API key: set the {self.api_key_env} environment variable.",
                retryable=False,
            )
        return key

    # ------------------------------------------------------------------ #
    # Request building                                                    #
    # ------------------------------------------------------------------ #

    @property
    def _endpoint(self) -> str:
        """URL the generation POST is sent to. Override if it differs from base_url."""
        return self.base_url

    def _build_payload(self, request: ImageGenerationRequest) -> dict[str, Any]:
        """Build the provider-specific JSON body. Must be overridden."""
        raise NotImplementedError

    def _build_headers(self) -> dict[str, str]:
        """Build request headers; Bearer auth by default."""
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # HTTP helpers                                                         #
    # ------------------------------------------------------------------ #

    async def _post_json(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST a JSON payload, converting transport errors to ImageGenerationError."""
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                return await client.post(url, headers=self._build_headers(), json=payload)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(
                f"{self.provider_label} request failed: {exc}"
            ) from exc

    async def _download_image(self, url: str) -> tuple[bytes, str | None]:
        """GET an image URL with redirects; returns (bytes, content_type)."""
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                response = await client.get(url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise ImageGenerationError(
                f"{self.provider_label} image download failed: {exc}"
            ) from exc
        if response.status_code != 200:
            raise ImageGenerationError(
                f"{self.provider_label} image download failed "
                f"({response.status_code}): {response.text[:300]}"
            )
        if not response.content:
            raise ImageGenerationError(
                f"{self.provider_label} image download returned empty content."
            )
        return response.content, response.headers.get("content-type")

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes onto retryability-aware ImageGenerationError."""
        label = self.provider_label
        if response.status_code == 401:
            raise ImageGenerationError(
                f"{label} authentication failed (401). Check the API key.",
                retryable=False,
            )
        if response.status_code == 402:
            raise ImageGenerationError(
                f"{label} credit limit reached (402).",
                retryable=False,
            )
        if response.status_code == 403:
            # 403 covers permanent account/model restrictions: age
            # confirmation required, model not allowed for the account,
            # moderation blocks, etc. Retrying cannot fix these.
            raise ImageGenerationError(
                f"{label} request forbidden (403): {response.text[:300]}",
                retryable=False,
            )
        if response.status_code == 404:
            raise ImageGenerationError(
                f"{label} model not found or endpoint unavailable (404) at {response.request.url}: "
                f"{response.text[:300]}",
                retryable=False,
            )
        if response.status_code == 429:
            # Rate limit or quota exhausted; retrying after a delay may help.
            raise ImageGenerationError(
                f"{label} rate/quota limit hit (429). Try again shortly.",
                retryable=True,
            )
        if response.status_code >= 500:
            raise ImageGenerationError(
                f"{label} server error ({response.status_code}). Try again shortly.",
                retryable=True,
            )
        if response.status_code != 200:
            raise ImageGenerationError(
                f"{label} request failed ({response.status_code}): {response.text[:300]}"
            )

    # ------------------------------------------------------------------ #
    # Response resolution                                                  #
    # ------------------------------------------------------------------ #

    def _parse_json(self, response: httpx.Response) -> Any:
        """Parse the response body as JSON with a clear error message."""
        try:
            return response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                f"Unexpected {self.provider_label} response format: {response.text[:300]}"
            ) from exc

    async def _extract_image_bytes(self, payload: Any, response: httpx.Response) -> bytes:
        """Obtain the raw image bytes from a provider response.

        Resolution order:
        1. ``image/*`` response content-type -> the body itself is the image.
        2. First usable value under a known field (base64 or URL), anywhere
           in the JSON: data URLs are decoded, http(s) URLs downloaded, and
           raw base64-looking strings decoded.
        3. Lenient pass: any string under a base64 field is decode-attempted
           so invalid data raises a precise "invalid base64" error.
        4. Nothing usable -> ImageGenerationError.
        """
        # 1. Inline binary body.
        mime = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if mime.startswith("image/"):
            if response.content:
                return response.content
            raise ImageGenerationError(
                f"{self.provider_label} response contained no image data."
            )

        # 2. Strict walk over the union of known fields.
        value = self._find_first(payload, self.BASE64_FIELDS + self.URL_FIELDS)
        if value is not None:
            if value.startswith("data:"):
                return self._decode_base64_image(value)
            if value.startswith(("http://", "https://")):
                image_bytes, _content_type = await self._download_image(value)
                return image_bytes
            return self._decode_base64_image(value)

        # 3. Lenient pass: garbage under a base64 field still gets decoded
        #    so it fails with "invalid base64" rather than a format error.
        value = self._find_first_string(payload, self.BASE64_FIELDS + ("image",))
        if value is not None:
            return self._decode_base64_image(value)

        raise ImageGenerationError(
            f"Unexpected {self.provider_label} response format: {response.text[:300]}"
        )

    def _find_first(self, payload: Any, keys: tuple[str, ...]) -> str | None:
        """Depth-first search for the first usable str value under `keys`.

        Usable means a data URL, an http(s) URL, or a base64-looking string.
        Returns None when nothing usable is found.
        """
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if self._is_usable_image_value(value):
                    return value
            for value in payload.values():
                found = self._find_first(value, keys)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._find_first(item, keys)
                if found is not None:
                    return found
        return None

    def _find_first_string(self, payload: Any, keys: tuple[str, ...]) -> str | None:
        """Depth-first search for the first non-empty str value under `keys`."""
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
            for value in payload.values():
                found = self._find_first_string(value, keys)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._find_first_string(item, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _is_usable_image_value(value: Any) -> bool:
        """True for strings that are data URLs, http(s) URLs, or base64-ish."""
        if not isinstance(value, str) or not value:
            return False
        return (
            value.startswith(("data:", "http://", "https://"))
            or _looks_like_base64(value)
        )

    def _decode_base64_image(self, value: str) -> bytes:
        """Decode base64 image data, stripping any data-URL prefix."""
        _, sep, payload_b64 = value.partition(",")
        if not sep and not value.startswith("data:"):
            payload_b64 = value
        try:
            image_bytes = base64.b64decode(payload_b64)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError(
                f"{self.provider_label} returned invalid base64 image data."
            ) from exc
        if not image_bytes:
            raise ImageGenerationError(
                f"{self.provider_label} returned empty base64 image data."
            )
        return image_bytes

    # ------------------------------------------------------------------ #
    # Cost extraction                                                      #
    # ------------------------------------------------------------------ #

    def _find_first_dict(self, payload: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
        """Depth-first search for the first dict value under `keys`."""
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            for value in payload.values():
                found = self._find_first_dict(value, keys)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._find_first_dict(item, keys)
                if found is not None:
                    return found
        return None

    def _first_number(self, usage: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
        """First numeric (non-bool) value under `keys` in a usage object."""
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return None

    def _first_string(self, usage: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
        """First non-empty string value under `keys` in a usage object."""
        for key in keys:
            value = usage.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_per_image_costs(self, usage: Mapping[str, Any]) -> tuple[
        tuple[float, ...] | None,
        tuple[float, ...] | None,
        tuple[int, ...] | None,
    ]:
        """Read per-image breakdowns from a usage object, when reported.

        Understands either a parallel-lists shape (``per_image_costs`` /
        ``image_costs`` / ``per_image_tokens``) or a list-of-entries shape
        (``images``/``data``/``items`` with per-entry cost keys).
        """
        credits = amounts = tokens = None

        for key, target in (
            ("per_image_costs", "amount"), ("image_costs", "amount"),
            ("per_image_credits", "credits"), ("per_image_tokens", "tokens"),
        ):
            value = usage.get(key)
            if isinstance(value, list):
                numbers = tuple(
                    entry for entry in value
                    if isinstance(entry, (int, float)) and not isinstance(entry, bool)
                )
                if numbers and len(numbers) == len(value):
                    if target == "amount" and amounts is None:
                        amounts = numbers
                    elif target == "credits" and credits is None:
                        credits = numbers
                    elif target == "tokens" and tokens is None:
                        tokens = numbers

        if credits is None and amounts is None and tokens is None:
            for key in ("images", "data", "items"):
                entries = usage.get(key)
                if not isinstance(entries, list) or not entries:
                    continue
                if all(isinstance(entry, Mapping) for entry in entries):
                    per_amounts = tuple(
                        self._first_number(entry, self.AMOUNT_KEYS) for entry in entries
                    )
                    per_credits = tuple(
                        self._first_number(entry, self.CREDITS_KEYS) for entry in entries
                    )
                    per_tokens = tuple(
                        self._first_number(entry, self.TOKEN_KEYS) for entry in entries
                    )
                    if any(value is not None for value in per_amounts):
                        amounts = per_amounts
                    if any(value is not None for value in per_credits):
                        credits = per_credits
                    if any(value is not None for value in per_tokens):
                        tokens = per_tokens
                break

        return credits, amounts, tokens

    @staticmethod
    def _coerce_int_tuple(
        values: tuple[float | None, ...] | None,
    ) -> tuple[int | None, ...] | None:
        """Int-coerce a per-image tuple, preserving None entries."""
        if values is None:
            return None
        return tuple(int(value) if value is not None else None for value in values)

    def _extract_cost(self, parsed: Any) -> ImageGenerationCost | None:
        """Extract cost information from a parsed provider response.

        The default implementation depth-first locates a usage/cost object
        under :attr:`USAGE_FIELDS` and maps well-known keys for total amounts,
        credits, tokens, image count, and per-image breakdowns. Returns
        ``None`` when nothing usable is found (unknown cost, not free).
        Subclasses override for exotic response shapes.
        """
        usage = self._find_first_dict(parsed, self.USAGE_FIELDS)
        if usage is None:
            return None

        amount = self._first_number(usage, self.AMOUNT_KEYS)
        credits = self._first_number(usage, self.CREDITS_KEYS)
        tokens = self._first_number(usage, self.TOKEN_KEYS)
        image_count = self._first_number(usage, self.IMAGE_COUNT_KEYS)
        currency = self._first_string(usage, self.CURRENCY_KEYS)

        if all(value is None for value in (amount, credits, tokens, image_count)):
            return None

        per_credits, per_amounts, per_tokens = self._extract_per_image_costs(usage)

        return ImageGenerationCost(
            total_credits=credits,
            total_amount=amount,
            currency=currency,
            total_tokens=int(tokens) if tokens is not None else None,
            image_count=int(image_count) if image_count is not None else None,
            per_image_credits=per_credits,
            per_image_amounts=per_amounts,
            per_image_tokens=self._coerce_int_tuple(per_tokens),
            raw_usage=usage,
        )

    # ------------------------------------------------------------------ #
    # Output handling                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sniff_suffix(image_bytes: bytes, content_type: str | None = None) -> str:
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

    def _write_temp_image(self, image_bytes: bytes, suffix: str, prefix: str) -> Path:
        """Write image bytes to a temp file and validate it is non-empty."""
        if not image_bytes:
            raise ImageGenerationError(
                f"{self.provider_label} response contained no image data."
            )
        # delete=False keeps the file alive after close; the base class
        # moves it into the media folder afterwards.
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="wb", suffix=suffix, prefix=prefix, delete=False
        )
        try:
            with tmp as f:
                f.write(image_bytes)
        except OSError as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise ImageGenerationError(
                f"Failed to save {self.provider_label} image: {exc}"
            ) from exc

        img_path = Path(tmp.name)
        if not img_path.is_file() or img_path.stat().st_size == 0:
            img_path.unlink(missing_ok=True)
            raise ImageGenerationError(
                f"{self.provider_label} image download produced an empty file."
            )
        return img_path

    # ------------------------------------------------------------------ #
    # Template method                                                      #
    # ------------------------------------------------------------------ #

    async def _generate_once_async(self, request: ImageGenerationRequest) -> ImageGenerationAttempt:
        """Run one provider attempt: POST, resolve, and save to a temp file.

        Returns an ImageGenerationAttempt carrying the cost extracted from the
        response (``None`` when the provider reports none) plus the raw JSON
        response body for caller-side parsing.
        """
        payload = self._build_payload(request)
        response = await self._post_json(self._endpoint, payload)
        return await self._resolve_and_save(request, response)

    async def _resolve_and_save(
        self, request: ImageGenerationRequest, response: httpx.Response
    ) -> ImageGenerationAttempt:
        """Shared post-response pipeline: status, parse, extract, save, cost."""
        self._raise_for_status(response)
        parsed = self._parse_json(response)
        raw_response_json = response.text if response.text else None
        image_bytes = await self._extract_image_bytes(parsed, response)
        suffix = self._sniff_suffix(image_bytes, response.headers.get("content-type"))
        path = self._write_temp_image(image_bytes, suffix, prefix=self._temp_prefix)
        return ImageGenerationAttempt(
            path=path,
            cost=self._extract_cost(parsed),
            raw_response_json=raw_response_json,
        )

    @property
    def _temp_prefix(self) -> str:
        """Prefix for generated temp files, derived from the label."""
        return f"{self.provider_label.lower().replace(' ', '_')}_"


def _looks_like_base64(value: str) -> bool:
    """Cheap heuristic: base64 charset without whitespace for a decent length."""
    if len(value) < 16 or len(value) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value) is not None
