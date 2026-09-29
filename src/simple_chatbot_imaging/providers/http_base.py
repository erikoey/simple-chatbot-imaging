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
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from simple_chatbot_imaging.base import BaseImageGenerator
from simple_chatbot_imaging.models import ImageGenerationError, ImageGenerationRequest


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

    async def _generate_once_async(self, request: ImageGenerationRequest) -> Path:
        """Run one provider attempt: POST, resolve, and save to a temp file."""
        payload = self._build_payload(request)
        response = await self._post_json(self._endpoint, payload)
        self._raise_for_status(response)
        parsed = self._parse_json(response)
        image_bytes = await self._extract_image_bytes(parsed, response)
        suffix = self._sniff_suffix(image_bytes, response.headers.get("content-type"))
        return self._write_temp_image(image_bytes, suffix, prefix=self._temp_prefix)

    @property
    def _temp_prefix(self) -> str:
        """Prefix for generated temp files, derived from the label."""
        return f"{self.provider_label.lower().replace(' ', '_')}_"


def _looks_like_base64(value: str) -> bool:
    """Cheap heuristic: base64 charset without whitespace for a decent length."""
    if len(value) < 16 or len(value) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value) is not None
