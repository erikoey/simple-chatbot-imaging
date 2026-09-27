"""Provider registry, factory, and opt-in fallback orchestration."""

from pathlib import Path
from typing import Callable
import importlib

from simple_chatbot_imaging.base import (
    BaseImageGenerator,
    ImageGenerationError,
    ImageGenerationState,
)
from simple_chatbot_imaging.config import _load_factory_from_config, merge_provider_kwargs

#: Registry of known provider factories. Values are lazy loaders so that
#: importing this module never pulls in optional dependencies.
PROVIDERS: dict[str, Callable[..., BaseImageGenerator]] = {}


def _load_provider_factory(name: str, config_path: str | Path | None = None) -> Callable[..., BaseImageGenerator]:
    """Resolve a provider factory from the built-in registry or custom config."""
    provider_name = name.lower()
    if provider_name in PROVIDERS:
        return PROVIDERS[provider_name]

    config = __import__("simple_chatbot_imaging.config", fromlist=["load_provider_config"]).load_provider_config(config_path)
    if provider_name not in config:
        raise ValueError(f"Unknown image provider '{provider_name}'")

    provider_config = config[provider_name]
    if "factory" not in provider_config:
        raise ValueError(f"Provider '{provider_name}' is not configured with a factory entry.")

    factory = _load_factory_from_config(provider_name, provider_config)
    PROVIDERS[provider_name] = factory
    return factory


def _load_openrouter(**kwargs) -> BaseImageGenerator:
    from simple_chatbot_imaging.providers.openrouter import OpenRouterImageGenerator

    return OpenRouterImageGenerator(**kwargs)


def _load_huggingface(**kwargs) -> BaseImageGenerator:
    try:
        from simple_chatbot_imaging.providers.huggingface import HuggingFaceImageGenerator
    except ImportError as exc:
        raise ImageGenerationError(
            "The Hugging Face provider requires the 'huggingface' extra. "
            "Install it with: uv add 'simple-chatbot-imaging[huggingface]'",
            retryable=False,
        ) from exc
    return HuggingFaceImageGenerator(**kwargs)


def _load_qwen(**kwargs) -> BaseImageGenerator:
    from simple_chatbot_imaging.providers.qwen import QwenImageGenerator
    return QwenImageGenerator(**kwargs)


PROVIDERS["openrouter"] = _load_openrouter
PROVIDERS["huggingface"] = _load_huggingface
PROVIDERS["qwen"] = _load_qwen


def register_provider(name: str, factory: Callable[..., BaseImageGenerator]) -> None:
    """Add a new provider to the registry (useful for plugins)."""
    PROVIDERS[name.lower()] = factory


def create_image_generator(
    provider: str,
    config_path: str | Path | None = None,
    **kwargs,
) -> BaseImageGenerator:
    """Create an image generator by provider name.

    Args:
        provider: One of the registry keys ("huggingface", "openrouter").
            There is no implicit fallback default; compose providers
            explicitly with :class:`FallbackImageGenerator`.
        config_path: Optional JSON file containing provider defaults and custom
            registrations.
        **kwargs: Forwarded to the provider constructor (e.g. media_path).

    Raises:
        ValueError: If the provider name is unknown.
        ImageGenerationError: If an optional provider's extra is missing.
    """
    provider = provider.lower()

    if provider in PROVIDERS:
        factory = PROVIDERS[provider]
    else:
        config = __import__("simple_chatbot_imaging.config", fromlist=["load_provider_config"]).load_provider_config(config_path)
        if provider not in config:
            known = ", ".join(sorted(set(PROVIDERS) | set(config)))
            raise ValueError(f"Unknown image provider '{provider}'. Known providers: {known}")

        provider_cfg = config[provider]
        if "factory" in provider_cfg:
            factory = _load_factory_from_config(provider, provider_cfg)
            PROVIDERS[provider] = factory
        else:
            factory = PROVIDERS.get(provider)
            if factory is None:
                known = ", ".join(sorted(set(PROVIDERS) | set(config)))
                raise ValueError(f"Unknown image provider '{provider}'. Known providers: {known}")

    merged_kwargs = merge_provider_kwargs(provider, config_path, **kwargs)
    return factory(**merged_kwargs)


class FallbackImageGenerator(BaseImageGenerator):
    """Try several image generators in order; the first success wins.

    Opt-in and caller-defined: pass fully constructed generators (e.g. built
    via :func:`create_image_generator`). Each provider runs its own full
    lifecycle (retries, state tracking, output movement); this class only
    advances to the next provider when the previous one failed.
    """

    def __init__(self,
                 generators: list[BaseImageGenerator],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if not generators:
            raise ValueError("FallbackImageGenerator requires a non-empty generator chain.")

        self.generators = list(generators)
        self.last_backend: str | None = None

    @property
    def provider_name(self) -> str:
        return "Fallback"

    async def _generate_once_async(self,
                                   prompt: str,
                                   negative_prompt: str,
                                   resolution: int,
                                   aspect_ratio: str,
                                   steps: int,
                                   seed: int) -> Path:
        """Try each generator in the chain; return the first successful result.

        Each generator's public async entry point is used so retries, state
        tracking, and output movement run inside that provider. The final
        successful provider's moved output is returned; this class's own
        base-class movement then relocates it to the requested filename.
        """
        errors: list[str] = []

        for generator in self.generators:
            try:
                result = await generator.generate_image_async(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    steps=steps,
                    seed=seed,
                    filename=f"fallback_{generator.provider_name}.png",
                )
                self.last_backend = generator.provider_name
                print(f"[Fallback] image generated via {generator.provider_name}")
                return Path(result)
            except (ImageGenerationError, ValueError) as exc:
                errors.append(f"{generator.provider_name}: {exc}")
                print(f"[Fallback] {generator.provider_name} failed: {exc}")

        raise ImageGenerationError(
            "All image providers failed:\n- " + "\n- ".join(errors)
        )