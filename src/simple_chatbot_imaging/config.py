"""Provider configuration helpers and default endpoint settings."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "providers.json"

DEFAULT_PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    "openrouter": {
        "kwargs": {
            "base_url": "https://openrouter.ai/api/v1/images",
            "model": "meta/muse-image",
            "api_key_env": "OPENROUTER_API_KEY",
            "request_timeout_seconds": 120,
        }
    },
    "huggingface": {
        "kwargs": {
            "model": "hugging-apps/qwen-image-2-1",
            "api_key_env": "HUGGINGFACE_ACCESS_TOKEN",
        }
    },
    "qwen": {
        "kwargs": {
            "base_url": "https://maas.qwencloudapi.com/api/v1/services/aigc/multimodal-generation/generation",
            "model": "qwen-image-3.0",
            "api_key_env": "DASHSCOPE_API_KEY",
            "request_timeout_seconds": 60,
        }
    },
    "sdxl": {
        "kwargs": {
            "model": "SG161222/RealVisXL_V5.0_Lightning",
            "vae": "madebyollin/sdxl-vae-fp16-fix",
            "device": "cuda:0",
            "steps": 7,
            "guidance_scale": 1.5,
        }
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries while preserving existing defaults."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_provider_config(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load provider config from disk, merging it with the built-in defaults.

    The JSON file can define provider-specific kwargs such as "base_url" or custom
    provider entries with a "factory" reference ("module:callable").
    """
    config_obj = copy.deepcopy(DEFAULT_PROVIDER_CONFIG)
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not config_file.exists():
        return config_obj

    with config_file.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError("Provider config must be a JSON object at the top level.")

    return _deep_merge(config_obj, loaded)


def _load_factory_from_config(provider_name: str, provider_config: Mapping[str, Any]) -> Any:
    """Resolve a custom provider factory from a configuration entry."""
    factory_ref = provider_config.get("factory")
    if factory_ref is None:
        raise ValueError(
            f"Provider '{provider_name}' is missing a 'factory' entry in the config."
        )

    if callable(factory_ref):
        return factory_ref

    if not isinstance(factory_ref, str):
        raise ValueError(
            f"Provider '{provider_name}' has an invalid factory entry in the config."
        )

    if ":" not in factory_ref:
        raise ValueError(
            f"Provider '{provider_name}' factory must use the format 'module:callable'."
        )

    module_name, attr_name = factory_ref.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError(
            f"Provider '{provider_name}' factory must use the format 'module:callable'."
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise ValueError(f"Configured factory for provider '{provider_name}' is not callable.")
    return factory
