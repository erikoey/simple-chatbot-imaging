"""Tests for the Hugging Face provider: lazy import and error mapping."""

import sys
import types

import pytest

from simple_chatbot_imaging.base import ImageGenerationError
from simple_chatbot_imaging.factory import create_image_generator


def test_import_package_does_not_load_gradio_client():
    import simple_chatbot_imaging

    assert "gradio_client" not in sys.modules


def test_factory_missing_extra_clear_error(monkeypatch):
    """When gradio_client is not importable, the factory gives a clear error."""
    # Ensure gradio_client is not importable in this test run.
    monkeypatch.setitem(sys.modules, "gradio_client", None)
    monkeypatch.setitem(sys.modules, "gradio_client.exceptions", None)
    monkeypatch.setitem(sys.modules, "gradio_client.utils", None)

    import importlib
    import simple_chatbot_imaging.huggingface as hf_mod

    importlib.reload(hf_mod)
    assert hf_mod.HAS_GRADIO_CLIENT is False

    gen = create_image_generator("huggingface", media_path="media_test_hf")
    with pytest.raises(ImageGenerationError, match="huggingface' extra"):
        _ = gen.client


def test_huggingface_generator_constructible_when_extra_missing(monkeypatch):
    """Construction must not fail; only client use requires the extra."""
    monkeypatch.setitem(sys.modules, "gradio_client", None)
    import importlib
    import simple_chatbot_imaging.huggingface as hf_mod

    importlib.reload(hf_mod)
    gen = hf_mod.HuggingFaceImageGenerator(media_path="media_test_hf2")
    assert gen.space_id == hf_mod.HuggingFaceImageGenerator.DEFAULT_SPACE_ID
    assert gen.token is None