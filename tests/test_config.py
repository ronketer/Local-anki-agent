"""Tests for typed configuration validation."""

import pytest

from anki_pipeline.config import Config
from anki_pipeline.errors import ConfigurationError


def test_missing_block_id_raises_typed_configuration_error() -> None:
    config = Config()
    config.TARGET_BLOCK_ID = ""

    with pytest.raises(ConfigurationError, match="TARGET_BLOCK_ID is required"):
        config.require_valid()


def test_runtime_block_override_is_honored() -> None:
    config = Config()
    config.TARGET_BLOCK_ID = "20260101000000-test"

    config.require_valid()

    assert config.validate() == []
