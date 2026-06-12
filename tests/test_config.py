"""
Unit tests for ConfigManager.
"""

import json
import logging

import pytest

from qaoa_portfolio.config import ConfigManager
from qaoa_portfolio.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(config_path=str(tmp_path / "settings.json"))


class TestDefaults:
    def test_defaults_loaded_when_no_file_exists(self, manager):
        assert manager.get("data_sources.default") == "yfinance"
        assert manager.get("portfolio.default_size") == 5
        assert manager.get("free_tier.yahoo_finance.enabled") is True

    def test_get_returns_default_for_missing_keys(self, manager):
        assert manager.get("does.not.exist") is None
        assert manager.get("does.not.exist", "fallback") == "fallback"
        # Traversing through a non-dict leaf falls back too
        assert manager.get("data_sources.default.deeper", 7) == 7


class TestSetAndGet:
    def test_set_overwrites_existing_value(self, manager):
        manager.set("portfolio.default_size", 9)
        assert manager.get("portfolio.default_size") == 9

    def test_set_creates_nested_sections(self, manager):
        manager.set("brand.new.key", "value")
        assert manager.get("brand.new.key") == "value"


class TestPersistence:
    def test_save_and_reload_round_trip(self, tmp_path):
        path = tmp_path / "nested" / "settings.json"
        manager = ConfigManager(config_path=str(path))
        manager.set("portfolio.default_size", 11)
        manager.save_config()

        assert path.exists()
        reloaded = ConfigManager(config_path=str(path))
        assert reloaded.get("portfolio.default_size") == 11

    def test_file_config_deep_merges_into_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"portfolio": {"default_size": 3}}))

        manager = ConfigManager(config_path=str(path))
        assert manager.get("portfolio.default_size") == 3
        # Sibling defaults survive the merge
        assert manager.get("portfolio.risk_free_rate") == 0.02

    def test_invalid_json_raises_configuration_error(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{not valid json")

        with pytest.raises(ConfigurationError, match="Failed to load"):
            ConfigManager(config_path=str(path))


class TestSetupLogging:
    def test_setup_logging_applies_configured_level(self, manager, monkeypatch):
        recorded = {}

        def fake_basic_config(**kwargs):
            recorded.update(kwargs)

        monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
        manager.set("logging.level", "debug")
        manager.setup_logging()

        assert recorded["level"] == logging.DEBUG
        assert recorded["format"] == manager.get("logging.format")
