"""Tests for configuration loading."""

from pathlib import Path

import yaml

from src.config import AppConfig, load_config


def test_default_config():
    """Default config should have sane values."""
    config = AppConfig()
    assert config.trading.capital == 10000
    assert config.trading.risk_per_trade_pct == 0.02
    assert config.trading.max_drawdown_pct == 0.10
    assert config.broker.paper is True


def test_load_from_yaml(tmp_path):
    settings = {
        "trading": {"capital": 5000, "mode": "paper"},
        "data": {"db_path": "test.db"},
    }
    path = tmp_path / "settings.yaml"
    with open(path, "w") as f:
        yaml.dump(settings, f)

    config = load_config(settings_path=path, secrets_path=tmp_path / "nope.yaml")
    assert config.trading.capital == 5000
    assert config.trading.mode == "paper"


def test_missing_settings_uses_defaults(tmp_path):
    config = load_config(
        settings_path=tmp_path / "nonexistent.yaml",
        secrets_path=tmp_path / "nonexistent.yaml",
    )
    assert config.trading.capital == 10000
