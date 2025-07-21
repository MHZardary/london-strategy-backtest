import os
import pytest
import pandas as pd
import yaml
from unittest import mock
from pathlib import Path

# Import functions from your main module
from main import (
    load_env_symbols,
    load_config_settings,
    ensure_market_data_exists,
    clean_exchange_rates,
)

# ----------------------------
# Test: load_env_symbols
# ----------------------------
def test_load_env_symbols(monkeypatch):
    monkeypatch.setenv("EURUSD_SYMBOL", "EURUSD.f")
    monkeypatch.setenv("GBPUSD_SYMBOL", "GBPUSD.f")
    monkeypatch.setenv("GBPJPY_SYMBOL", "GBPJPY.f")
    monkeypatch.setenv("MTN_TIMEZONE", "Europe/London")

    result = load_env_symbols()
    assert result == {
        "EURUSD": "EURUSD.f",
        "GBPUSD": "GBPUSD.f",
        "GBPJPY": "GBPJPY.f",
        "time_zone": "Europe/London",
    }

# ----------------------------
# Test: load_config_settings
# ----------------------------
def test_load_config_settings(tmp_path, monkeypatch):
    dummy_config = {
        "data_config": {"time_frame": "H1", "start_date": "2023-01-01", "end_date": "2023-12-31"},
        "signal_settings": {"rsi": 14},
        "rates": {"EURUSD": "1.1"},
    }

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "setting.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(dummy_config, f)

    monkeypatch.setattr(Path, "resolve", lambda self: tmp_path)
    monkeypatch.setattr(Path, "parent", tmp_path)

    result = load_config_settings()
    assert result["data_config"]["time_frame"] == "H1"

# ----------------------------
# Test: ensure_market_data_exists
# ----------------------------
@mock.patch("src.mt5_connector.get_historical_data")
def test_ensure_market_data_exists_creates_file_if_missing(mock_get_data, tmp_path, monkeypatch):
    dummy_df = pd.DataFrame({
        "time": ["2024-01-01 00:00:00"],
        "open": [1.1],
        "high": [1.2],
        "low": [1.0],
        "close": [1.15],
    })
    mock_get_data.return_value = dummy_df

    monkeypatch.chdir(tmp_path)
    os.makedirs("data/market", exist_ok=True)

    symbols = ["TESTUSD"]
    config = {"time_frame": "H1", "start_date": "2024-01-01", "end_date": "2024-01-31"}

    ensure_market_data_exists(symbols, config)

    expected_file = tmp_path / "data" / "market" / "data_TESTUSD.csv"
    assert expected_file.exists()

# ----------------------------
# Test: clean_exchange_rates
# ----------------------------
def test_clean_exchange_rates():
    raw_rates = {
        "EURUSD": "1,234.56",
        "GBPUSD": "1.1234",
        "GBPJPY": "145.67 ",
    }

    cleaned = clean_exchange_rates(raw_rates)
    assert cleaned["EURUSD"] == 1234.56
    assert cleaned["GBPUSD"] == 1.1234
    assert cleaned["GBPJPY"] == 145.67
