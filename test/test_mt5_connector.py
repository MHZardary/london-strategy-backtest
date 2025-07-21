import os
import pytest
from unittest import mock
from datetime import datetime
import pandas as pd

from src import mt5_connector

# ----------------------------
# Test: load_mt5_credentials
# ----------------------------
def test_load_mt5_credentials(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "testpass")
    monkeypatch.setenv("MT5_SERVER", "TestServer")
    monkeypatch.setenv("MT5_PATH", "/path/to/terminal.exe")

    creds = mt5_connector.load_mt5_credentials()

    assert creds["login"] == 123456
    assert creds["password"] == "testpass"
    assert creds["server"] == "TestServer"
    assert creds["path"] == "/path/to/terminal.exe"

# ----------------------------
# Test: initialize_mt5_connection
# ----------------------------
@mock.patch("src.mt5_connector.mt5.initialize")
@mock.patch("src.mt5_connector.load_mt5_credentials")
def test_initialize_mt5_connection_success(mock_creds, mock_mt5_init):
    mock_creds.return_value = {
        "login": 123456,
        "password": "testpass",
        "server": "TestServer",
        "path": "/fake/path"
    }
    mock_mt5_init.return_value = True

    result = mt5_connector.initialize_mt5_connection()
    assert result is True

# ----------------------------
# Test: get_historical_data
# ----------------------------
@mock.patch("src.mt5_connector.mt5.copy_rates_range")
@mock.patch("src.mt5_connector.mt5.initialize")
@mock.patch("src.mt5_connector.mt5.shutdown")
@mock.patch("src.mt5_connector.load_mt5_credentials")
def test_get_historical_data_returns_dataframe(mock_creds, mock_shutdown, mock_init, mock_copy_rates):
    mock_creds.return_value = {
        "login": 123456,
        "password": "testpass",
        "server": "TestServer",
        "path": "/fake/path"
    }
    mock_init.return_value = True

    # Mocking MT5 copy_rates_range to return dummy data
    mock_copy_rates.return_value = [{
        "time": int(datetime(2023, 1, 1, 0, 0).timestamp()),
        "open": 1.1,
        "high": 1.2,
        "low": 1.0,
        "close": 1.15,
        "tick_volume": 1000,
        "spread": 10,
        "real_volume": 1000
    }]

    df = mt5_connector.get_historical_data("EURUSD", "H1", "2023-01-01", "2023-01-02")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "open" in df.columns
