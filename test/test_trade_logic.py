import pandas as pd
from datetime import datetime, time

from src.trade_logic import (
    compute_asian_high,
    compute_asian_low,
    enter_signal_condition
)

# ----------------------------
# Test: compute_asian_high
# ----------------------------
def test_compute_asian_high():
    data = {
        "time": pd.date_range("2024-01-01 00:00", periods=5, freq="h"),
        "high": [1.1, 1.2, 1.3, 1.25, 1.15]
    }
    df = pd.DataFrame(data).set_index("time")
    high_series = compute_asian_high(df, [time(1, 0), time(4, 0)])
    assert "Asian_high" in high_series
    assert abs(high_series["Asian_high"] - 1.3) < 1e-6

# ----------------------------
# Test: compute_asian_low
# ----------------------------
def test_compute_asian_low():
    data = {
        "time": pd.date_range("2024-01-01 00:00", periods=5, freq="h"),
        "low": [1.1, 1.0, 1.05, 1.15, 1.2]
    }
    df = pd.DataFrame(data).set_index("time")
    low_series = compute_asian_low(df, [time(1, 0), time(4, 0)])
    assert "Asian_low" in low_series
    assert abs(low_series["Asian_low"] - 1.0) < 1e-6

# ----------------------------
# Test: enter_signal_condition (buy signal)
# ----------------------------
def test_enter_signal_condition_buy_signal():
    row = pd.Series({
        "close": 1.25,
        "Asian_high": 1.2,
        "Asian_low": 1.1,
        "news": 0,
        "rsi": 60,
        "macd": 1.0,
        "macd_signal": 0.5,
        "ma_diff": 0.02
    })
    signal = enter_signal_condition(row, news_check=True, rsi_check=True, macd_check=False, ma_check=True)
    assert signal == 1

# ----------------------------
# Test: enter_signal_condition blocked by RSI
# ----------------------------
def test_enter_signal_condition_blocked_by_rsi():
    row = pd.Series({
        "close": 1.25,
        "Asian_high": 1.2,
        "Asian_low": 1.1,
        "news": 0,
        "rsi": 50,
        "macd": 1.0,
        "macd_signal": 0.5,
        "ma_diff": 0.02
    })
    signal = enter_signal_condition(row, news_check=False, rsi_check=True, macd_check=False, ma_check=False)
    assert signal == 0
