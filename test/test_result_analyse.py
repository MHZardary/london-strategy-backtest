import pandas as pd
from datetime import datetime
import tempfile
from pathlib import Path

from src.result_analyse import candles_to_positions, positions_analyse


# ----------------------------
# Test: candles_to_positions
# ----------------------------
def test_candles_to_positions_extracts_positions():
    # Simulated time series with position signals
    data = {
        "datetime": pd.date_range("2024-01-01", periods=6, freq="h"),
        "close": [1.10, 1.12, 1.14, 1.15, 1.13, 1.11],
        "position": [0, 1, 1, 0, -1, 0],
        "symbol": ["EURUSD"] * 6
    }
    df = pd.DataFrame(data).set_index("datetime")

    trades = candles_to_positions(df)

    assert len(trades) == 2
    assert trades.iloc[0]["direction"] == "buy"
    assert trades.iloc[1]["direction"] == "sell"
    assert abs(trades.iloc[0]["price_change"] - 0.03) < 1e-6
    assert abs(trades.iloc[1]["price_change"] - 0.02) < 1e-6
