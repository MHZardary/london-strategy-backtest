import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
import MetaTrader5 as mt5

env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

def load_mt5_credentials() -> dict:
    """
    Loads MetaTrader 5 credentials from environment variables.
    :return: dict with keys 'login','password','server','path'
    """
    return {
        "login": int(os.getenv("MT5_LOGIN")),
        "password": os.getenv("MT5_PASSWORD"),
        "server": os.getenv("MT5_SERVER"),
        "path": os.getenv("MT5_PATH")
    }

def initialize_mt5_connection() -> bool:
    """
    Initializes connection to the MetaTrader 5 terminal.

    :return: True if successful, False otherwise
    """
    creds = load_mt5_credentials()
    return mt5.initialize(path=creds["path"], login=creds["login"], password=creds["password"], server=creds["server"])

def get_historical_data(symbol: str, timeframe_str: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves historical OHLC data from MT5 for a symbol over a date range.

    :param symbol: string trading symbol
    :param timeframe_str: timeframe code (e.g. 'H1','M15')
    :param start_date: 'YYYY-MM-DD' string of start
    :param end_date: 'YYYY-MM-DD' string of end
    :return: pd.DataFrame of rates indexed by datetime, or None if failure
    """

    if not(initialize_mt5_connection()):
        return pd.DataFrame()

    timeframe_mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }

    timeframe = timeframe_mapping.get(timeframe_str)

    rates = mt5.copy_rates_range(symbol, timeframe, datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d"))
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df
