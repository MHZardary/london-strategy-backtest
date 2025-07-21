import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import yaml

from src import mt5_connector
from src import trade_logic
from src import result_analyse

env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

def load_env_symbols() -> dict:
    """
    Loads currency pair symbols and time zone from environment variables.
    :return: dict mapping currency pair labels and time zone
    """

    return {
        "EURUSD": os.getenv("EURUSD_SYMBOL"),
        "GBPUSD": os.getenv("GBPUSD_SYMBOL"),
        "GBPJPY": os.getenv("GBPJPY_SYMBOL"),
        "time_zone": os.getenv("MTN_TIMEZONE")
    }

def load_config_settings() -> dict:
    """
    Loads YAML configuration settings from the config directory.
    :return: dict of configuration settings
    """

    config_path = Path(__file__).resolve().parent / "config" / "setting.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def ensure_market_data_exists(symbols_list: list, data_config: dict)-> None:
    """
    Ensures historical market data CSVs exist, fetching missing ones from MT5.

    :param symbols_list: list of symbol strings to check
    :param data_config: dict with keys 'time_frame', 'start_date', 'end_date'
    :return: None
    """

    load_config_settings()
    for single_symbol in symbols_list:
        file_path = os.path.join("data", "market", f"data_{single_symbol}.csv")
        if os.path.isfile(file_path):
            continue
        else:
            df_data_single_symbol = mt5_connector.get_historical_data(single_symbol, data_config["time_frame"], data_config["start_date"], data_config["end_date"])
            df_data_single_symbol.to_csv(file_path)

def clean_exchange_rates(rates: dict) -> dict:
    """
    Cleans and converts exchange rate strings to float values.

    :param rates: dict mapping symbol to rate string (may contain commas)
    :return: dict mapping symbol to float rate
    """
    cleaned = {}
    for symbol, rate in rates.items():
        cleaned[symbol] = float(str(rate).replace(',', '').strip())
    return cleaned

def run_london_breakout_analysis(
        use_rsi_filter: bool=False,
        rsi_period: int=14,
        use_macd_filter: bool=False,
        use_ma_filter: bool = False,
        ma_window: int=50,
        use_news_filter: bool=False
) -> None:
    """
    Executes the full London Breakout trading analysis pipeline.

    :param use_rsi_filter: whether to apply RSI filter on entry
    :param rsi_period: lookback period for RSI calculation
    :param use_macd_filter: whether to apply MACD filter on entry
    :param use_ma_filter: whether to apply moving average filter on entry
    :param ma_window: window size for moving average
    :param use_news_filter: whether to avoid entries on high-impact news days
    :return: None
    """

    env_pars = load_env_symbols()
    config_pars = load_config_settings()
    data_config = config_pars['data_config']
    symbol_labels = ['EURUSD', 'GBPUSD', 'GBPJPY']
    symbols = [env_pars["EURUSD"], env_pars["GBPUSD"], env_pars["GBPJPY"]]
    ensure_market_data_exists(symbols, data_config)
    all_positions = []
    all_market_data = []
    for symbol_name in symbol_labels:
        market_df = trade_logic.enrich_market_data(env_pars[symbol_name], rsi_period, ma_window, config_pars['signal_settings'],
                                                   env_pars['time_zone'])
        market_df = trade_logic.enter_signal_extractor(market_df, use_rsi_filter, use_macd_filter, use_ma_filter, use_news_filter, config_pars['signal_settings'])
        market_df = trade_logic.exit_signal_extractor(market_df, config_pars['signal_settings'])
        market_df['symbol'] = symbol_name
        market_df['datetime'] = market_df.index
        final_positions_df = result_analyse.candles_to_positions(market_df)
        all_positions.append(final_positions_df)
        all_market_data.append(market_df)

    final_positions_df = pd.concat(all_positions, ignore_index=True)
    market_df = pd.concat(all_market_data, ignore_index=True)
    rates = clean_exchange_rates(config_pars['rates'])
    result_analyse.positions_analyse(final_positions_df, rates)
    result_analyse.positions_plotter(final_positions_df, rates)

    result_analyse.buy_true_sample_plotter(market_df, final_positions_df)
    result_analyse.buy_false_sample_plotter(market_df, final_positions_df)
    result_analyse.sell_true_sample_plotter(market_df, final_positions_df)
    result_analyse.sell_false_sample_plotter(market_df, final_positions_df)

if __name__=="__main__":
    run_london_breakout_analysis(use_ma_filter=True, use_news_filter=True)
