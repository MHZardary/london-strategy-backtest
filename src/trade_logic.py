import os
from datetime import time

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def load_env_symbols() -> dict:
    """
    Loads currency pair symbols from environment variables.
    :return: dict mapping currency labels to MT5 symbols
    """
    return {
        "EURUSD": os.getenv("EURUSD_SYMBOL"),
        "GBPUSD": os.getenv("GBPUSD_SYMBOL"),
        "GBPJPY": os.getenv("GBPJPY_SYMBOL")
    }

def compute_asian_high(group: pd.DataFrame, asian_time_range: list) -> pd.Series:
    """
    Computes the high price during the specified Asian session for a group of OHLC data.

    :param group: Pandas DataFrame grouped by date
    :param asian_time_range: list [start_time, end_time] of Asian session
    :return: Pandas Series which include the Asian_high value
    """
    asian_session = group.between_time(asian_time_range[0], asian_time_range[1])
    return pd.Series({'Asian_high': asian_session['high'].max()})

def compute_asian_low(group: pd.DataFrame, asian_time_range: list) -> pd.Series:
    """
    Computes the low price during the specified Asian session for a group of OHLC data.

    :param group: Pandas DataFrame grouped by date
    :param asian_time_range: list [start_time, end_time] of Asian session
    :return: Pandas Series which include the Asian_low value
    """

    asian_session = group.between_time(asian_time_range[0], asian_time_range[1])
    return pd.Series({'Asian_low': asian_session['low'].min()})

def enrich_market_data(symbol: str, rsi_var: int, ma_var: int, signal_config: dict, timezone_of_data: str) -> pd.DataFrame:
    """
    Loads raw data CSV and enriches it with technical indicators and news flags.

    :param symbol: string MT5 symbol to load
    :param rsi_var: window length for RSI indicator
    :param ma_var: window length for moving average
    :param signal_config: dict with signal timing and range settings
    :param timezone_of_data: original timezone of the CSV data
    :return: pd.DataFrame enriched with Asian ranges, RSI, MACD, MA, and news flag
    """

    env_pars = load_env_symbols()

    if symbol == env_pars["EURUSD"]:
        currencies = ['EUR', 'USD']

    elif symbol == env_pars["GBPUSD"]:
        currencies = ['GBP', 'USD']

    else:
        currencies = ['GBP', 'JPY']

    file_path = os.path.join("data", "market", f"data_{symbol}.csv")

    df = pd.read_csv(file_path)

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    df.index = df.index.tz_localize(timezone_of_data).tz_convert('UTC')

    asian_time_range = [signal_config['asian_time_range_start'], signal_config['asian_time_range_end']]

    asian_high_df = df.groupby(df.index.date).apply(compute_asian_high, asian_time_range)
    asian_high_df.index = pd.to_datetime(asian_high_df.index).tz_localize('GMT')
    df['Asian_high'] = df.index.normalize().map(asian_high_df['Asian_high'])

    asian_low_df = df.groupby(df.index.date).apply(compute_asian_low, asian_time_range)
    asian_low_df.index = pd.to_datetime(asian_low_df.index).tz_localize('GMT')
    df['Asian_low'] = df.index.normalize().map(asian_low_df['Asian_low'])

    df['rsi'] = RSIIndicator(close=df['close'], window=rsi_var).rsi()

    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    sma = SMAIndicator(close=df['close'], window=ma_var)
    df['ma'] = sma.sma_indicator()
    df['ma_diff'] = df['ma'].diff()

    df['Asian_range'] = df['Asian_high'] - df['Asian_low']

    news_path = os.path.join("data", "news", "news.xlsx")
    news = pd.read_excel(news_path)
    news['day'] = pd.to_datetime(news['day'])
    news.set_index('day', inplace=True)

    news['combined'] = news[currencies[0]] | news[currencies[1]]

    df['news'] = None

    df['date'] = df.index.date
    daily_values = news['combined'].to_dict()

    df['news'] = df['date'].map(daily_values)

    df = df.drop(columns='date')

    return df

def enter_signal_condition(df_row: pd.Series, news_check: bool, rsi_check: bool, macd_check: bool, ma_check: bool) -> int:
    """
    Evaluates entry signal conditions for a single data row.

    :param df_row: a single row of enriched market data
    :param news_check: whether to block on-news-day entries
    :param rsi_check: whether to apply RSI threshold
    :param macd_check: whether to apply MACD threshold
    :param ma_check: whether to apply MA differential threshold
    :return: int 1 for buy, -1 for sell, 0 for no trade
    """

    if (df_row['close'] < df_row['Asian_high']) and (df_row['close'] > df_row['Asian_low']):
        return 0
    elif df_row['close'] > df_row['Asian_high']:
        if news_check and (df_row['news']==1):
            return 0
        elif rsi_check and (df_row['rsi']<55):
            return 0
        elif macd_check and (df_row['macd'] > df_row['macd_signal']):
            return 0
        elif ma_check and (df_row['ma_diff']<0):
            return 0
        else:
            return 1
    else:
        if news_check and (df_row['news']==1):
            return 0
        elif rsi_check and (df_row['rsi']>45):
            return 0
        elif macd_check and (df_row['macd'] < df_row['macd_signal']):
            return 0
        elif ma_check and (df_row['ma_diff']>0):
            return 0
        else:
            return -1

def enter_signal_extractor(df: pd.DataFrame, rsi_check, macd_check, ma_check, news_check, signal_config: dict):
    """
    Generates entry signals for each trading day based on London session.

    :param df: enriched market DataFrame with datetime index
    :param rsi_check: bool to enforce RSI rule
    :param macd_check: bool to enforce MACD rule
    :param ma_check: bool to enforce MA rule
    :param news_check: bool to enforce news rule
    :param signal_config: dict with 'london_start'/'london_end'
    :return: pd.DataFrame with new 'position' column
    """
    london_start = time(signal_config["london_start"])
    london_end = time(signal_config["london_start"])

    df['position'] = 0

    df['date'] = df.index.date

    result_frames = []

    for date, group in df.groupby('date'):
        day_df = group.copy()

        # Filter to london opening session
        opening_rows = day_df.between_time(london_start, london_end)

        signal = 0
        signal_index = None

        for idx, row in opening_rows.iterrows():
                signal = enter_signal_condition(row, news_check, rsi_check, macd_check, ma_check)
                if bool(signal):
                    signal_index = idx
                    break

        if signal_index:
            apply_mask = (day_df.index >= signal_index)
            day_df.loc[apply_mask, 'position'] = signal

        result_frames.append(day_df)

    final_df = pd.concat(result_frames)
    final_df.drop(columns='date', inplace=True)

    return final_df

def exit_signal_extractor(df: pd.DataFrame, signal_cond: dict) -> pd.DataFrame:
    """
    Generates exit signals and updates positions based on SL/TP and EOD.

    :param df: DataFrame with entry 'position' signals
    :param signal_cond: dict with 'TP_mode', 'SL_mode' and values
    :return: pd.DataFrame with updated 'position' reflecting closed trades
    """
    df = df.copy()
    df['time'] = df.index.time
    df['date'] = df.index.date

    result = []

    for date, day_data in df.groupby('date'):
        asian_high = day_data['Asian_high'].iloc[0]
        asian_low = day_data['Asian_low'].iloc[0]

        if signal_cond['TP_mode'] == 'Asian_range':
            tp_buy = 3 * asian_high - 2 * asian_low
            tp_sell = 3 * asian_low - 2 * asian_high
        elif isinstance(signal_cond['TP_mode'], int):
            tp_buy = asian_high + signal_cond['TP_mode']
            tp_sell = asian_low - signal_cond['TP_mode']
        else:
            tp_buy = 100
            tp_sell = -100

        if signal_cond['SL_mode'] == 'Asian_range':
            sl_buy = asian_low
            sl_sell = asian_high
        elif isinstance(signal_cond['SL_mode'], int):
            sl_buy = asian_high - signal_cond['SL_mode']
            sl_sell = asian_low + signal_cond['SL_mode']
        else:
            sl_buy = -100
            sl_sell = 100


        position = 0
        triggered = False

        for i, row in day_data.iterrows():
            close = row['close']
            current_pos = row['position']

            if not triggered:
                if position == 0 and current_pos != 0:
                    position = current_pos
                elif position == 1:
                    if close >= tp_buy or close <= sl_buy or row['time'] >= pd.to_datetime("23:15").time():
                        triggered = True
                        position = 0
                elif position == -1:
                    if close <= tp_sell or close >= sl_sell or row['time'] >= pd.to_datetime("23:15").time():
                        triggered = True
                        position = 0
            else:
                position = 0

            result.append(position)

    df['position'] = result
    df.drop(columns=['time', 'date'], inplace=True)
    return df
