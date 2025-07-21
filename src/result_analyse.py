from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import mplfinance as mpf
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true: list, y_pred: list, title: str)-> None:
    """
    Plots a confusion matrix comparing predicted vs actual trade outcomes.

    :param y_true: list or array of actual outcomes (-1 or 1)
    :param y_pred: list or array of predicted outcomes (-1 or 1)
    :param title: string title for the plot
    :return: None
    """
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['sell', 'buy'], yticklabels=['price down', 'price up'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"results/{title}.png", dpi=300)
    plt.close()

def plot_confusion_matrices_by_symbol(position_df: pd.DataFrame)-> None:
    """
    Generates and displays confusion matrices per symbol and overall.

    :param position_df: DataFrame containing 'symbol','direction','price_change'
    :return: None
    """
    df = position_df.copy()
    symbols = df['symbol'].unique()
    y_pred_tot = []
    y_true_tot = []

    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol]
        y_pred = [1 if dir == 'buy' else -1 for dir in df_symbol['direction']]
        y_true = [
            pred if price > 0 else -pred
            for pred, price in zip(y_pred, df_symbol['price_change'])
        ]
        y_pred_tot.extend(y_pred)
        y_true_tot.extend(y_true)
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,  # treating direction as a kind of "expected outcome"
            title=f"Confusion Matrix for {symbol}",
        )

    if len(symbols) > 1:
        plot_confusion_matrix(
            y_true=y_true_tot,
            y_pred=y_pred_tot,
            title="Total Confusion Matrix (All Symbols)",
        )

def candles_to_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts discrete trade entries/exits from a continuous 'position' signal series.

    :param df: DataFrame with 'position' column and datetime index
    :return: DataFrame listing each trade with entry/exit times and P&L
    """

    df = df.copy()
    df['datetime'] = df.index

    positions = []
    in_position = False
    start_idx = None
    current_pos = 0

    for i in range(len(df)):
        pos = df.iloc[i]['position']

        if not in_position and pos != 0:
            # Position opened
            in_position = True
            start_idx = i
            current_pos = pos

        elif in_position and (pos == 0 or pos != current_pos):
            # Position closed or reversed
            start_row = df.iloc[start_idx]
            end_row = df.iloc[i]

            entry_price = start_row['close']
            exit_price = end_row['close']
            start_time = start_row['datetime']
            end_time = end_row['datetime']

            # Price change logic based on position direction
            if current_pos == 1:  # Buy
                price_change = exit_price - entry_price
                direction = "buy"
            elif current_pos == -1:  # Sell
                price_change = entry_price - exit_price
                direction = "sell"
            else:
                continue  # Shouldn't happen, safety check

            positions.append({
                "symbol": start_row['symbol'],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "price_change": price_change,
                "start_time": start_time,
                "end_time": end_time
            })

            # Prepare for the next position
            in_position = pos != 0
            start_idx = i if in_position else None
            current_pos = pos

    return pd.DataFrame(positions)

def positions_analyse(position_df: pd.DataFrame, exchange_rates: dict) -> None:
    """
    Analyzes trading positions and writes a detailed Excel report.

    :param position_df: DataFrame of individual trades
    :param exchange_rates: dict mapping symbol to USD conversion rate
    :return: dict of summary statistics (PnL counts and totals)
    """
    file_path = Path(__file__).resolve().parent.parent / "results" / "report.xlsx"
    df = position_df.copy()
    df['start_time'] = df['start_time'].apply(lambda dt: dt.replace(tzinfo=None))
    df['end_time'] = df['end_time'].apply(lambda dt: dt.replace(tzinfo=None))


    df['pnl_usd'] = df.apply(lambda row: row['price_change'] * float(exchange_rates.get(row['symbol'], 1)), axis=1)

    df['result'] = df['pnl_usd'].apply(lambda x: 'profit' if x > 0 else 'loss')

    buy_df = df[df['direction'] == 'buy']
    sell_df = df[df['direction'] == 'sell']

    def count_group(df, direction=None, result=None):
        filtered = df
        if direction:
            filtered = filtered[filtered['direction'] == direction]
        if result:
            filtered = filtered[filtered['result'] == result]
        return filtered.groupby('symbol').size().to_frame(name='count')

    def sum_group(df, result=None):
        filtered = df
        if result:
            filtered = filtered[filtered['result'] == result]
        return filtered.groupby('symbol')['pnl_usd'].sum().to_frame()


    summary = {
        "total_positions": len(df),
        "total_profit_positions": (df['result'] == 'profit').sum(),
        "total_loss_positions": (df['result'] == 'loss').sum(),

        "total_buy_positions": len(buy_df),
        "buy_profit_positions": (buy_df['result'] == 'profit').sum(),
        "buy_loss_positions": (buy_df['result'] == 'loss').sum(),

        "total_sell_positions": len(sell_df),
        "sell_profit_positions": (sell_df['result'] == 'profit').sum(),
        "sell_loss_positions": (sell_df['result'] == 'loss').sum(),

        "total_pnl_usd": df['pnl_usd'].sum(),
        "total_loss_usd": df[df['result'] == 'loss']['pnl_usd'].sum(),
        "total_profit_usd": df[df['result'] == 'profit']['pnl_usd'].sum()
    }

    # Write to Excel
    with pd.ExcelWriter(file_path) as writer:
        df.to_excel(writer, sheet_name='Positions', index=False)

        pd.DataFrame([summary]).T.rename(columns={0: 'value'}).to_excel(writer, sheet_name='Summary')

        count_group(df, direction='buy').to_excel(writer, sheet_name='Buy Counts')
        count_group(df, direction='sell').to_excel(writer, sheet_name='Sell Counts')
        count_group(df, direction='buy', result='profit').to_excel(writer, sheet_name='Buy Profits')
        count_group(df, direction='sell', result='profit').to_excel(writer, sheet_name='Sell Profits')
        count_group(df, direction='buy', result='loss').to_excel(writer, sheet_name='Buy Losses')
        count_group(df, direction='sell', result='loss').to_excel(writer, sheet_name='Sell Losses')

        sum_group(df, result='profit').to_excel(writer, sheet_name='Symbol Profit USD')
        sum_group(df, result='loss').to_excel(writer, sheet_name='Symbol Loss USD')
        df.groupby('symbol')['pnl_usd'].sum().to_frame().to_excel(writer, sheet_name='Symbol Total PnL')

def positions_plotter(positions_df: pd.DataFrame, exchange_rates: dict) -> None:
    """
    Plots PnL distributions, cumulative performance, and confusion matrices.

    :param positions_df: DataFrame of trades
    :param exchange_rates: dict mapping symbol to USD rate
    :return: None
    """

    df = positions_df.copy()

    # Convert PnL to USD
    df['pnl_usd'] = df.apply(
        lambda row: row['price_change'] * exchange_rates.get(row['symbol'], 1),
        axis=1
    )
    df['date'] = pd.to_datetime(df['end_time']).dt.date

    symbols = df['symbol'].unique()
    results = {}

    # Distribution plots per symbol
    def plot_distribution(symbol_df, title_suffix=''):
        pnl_values = symbol_df['pnl_usd'].values
        mu, std = norm.fit(pnl_values)

        plt.figure()
        plt.hist(pnl_values, bins=30, density=True, alpha=0.6)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f"PnL Histogram {title_suffix} (μ={mu:.6f}, σ={std:.6f})")
        plt.xlabel("PnL (USD)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/PnL Histogram {title_suffix}.png", dpi=300)
        plt.close()

        return {"mean": mu, "std": std, "count": len(pnl_values)}

    # Plot cumulative PnL logic
    if len(symbols) == 1:
        symbol = symbols[0]
        symbol_df = df[df['symbol'] == symbol]
        # Single symbol PnL over time
        daily_pnl = symbol_df.groupby('date')['pnl_usd'].sum()
        full_range = pd.date_range(symbol_df['date'].min(), symbol_df['date'].max(), freq='D')
        daily_pnl = daily_pnl.reindex(full_range, fill_value=0)
        cumulative_pnl = daily_pnl.cumsum()

        plt.figure()
        plt.plot(cumulative_pnl.index, cumulative_pnl.values, label=f"{symbol}")
        plt.title(f"Cumulative PnL Over Time ({symbol})")
        plt.xlabel("Date")
        plt.ylabel("PnL (USD)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"results/Cumulative PnL Over Time ({symbol}).png", dpi=300)
        plt.close()

        results[symbol] = plot_distribution(symbol_df, title_suffix=f"({symbol})")

    else:
        plt.figure()
        all_daily_pnls = []

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol]
            daily_pnl = symbol_df.groupby('date')['pnl_usd'].sum()
            full_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
            daily_pnl = daily_pnl.reindex(full_range, fill_value=0)
            cumulative_pnl = daily_pnl.cumsum()
            all_daily_pnls.append(daily_pnl)

            plt.plot(cumulative_pnl.index, cumulative_pnl.values, label=f"{symbol}")

        # Total PnL
        total_daily_pnl = sum(all_daily_pnls)
        total_cumulative = total_daily_pnl.cumsum()
        plt.plot(total_cumulative.index, total_cumulative.values, label="TOTAL", linewidth=1, color='black')

        plt.title("Cumulative PnL Over Time (All Symbols)")
        plt.xlabel("Date")
        plt.ylabel("PnL (USD)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/Cumulative PnL Over Time (All Symbols)", dpi=300)
        plt.close()

        for symbol in symbols:
            results[symbol] = plot_distribution(symbol_df, title_suffix=f"({symbol})")

        results['TOTAL'] = plot_distribution(df, title_suffix="(TOTAL)")

    plot_confusion_matrices_by_symbol(positions_df)

    return

def buy_true_sample_plotter(df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    """
    Randomly selects and plots a profitable buy trade with indicators.

    :param df: enriched market DataFrame
    :param positions_df: DataFrame of trades
    :return: None
    """

    profitable_buys = positions_df[
        (positions_df['direction'] == 'buy') &
        (positions_df['price_change'] > 0)
    ]

    trade = profitable_buys.sample(1).iloc[0]
    symbol = trade['symbol']
    entry_time = pd.to_datetime(trade['start_time'])
    exit_time = pd.to_datetime(trade['end_time'])
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']

    df['datetime'] = pd.to_datetime(df['datetime'])

    trade_day = entry_time.normalize()
    next_day = trade_day + pd.Timedelta(days=1)

    # Filter df based on time column instead of index
    df_day = df.loc[
        (df['symbol'] == symbol) &
        (df['datetime'] >= trade_day) &
        (df['datetime'] < next_day)
        ].copy()

    df_day.set_index('datetime', inplace=True)  # Needed for plotting with mplfinance

    df_plot = df_day[['open', 'high', 'low', 'close']].copy()

    rsi = df_day.get('rsi', pd.Series(np.nan, index=df_day.index))
    macd = df_day.get('macd', pd.Series(np.nan, index=df_day.index))
    macd_signal = df_day.get('macd_signal', pd.Series(np.nan, index=df_day.index))
    macd_hist = df_day.get('macd_diff', macd - macd_signal)

    entry_marker = pd.Series(np.nan, index=df_day.index)
    exit_marker = pd.Series(np.nan, index=df_day.index)

    if entry_time in entry_marker.index:
        entry_marker.loc[entry_time] = entry_price
    if exit_time in exit_marker.index:
        exit_marker.loc[exit_time] = exit_price

    apds = [mpf.make_addplot([df_day['Asian_high'].iloc[0]] * len(df_day), color='lime', linestyle='--'),
            mpf.make_addplot([df_day['Asian_low'].iloc[0]] * len(df_day), color='orange', linestyle='--')]

    apds.extend([
        mpf.make_addplot(entry_marker, type='scatter', marker='^', markersize=200, color='lime'),
        mpf.make_addplot(exit_marker, type='scatter', marker='v', markersize=200, color='red'),
        mpf.make_addplot(rsi, panel=1, color='green', ylabel='RSI'),
        mpf.make_addplot(macd, panel=2, color='yellow', ylabel='MACD'),
        mpf.make_addplot(macd_signal, panel=2, color='orange'),
        mpf.make_addplot(macd_hist, panel=2, type='bar', color='gray', alpha=0.5),
    ])

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        ohlc='i'
    )

    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridcolor='gray',
        gridaxis='both',
        facecolor='#1a1a1a',
        y_on_right=False
    )

    figure_title = (
        f"{symbol} Trade on {trade_day.date()} | "
        f"Entry: {entry_time.strftime('%H:%M')} @ {entry_price:.5f} | "
        f"Exit: {exit_time.strftime('%H:%M')} @ {exit_price:.5f}"
    )

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        addplot=apds,
        volume=False,
        figsize=(14, 8),
        panel_ratios=(5, 1.2, 1.2),
        tight_layout=True,
        datetime_format='%H:%M',
        xrotation=0,
        returnfig=True,
        ylabel='Price'
    )

    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.savefig("results/profitable_buy_sample.png", dpi=300)
    plt.close(fig)

def buy_false_sample_plotter(df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    """
    Randomly selects and plots a profitable sell trade with indicators.

    :param df: enriched market DataFrame
    :param positions_df: DataFrame of trades
    :return: None
    """
    nprofitable_buys = positions_df[
        (positions_df['direction'] == 'buy') &
        (positions_df['price_change'] < 0)
        ]

    trade = nprofitable_buys.sample(1).iloc[0]
    symbol = trade['symbol']
    entry_time = pd.to_datetime(trade['start_time'])
    exit_time = pd.to_datetime(trade['end_time'])
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']

    df['datetime'] = pd.to_datetime(df['datetime'])

    trade_day = entry_time.normalize()
    next_day = trade_day + pd.Timedelta(days=1)

    # Filter df based on time column instead of index
    df_day = df.loc[
        (df['symbol'] == symbol) &
        (df['datetime'] >= trade_day) &
        (df['datetime'] < next_day)
        ].copy()

    df_day.set_index('datetime', inplace=True)  # Needed for plotting with mplfinance

    df_plot = df_day[['open', 'high', 'low', 'close']].copy()

    rsi = df_day.get('rsi', pd.Series(np.nan, index=df_day.index))
    macd = df_day.get('macd', pd.Series(np.nan, index=df_day.index))
    macd_signal = df_day.get('macd_signal', pd.Series(np.nan, index=df_day.index))
    macd_hist = df_day.get('macd_diff', macd - macd_signal)

    entry_marker = pd.Series(np.nan, index=df_day.index)
    exit_marker = pd.Series(np.nan, index=df_day.index)

    if entry_time in entry_marker.index:
        entry_marker.loc[entry_time] = entry_price
    if exit_time in exit_marker.index:
        exit_marker.loc[exit_time] = exit_price

    apds = [mpf.make_addplot([df_day['Asian_high'].iloc[0]] * len(df_day), color='lime', linestyle='--'),
            mpf.make_addplot([df_day['Asian_low'].iloc[0]] * len(df_day), color='orange', linestyle='--')]

    apds.extend([
        mpf.make_addplot(entry_marker, type='scatter', marker='^', markersize=200, color='lime'),
        mpf.make_addplot(exit_marker, type='scatter', marker='v', markersize=200, color='red'),
        mpf.make_addplot(rsi, panel=1, color='green', ylabel='RSI'),
        mpf.make_addplot(macd, panel=2, color='yellow', ylabel='MACD'),
        mpf.make_addplot(macd_signal, panel=2, color='orange'),
        mpf.make_addplot(macd_hist, panel=2, type='bar', color='gray', alpha=0.5),
    ])

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        ohlc='i'
    )

    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridcolor='gray',
        gridaxis='both',
        facecolor='#1a1a1a',
        y_on_right=False
    )

    figure_title = (
        f"{symbol} Trade on {trade_day.date()} | "
        f"Entry: {entry_time.strftime('%H:%M')} @ {entry_price:.5f} | "
        f"Exit: {exit_time.strftime('%H:%M')} @ {exit_price:.5f}"
    )

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        addplot=apds,
        volume=False,
        figsize=(14, 8),
        panel_ratios=(5, 1.2, 1.2),
        tight_layout=True,
        datetime_format='%H:%M',
        xrotation=0,
        returnfig=True,
        ylabel='Price'
    )

    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.savefig("results/Unnprofitable_buy_sample.png", dpi=300)
    plt.close(fig)

def sell_true_sample_plotter(df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    """
    Randomly selects and plots an unprofitable sell trade with indicators.

    :param df: enriched market DataFrame
    :param positions_df: DataFrame of trades
    :return: None
    """

    profitable_sells = positions_df[
        (positions_df['direction'] == 'sell') &
        (positions_df['price_change'] > 0)
        ]

    trade = profitable_sells.sample(1).iloc[0]
    symbol = trade['symbol']
    entry_time = pd.to_datetime(trade['start_time'])
    exit_time = pd.to_datetime(trade['end_time'])
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']

    df['datetime'] = pd.to_datetime(df['datetime'])

    trade_day = entry_time.normalize()
    next_day = trade_day + pd.Timedelta(days=1)

    # Filter df based on time column instead of index
    df_day = df.loc[
        (df['symbol'] == symbol) &
        (df['datetime'] >= trade_day) &
        (df['datetime'] < next_day)
        ].copy()

    df_day.set_index('datetime', inplace=True)  # Needed for plotting with mplfinance

    df_plot = df_day[['open', 'high', 'low', 'close']].copy()

    rsi = df_day.get('rsi', pd.Series(np.nan, index=df_day.index))
    macd = df_day.get('macd', pd.Series(np.nan, index=df_day.index))
    macd_signal = df_day.get('macd_signal', pd.Series(np.nan, index=df_day.index))
    macd_hist = df_day.get('macd_diff', macd - macd_signal)

    entry_marker = pd.Series(np.nan, index=df_day.index)
    exit_marker = pd.Series(np.nan, index=df_day.index)

    if entry_time in entry_marker.index:
        entry_marker.loc[entry_time] = entry_price
    if exit_time in exit_marker.index:
        exit_marker.loc[exit_time] = exit_price

    apds = [mpf.make_addplot([df_day['Asian_high'].iloc[0]] * len(df_day), color='lime', linestyle='--'),
            mpf.make_addplot([df_day['Asian_low'].iloc[0]] * len(df_day), color='orange', linestyle='--')]

    apds.extend([
        mpf.make_addplot(entry_marker, type='scatter', marker='^', markersize=200, color='lime'),
        mpf.make_addplot(exit_marker, type='scatter', marker='v', markersize=200, color='red'),
        mpf.make_addplot(rsi, panel=1, color='green', ylabel='RSI'),
        mpf.make_addplot(macd, panel=2, color='yellow', ylabel='MACD'),
        mpf.make_addplot(macd_signal, panel=2, color='orange'),
        mpf.make_addplot(macd_hist, panel=2, type='bar', color='gray', alpha=0.5),
    ])

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        ohlc='i'
    )

    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridcolor='gray',
        gridaxis='both',
        facecolor='#1a1a1a',
        y_on_right=False
    )

    figure_title = (
        f"{symbol} Trade on {trade_day.date()} | "
        f"Entry: {entry_time.strftime('%H:%M')} @ {entry_price:.5f} | "
        f"Exit: {exit_time.strftime('%H:%M')} @ {exit_price:.5f}"
    )

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        addplot=apds,
        volume=False,
        figsize=(14, 8),
        panel_ratios=(5, 1.2, 1.2),
        tight_layout=True,
        datetime_format='%H:%M',
        xrotation=0,
        returnfig=True,
        ylabel='Price'
    )

    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.savefig("results/profitable_sell_sample.png", dpi=300)
    plt.close(fig)

def sell_false_sample_plotter(df: pd.DataFrame, positions_df: pd.DataFrame) -> None:
    nprofitable_sells = positions_df[
        (positions_df['direction'] == 'sell') &
        (positions_df['price_change'] < 0)
        ]

    trade = nprofitable_sells.sample(1).iloc[0]
    symbol = trade['symbol']
    entry_time = pd.to_datetime(trade['start_time'])
    exit_time = pd.to_datetime(trade['end_time'])
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']

    df['datetime'] = pd.to_datetime(df['datetime'])

    trade_day = entry_time.normalize()
    next_day = trade_day + pd.Timedelta(days=1)

    # Filter df based on time column instead of index
    df_day = df.loc[
        (df['symbol'] == symbol) &
        (df['datetime'] >= trade_day) &
        (df['datetime'] < next_day)
        ].copy()

    df_day.set_index('datetime', inplace=True)  # Needed for plotting with mplfinance

    df_plot = df_day[['open', 'high', 'low', 'close']].copy()

    rsi = df_day.get('rsi', pd.Series(np.nan, index=df_day.index))
    macd = df_day.get('macd', pd.Series(np.nan, index=df_day.index))
    macd_signal = df_day.get('macd_signal', pd.Series(np.nan, index=df_day.index))
    macd_hist = df_day.get('macd_diff', macd - macd_signal)

    entry_marker = pd.Series(np.nan, index=df_day.index)
    exit_marker = pd.Series(np.nan, index=df_day.index)

    if entry_time in entry_marker.index:
        entry_marker.loc[entry_time] = entry_price
    if exit_time in exit_marker.index:
        exit_marker.loc[exit_time] = exit_price

    apds = [mpf.make_addplot([df_day['Asian_high'].iloc[0]] * len(df_day), color='lime', linestyle='--'),
            mpf.make_addplot([df_day['Asian_low'].iloc[0]] * len(df_day), color='orange', linestyle='--')]

    apds.extend([
        mpf.make_addplot(entry_marker, type='scatter', marker='^', markersize=200, color='lime'),
        mpf.make_addplot(exit_marker, type='scatter', marker='v', markersize=200, color='red'),
        mpf.make_addplot(rsi, panel=1, color='green', ylabel='RSI'),
        mpf.make_addplot(macd, panel=2, color='yellow', ylabel='MACD'),
        mpf.make_addplot(macd_signal, panel=2, color='orange'),
        mpf.make_addplot(macd_hist, panel=2, type='bar', color='gray', alpha=0.5),
    ])

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        ohlc='i'
    )

    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridcolor='gray',
        gridaxis='both',
        facecolor='#1a1a1a',
        y_on_right=False
    )

    figure_title = (
        f"{symbol} Trade on {trade_day.date()} | "
        f"Entry: {entry_time.strftime('%H:%M')} @ {entry_price:.5f} | "
        f"Exit: {exit_time.strftime('%H:%M')} @ {exit_price:.5f}"
    )

    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        addplot=apds,
        volume=False,
        figsize=(14, 8),
        panel_ratios=(5, 1.2, 1.2),
        tight_layout=True,
        datetime_format='%H:%M',
        xrotation=0,
        returnfig=True,
        ylabel='Price'
    )

    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.suptitle(figure_title, fontsize=13, color='white', weight='bold', y=0.96)
    fig.savefig("results/Unprofitable_sell_sample.png", dpi=300)
    plt.close(fig)

def plot_symbol_pnl(stats: dict)-> None:
    """
    Plots total PnL per symbol as a bar chart.

    :param stats: dict containing 'symbol_total_pnl_usd' mapping symbols to PnL
    :return: None
    """
    pnl_data = stats.get('symbol_total_pnl_usd', {})
    if not pnl_data:
        print("No PnL data to plot.")
        return

    symbols = list(pnl_data.keys())
    values = list(pnl_data.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(symbols, values)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("PnL per Symbol (USD)")
    plt.ylabel("Total PnL")
    plt.xlabel("Symbol")
    plt.xticks(rotation=45)

    # Label bars with values
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("results/PnL per Symbol (USD).png", dpi=300)
    plt.close()

