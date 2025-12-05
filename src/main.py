import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from gymnasium import spaces

from rule_based_strategy import *
from trade_env import *
from tickers import *

from config import *

current_directory = os.path.dirname(os.path.abspath(__file__))


# ------------------------ import stock data ------------------------
start_date = '2024-01-01'
end_date = '2025-11-01'
interval = "1h"
# tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "JPM", "V", "UNH"]
tickers = QQQ_TICKERS
df = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=True)

# Or load data from local CSV
# data_path = "/Volumes/Elements/stock_market_data/nasdaq_100_2025_constituents_processed/1_hour/AAPL_1_hour_2015-12-07_to_2025-12-01.csv"
# df = pd.read_csv(data_path, index_col=0, header=[0,1], parse_dates=True)


# ------------------------ add new indicator columns ------------------------
# Volume
RVOL14 = df["Volume"] / df["Volume"].rolling(14).mean()
RVOL14.columns = pd.MultiIndex.from_product([['RVOL14'], RVOL14.columns])
df = pd.concat([df, RVOL14], axis=1)

RVOL50 = df["Volume"] / df["Volume"].rolling(50).mean()
RVOL50.columns = pd.MultiIndex.from_product([['RVOL50'], RVOL50.columns])
df = pd.concat([df, RVOL50], axis=1)

RVOL200 = df["Volume"] / df["Volume"].rolling(200).mean()
RVOL200.columns = pd.MultiIndex.from_product([['RVOL200'], RVOL200.columns])
df = pd.concat([df, RVOL200], axis=1)

# SMA
sma14 = df['Close'].apply(lambda s: ta.sma(s, length=14))
sma14.columns = pd.MultiIndex.from_product([['SMA_14'], sma14.columns])
df = pd.concat([df, sma14], axis=1)

sma50 = df['Close'].apply(lambda s: ta.sma(s, length=50))
sma50.columns = pd.MultiIndex.from_product([['SMA_50'], sma50.columns])
df = pd.concat([df, sma50], axis=1)

sma200 = df['Close'].apply(lambda s: ta.sma(s, length=200))
sma200.columns = pd.MultiIndex.from_product([['SMA_200'], sma200.columns])
df = pd.concat([df, sma200], axis=1)

# RSI
rsi14 = df['Close'].apply(lambda s: ta.rsi(s, length=14))
rsi14.columns = pd.MultiIndex.from_product([['RSI_14'], rsi14.columns])
df = pd.concat([df, rsi14], axis=1)

rsi50 = df['Close'].apply(lambda s: ta.rsi(s, length=50))
rsi50.columns = pd.MultiIndex.from_product([['RSI_50'], rsi50.columns])
df = pd.concat([df, rsi50], axis=1)

rsi200 = df['Close'].apply(lambda s: ta.rsi(s, length=200))
rsi200.columns = pd.MultiIndex.from_product([['RSI_200'], rsi200.columns])
df = pd.concat([df, rsi200], axis=1)

rsi14_ma7 = df['RSI_14'].apply(lambda s: ta.ema(s, length=7))
rsi14_ma7.columns = pd.MultiIndex.from_product([['RSI_14_MA_7'], rsi14_ma7.columns])
df = pd.concat([df, rsi14_ma7], axis=1)

# drop NaN (not allowed by gymnasium)
df = df.dropna()
df_baseline = yf.download("QQQ", start=df.index[0], end=end_date, interval=interval, auto_adjust=True)


# ------------------------ run trading simulation ------------------------
TOTAL_ASSET = 1e6
BUYIN_PORTION = 1 # e.g. every buy-in uses $BUYIN_PORTION of current cash

#strategy = SMAStrategy(short_term_sma=df['SMA_50'], long_term_sma=df['SMA_200'])
strategy = RsiMAStrategy(rsi=df['RSI_14'], rsi_ma=df['RSI_14_MA_7'])

env = TradeEnv(data=df,
               tickers=strategy.tickers,
               cash=TOTAL_ASSET,
               buyin_portion=BUYIN_PORTION,
               truncation_period=None,
               reward_look_ahead=20,
               eval_mode=True)

obs, _ = env.reset()
for _ in range(10000):
    actions = strategy.trade(env.current_tick, env.i_current_tick, env.portfolio)
    obs, reward, terminated, truncated, info = env.step(actions)
    strategy.last_action = env.last_action
    if truncated:
        # obs, _ = env.reset()
        # strategy.reset()
        break


# ------------------------ plot results ------------------------
if 'NVDA' in df['Close'].columns:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df.index, df['Close', 'NVDA'], label='Closing Price')
    ax.plot(df.index, df['SMA_50', 'NVDA'], label='SMA 50')
    ax.plot(df.index, df['SMA_200', 'NVDA'], label='SMA 200')
    ax.legend()
    ax.grid()
    plt.savefig('../figures/nvda_closing_prices.png')

# plot total portfolio value over time, compared with QQQ
# normalize total values as percentages
portfolio_total = env.portfolio.total_value_history['Total Value']
baseline_total = df_baseline['Close', 'QQQ']
portfolio_total = portfolio_total / portfolio_total.iloc[0] * 100
baseline_total = baseline_total / baseline_total.iloc[0] * 100

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(env.portfolio.total_value_history.index, portfolio_total, label='Total Portfolio Value')
ax.plot(df.index, baseline_total, label='QQQ')
ax.set_title('RSI MA Strategy')
ax.legend()
ax.grid()
plt.savefig(os.path.join(current_directory, '../figures/portfolio_total_value.png'))
