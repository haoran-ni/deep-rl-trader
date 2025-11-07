import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt

from rule_based_strategy import SMAStrategy
from trade_env import *

current_directory = os.path.dirname(os.path.abspath(__file__))

# import stock data
start_date = "2014-01-01"
end_date = "2025-01-01" # will not be reached by yf.download
interval = "1d"
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "JPM", "V", "UNH"]
df = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=True)

# add new indicator columns
sma50 = df['Close'].apply(lambda s: ta.sma(s, length=50))
sma50.columns = pd.MultiIndex.from_product([['SMA_50'], sma50.columns])
df = pd.concat([df, sma50], axis=1)

sma200 = df['Close'].apply(lambda s: ta.sma(s, length=200))
sma200.columns = pd.MultiIndex.from_product([['SMA_200'], sma200.columns])
df = pd.concat([df, sma200], axis=1)

rsi14 = df['Close'].apply(lambda s: ta.rsi(s, length=14))
rsi14.columns = pd.MultiIndex.from_product([['RSI_14'], rsi14.columns])
df = pd.concat([df, rsi14], axis=1)

# drop NaN (not allowed by gymnasium)
df = df.dropna()
df_baseline = yf.download("QQQ", start=df.index[0], end=end_date, interval=interval, auto_adjust=True)


# initialize strategy
TOTAL_ASSET = 1e6
BUYIN_PORTION = 0.2 # e.g. every buy-in uses 20% of current cash

strategy = SMAStrategy(short_term_sma=df['SMA_50'], long_term_sma=df['SMA_200'])

env = TradeEnv(data=df,
               tickers=strategy.tickers,
               cash=TOTAL_ASSET,
               buyin_portion=BUYIN_PORTION,
               reward_look_ahead=20)

obs = env.reset()
for _ in range(10000):
    actions = strategy.trade(env.current_tick)
    obs, reward, terminated, truncated, info = env.step(actions)
    strategy.last_action = env.last_action
    if truncated:
        # env.reset()
        # strategy.reset()
        break


# plot stock price
if 'NVDA' in df['Close'].columns:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df.index, df['Close', 'NVDA'], label='Closing Price')
    ax.plot(df.index, df['SMA_50', 'NVDA'], label='SMA 50')
    ax.plot(df.index, df['SMA_200', 'NVDA'], label='SMA 200')
    ax.plot(df.index, df['RSI_14', 'NVDA'], label='RSI 14')
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
ax.legend()
ax.grid()
plt.savefig(os.path.join(current_directory, '../figures/portfolio_total_value.png'))
