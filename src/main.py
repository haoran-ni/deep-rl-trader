import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from rule_based_strategy import SMAStrategy
from utils import *

# import stock data
start_date = "2020-01-01"
end_date = "2025-01-01"
interval = "1d"
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "JPM", "V", "UNH"]
df = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=True)
df_baseline = yf.download("QQQ", start=start_date, end=end_date, interval=interval, auto_adjust=True)

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


# initialize strategy
TOTAL_ASSET = 1000000
BUYIN_UNIT = 100000
portfolio = Portfolio(tickers, TOTAL_ASSET)
trade_history = []

strategy = SMAStrategy(short_term_sma=df['SMA_50'], long_term_sma=df['SMA_200'])


# the sequence may be different from the initial ticker list, so we call it again
tickers = df['Close'].columns.to_list()
actions = np.zeros(len(tickers))

for tick in df.index:
    open_prices = df.loc[tick]["Open"]
    close_prices = df.loc[tick]["Close"]
    for i, ticker in enumerate(tickers):
        open_price = open_prices[ticker]
        close_price = close_prices[ticker]
        if actions[i] > 0: # e.g. action = 0.5, buy in with 0.5*BUYIN_UNIT
            # buy in at the open price of the next tick
            buyin_price = open_price
            buyin_money = BUYIN_UNIT * actions[i]
            shares = buyin_shares(buyin_money, buyin_price)
            portfolio.update_cash(-buyin_money)
            if portfolio.cash_enough:
                portfolio.update_trade_history(tick, 'BUY', ticker, shares, buyin_price)
                portfolio.update_holdings(ticker, shares, buyin_price)
            else:
                # action not executed, set last action to hold
                strategy.last_action[i] = strategy.hold
                
        elif actions[i] < 0: # e.g. action = -0.5, sell half of the shares
            # sell at the open price of the next tick
            sell_price = open_price
            # shares is a positive number
            shares = portfolio.holdings_info[ticker]["shares"]*(-actions[i])
            if shares > 0:
                revenue = sell_shares(shares, sell_price)  # revenue after fees
                portfolio.update_cash(revenue)
                portfolio.update_trade_history(tick, 'SELL', ticker, shares, sell_price)
                portfolio.update_holdings(ticker, -shares, sell_price)
            else:
                # action not executed, set last action to hold
                strategy.last_action[i] = strategy.hold

    # update portfolio data at the close price
    portfolio.step(tick, close_prices)
    
    actions = strategy.trade(tick)

print(portfolio.trade_history['NVDA'])

# plot stock price
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
portfolio_total = portfolio.total_value_history['Total Value']
baseline_total = df_baseline['Close', 'QQQ']
portfolio_total = portfolio_total / portfolio_total.iloc[0] * 100
baseline_total = baseline_total / baseline_total.iloc[0] * 100

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(portfolio.total_value_history.index, portfolio_total, label='Total Portfolio Value')
ax.plot(df.index, baseline_total, label='QQQ')
ax.legend()
ax.grid()
plt.savefig('../figures/portfolio_total_value.png')
