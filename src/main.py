import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from collections import defaultdict

from rule_based_strategy import SMAStrategy
from utils import *

# import stock data
tickers = ["NVDA", "MSFT"]
df = yf.download(tickers, start="2020-01-01", end="2025-01-01", interval="1d", auto_adjust=True)

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
total_asset = 1000000
buyin_money = 1000
portfolio = Portfolio(total_asset)
trade_history = []


strategy = SMAStrategy(short_term_sma=df['SMA_50'], long_term_sma=df['SMA_200'])

print(df.loc["2020-01-06"]['Open']["NVDA"])




exit()







action = 0
# the sequence may be different from the initial ticker list, so we call it again
tickers = df['Close'].columns.to_list()

for tick in df.index[190:210]:
    open_prices = df.loc[tick]["Open"]
    close_prices = df.loc[tick]["Close"]
    for i, ticker in enumerate(tickers):
        open_price = open_prices[ticker]
        close_price = close_prices[ticker]
        if actions[i] == 1:
            # buy in at the open price of the next tick
            buyin_price = open_price
            shares, cost = buyin_shares(buyin_money, buyin_price) # cost includes fees
            portfolio.update_cash(-cost)
            portfolio.update_holdings(ticker, shares, buyin_price)
            portfolio.update_trade_history(tick, 'BUY', ticker, shares, buyin_price)
        elif actions[i] == -1:
            # sell at the open price of the next tick
            sell_price = open_price
            shares = portfolio.holdings_info[ticker]["shares"]
            revenue = sell_shares(shares, sell_price)  # revenue after fees
            portfolio.update_cash(revenue)
            portfolio.update_holdings(ticker, -shares, sell_price)
            portfolio.update_trade_history(tick, 'SELL', ticker, shares, sell_price)

    # update portfolio at the close price of the current tick
    # TODO: update portfolio for all tickers
    portfolio.step(close_prices)
    
    actions = strategy.trade(tick)


# plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df.index, df['Close', 'NVDA'], label='Closing Price')
ax.plot(df.index, df['SMA_50', 'NVDA'], label='SMA 50')
ax.plot(df.index, df['SMA_200', 'NVDA'], label='SMA 200')
ax.plot(df.index, df['RSI_14', 'NVDA'], label='RSI 14')
ax.legend()
ax.grid()
# save fig
plt.savefig('../figures/nvda_closing_prices.png')
