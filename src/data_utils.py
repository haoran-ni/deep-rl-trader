# data_utils.py
from __future__ import annotations
from typing import Tuple

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from wrappers import SeqObsWithRulesWrapper
from rule_based_strategy import SMAStrategy  # uses your existing strategy
from trade_env import TradeEnv               # uses your existing environment

def load_single_ticker_df(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download([ticker], start=start, end=end, interval=interval, auto_adjust=True).dropna()

    sma50 = df['Close'].apply(lambda s: ta.sma(s, 50))
    sma50.columns = pd.MultiIndex.from_product([['SMA_50'], sma50.columns])
    df = pd.concat([df, sma50], axis=1)

    sma200 = df['Close'].apply(lambda s: ta.sma(s, 200))
    sma200.columns = pd.MultiIndex.from_product([['SMA_200'], sma200.columns])
    df = pd.concat([df, sma200], axis=1)

    rsi14 = df['Close'].apply(lambda s: ta.rsi(s, 14))
    rsi14.columns = pd.MultiIndex.from_product([['RSI_14'], rsi14.columns])
    df = pd.concat([df, rsi14], axis=1)

    return df.dropna()

def slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()

def make_wrapped_env(
    df: pd.DataFrame,
    ticker: str,
    total_asset: float,
    buyin_portion: float,
    reward_look_ahead: int,
    T: int,
    warmup_steps: int,
):
    env = TradeEnv(
        data=df,
        tickers=[ticker],
        cash=total_asset,
        buyin_portion=buyin_portion,
        reward_look_ahead=reward_look_ahead,
    )

    short_df = df['SMA_50'][[ticker]]
    long_df  = df['SMA_200'][[ticker]]
    strategy = SMAStrategy(short_term_sma=short_df, long_term_sma=long_df)

    wrapped = SeqObsWithRulesWrapper(
        env=env,
        strategy=strategy,
        time_window=T,
        warmup_steps=warmup_steps,
    )
    return wrapped
