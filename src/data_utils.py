# data_utils.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from rule_based_strategy import SMAStrategy, RsiMAStrategy
from trade_env import TradeEnv
from wrappers import SeqObsWithRulesWrapper


# ------------------------- caching & panel loading -------------------------

def _ticker_cache_path(cache_dir: str, interval: str, ticker: str) -> Path:
    return Path(cache_dir) / interval / f"{ticker}.parquet"


def _ensure_cached_one(ticker: str, start: str, end: str, interval: str, cache_dir: str) -> pd.DataFrame:
    """
    Cache per-ticker OHLCV to a parquet file. Auto-adjusted prices.
    Columns are MultiIndex, top-level: ['Open','High','Low','Close','Adj Close','Volume'].
    """
    p = _ticker_cache_path(cache_dir, interval, ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return pd.read_parquet(p)

    df = yf.download([ticker], start=start, end=end, interval=interval, auto_adjust=True)
    df = df.dropna()
    df.to_parquet(p)
    return df


def load_panel_cached(tickers: List[str], interval: str, cache_dir: str,
                      start: str, end: str) -> pd.DataFrame:
    """
    Load a MultiIndex-column panel for `tickers`, caching each ticker to disk if needed.
    Returns a single DataFrame whose columns are MultiIndex across all tickers.
    """
    frames = []
    for t in tickers:
        frames.append(_ensure_cached_one(t, start, end, interval, cache_dir))
    # Align on index (outer product of columns is fine since they’re disjoint per ticker)
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def load_qqq_cached(etf: str, interval: str, cache_dir: str,
                    start: str, end: str) -> pd.Series:
    """
    Cache and return the Close series for ETF benchmark (e.g., 'QQQ'), as a simple Series.
    """
    p = _ticker_cache_path(cache_dir, interval, f"{etf}_CLOSE")
    if p.exists():
        s = pd.read_parquet(p)
        # stored as DataFrame with a single column; convert to Series
        return s.iloc[:, 0]

    df = yf.download([etf], start=start, end=end, interval=interval, auto_adjust=True).dropna()
    close = df["Close"][etf].astype(float)
    p.parent.mkdir(parents=True, exist_ok=True)
    close.to_frame(name=etf).to_parquet(p)
    return close


# ------------------------- indicators -------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicators across all tickers.
    Input: MultiIndex columns (level=1 is ticker).
    Output: same index with added top-level indicator blocks; rows with NaNs dropped.
    """
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

    rsi14_ma7 = df['RSI_14'].apply(lambda s: ta.ema(s, length=7))  # (for RSI-based crossover rule)
    rsi14_ma7.columns = pd.MultiIndex.from_product([['RSI_14_MA_7'], rsi14_ma7.columns])
    df = pd.concat([df, rsi14_ma7], axis=1)

    return df.dropna().sort_index()


# ------------------------- splitting with history -------------------------

def slice_with_history(df: pd.DataFrame, start: str, end: str, T: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Return df slice including up to T-1 rows *before* the first row >= start,
    so our wrapper can build (T, D) from *true past*. Also return the exact start_at timestamp.

    This works for either a single-ticker view (columns filtered to one ticker)
    or a multi-ticker panel.
    """
    idx = df.index
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    i_start = idx.get_indexer([start_ts], method="backfill")[0]
    i_end   = idx.get_indexer([end_ts],   method="pad")[0]

    i_hist = max(0, i_start - (T - 1))
    df_ext = df.iloc[i_hist: i_end + 1].copy()
    start_at = idx[i_start]
    return df_ext, start_at


# ------------------------- rule strategies -------------------------

def build_strategies_single(df: pd.DataFrame, ticker: str, names: List[str]):
    """
    Build rule strategies JUST for one ticker (for training wrapper env).
    Recognized names: "sma", "rsi_ma".
    """
    strategies = []
    for name in (names or []):
        k = str(name).lower()
        if k == "sma":
            strategies.append(SMAStrategy(df["SMA_50"][[ticker]], df["SMA_200"][[ticker]]))
        elif k == "rsi_ma":
            strategies.append(RsiMAStrategy(df["RSI_14"][[ticker]], df["RSI_14_MA_7"][[ticker]]))
        else:
            raise ValueError(f"Unknown rule strategy: {name}")
    return strategies


def build_strategies_multi(df: pd.DataFrame, tickers: List[str], names: List[str]):
    """
    Build rule strategies for MULTIPLE tickers (for multi-ticker evaluation env).
    Each strategy receives a DataFrame with columns exactly in `tickers` order.
    """
    strategies = []
    for name in (names or []):
        k = str(name).lower()
        if k == "sma":
            strategies.append(SMAStrategy(df["SMA_50"][tickers], df["SMA_200"][tickers]))
        elif k == "rsi_ma":
            strategies.append(RsiMAStrategy(df["RSI_14"][tickers], df["RSI_14_MA_7"][tickers]))
        else:
            raise ValueError(f"Unknown rule strategy: {name}")
    return strategies


# ------------------------- env builders -------------------------

def make_wrapped_env(
    df: pd.DataFrame,
    ticker: str,
    total_asset: float,
    buyin_portion: float,
    reward_look_ahead: int,
    T: int,
    start_at: pd.Timestamp,
    rule_components: List[str] | None = None,
):
    """
    Build base TradeEnv (starting at start_at) and wrap with sequence+rules observation for ONE ticker.
    df must include at least T-1 rows prior to start_at to populate the rolling window.
    """
    env = TradeEnv(
        data=df[[c for c in df.columns if c[1] == ticker]],  # keep this ticker only
        tickers=[ticker],
        cash=total_asset,
        buyin_portion=buyin_portion,
        reward_look_ahead=reward_look_ahead,
        start_at=start_at,
        eval_mode=False,  # training mode
    )
    strategies = build_strategies_single(df, ticker, rule_components or [])
    return SeqObsWithRulesWrapper(env=env, strategies=strategies, time_window=T)


def make_multi_env(
    df: pd.DataFrame,
    tickers: List[str],
    total_asset: float,
    buyin_portion: float,
    reward_look_ahead: int,
    start_at: pd.Timestamp,
):
    """
    Multi-ticker `TradeEnv` (NO wrapper; used for evaluation).
    """
    env = TradeEnv(
        data=df[[c for c in df.columns if c[1] in tickers]],
        tickers=list(tickers),
        cash=total_asset,
        buyin_portion=buyin_portion,
        reward_look_ahead=reward_look_ahead,
        start_at=start_at,
        eval_mode=True,  # eval mode
    )
    return env
