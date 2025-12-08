# data_utils.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pandas_ta as ta
import yfinance as yf
from stable_baselines3.common.monitor import Monitor

from rule_based_strategy import SMAStrategy, RsiMAStrategy
from trade_env import TradeEnv
from wrappers import SeqObsWithRulesWrapper
from config import TZ


# ------------------------- local CSV loading -------------------------

def get_available_tickers(dataset_dir: str, interval: str = "1_hour") -> List[str]:
    """
    Get list of all available tickers in the local dataset directory.

    Args:
        dataset_dir: Path to the root dataset directory
        interval: One of "1_hour", "15_mins", or "1_min"

    Returns:
        Sorted list of unique ticker symbols available in the dataset
    """
    interval_dir = Path(dataset_dir) / interval
    if not interval_dir.exists():
        raise ValueError(f"Interval directory does not exist: {interval_dir}")

    # Find all CSV files and extract unique ticker names
    csv_files = list(interval_dir.glob("*.csv"))
    tickers = sorted(set([csv_file.stem.split('_')[0] for csv_file in csv_files]))

    return tickers


def load_local_csv_panel(dataset_dir: str, interval: str = "1_hour",
                         tickers: List[str] | None = None) -> pd.DataFrame:
    """
    Load pre-processed CSV files from local directory and combine into a MultiIndex panel.

    This is the DEFAULT and RECOMMENDED data loading method as it's faster and doesn't
    depend on network connectivity.

    Args:
        dataset_dir: Path to the root dataset directory
                    (e.g., "C:/path/to/dataset/nasdaq_100_2025_constituents_processed")
        interval: One of "1_hour", "15_mins", or "1_min"
        tickers: Optional list of ticker symbols to load. If None, loads all available tickers.

    Returns:
        MultiIndex DataFrame with columns like ('Open', 'AAPL'), ('Close', 'AAPL'), etc.
        Index is DatetimeIndex.

    Example:
        # Load all stocks from 1_hour dataset
        df = load_local_csv_panel("C:/Users/.../dataset/nasdaq_100_2025_constituents_processed",
                                   interval="1_hour")

        # Load specific tickers from 15_mins dataset
        df = load_local_csv_panel("C:/Users/.../dataset/nasdaq_100_2025_constituents_processed",
                                   interval="15_mins",
                                   tickers=["AAPL", "NVDA", "MSFT"])
    """
    interval_dir = Path(dataset_dir) / interval
    if not interval_dir.exists():
        raise ValueError(f"Interval directory does not exist: {interval_dir}")

    # Find all CSV files in the interval directory
    csv_files = list(interval_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {interval_dir}")

    # Extract ticker from filename (e.g., "AAPL_1_hour_2015-12-07_to_2025-12-01.csv" -> "AAPL")
    available_tickers = []
    ticker_to_file = {}
    for csv_file in csv_files:
        ticker = csv_file.stem.split('_')[0]  # First part before underscore is the ticker
        available_tickers.append(ticker)
        ticker_to_file[ticker] = str(csv_file)

    # Filter by requested tickers if provided
    if tickers is not None:
        tickers_to_load = [t for t in tickers if t in available_tickers]
        if not tickers_to_load:
            raise ValueError(f"None of the requested tickers {tickers} found in {interval_dir}")
        missing = set(tickers) - set(tickers_to_load)
        if missing:
            print(f"Warning: Tickers {missing} not found in dataset, skipping...")
    else:
        tickers_to_load = sorted(available_tickers)

    # Load each ticker's CSV and combine
    ticker_dfs = []
    for ticker in tickers_to_load:
        csv_path = ticker_to_file[ticker]
        # CSV format:
        # Row 0: Price,Close,High,Low,Open,Volume,VWAP,Transactions,Timestamp
        # Row 1: Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL
        # Row 2: Datetime,,,,,,,,
        # Row 3+: Data
        # We'll skip the first 2 rows and set column names from row 0

        # Read the CSV, skipping row 1 (Ticker row) and row 2 (Datetime row)
        # Row 0 is the header: Price,Close,High,Low,Open,Volume,VWAP,Transactions,Timestamp
        # Row 1 is: Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL
        # Row 2 is: Datetime,,,,,,,,
        # Row 3+: Data rows
        df_ticker = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0, parse_dates=True)

        # Rename the first column from 'Price' to something we can drop (it's actually Close)
        # The columns are now: Close, High, Low, Open, Volume, VWAP, Transactions, Timestamp
        # But the first column is labeled 'Price' due to the CSV format
        df_ticker.index.name = 'Datetime'

        # Drop the Timestamp column if it exists (we don't need it)
        if 'Timestamp' in df_ticker.columns:
            df_ticker = df_ticker.drop(columns=['Timestamp'])

        # Create MultiIndex columns: (feature, ticker)
        df_ticker.columns = pd.MultiIndex.from_product([df_ticker.columns, [ticker]])

        ticker_dfs.append(df_ticker)

    # Concatenate all tickers along columns
    panel = pd.concat(ticker_dfs, axis=1).sort_index()

    # Reorder columns to match yfinance format: (feature, ticker) for each feature
    # yfinance returns columns in order: Open, High, Low, Close, Volume
    # We need to pivot to have all Opens together, all Closes together, etc.
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'VWAP' in panel.columns.get_level_values(0):
        features.append('VWAP')
    if 'Transactions' in panel.columns.get_level_values(0):
        features.append('Transactions')

    # Reorder: for each feature, get all tickers
    reordered_cols = []
    for feature in features:
        if feature in panel.columns.get_level_values(0):
            for ticker in tickers_to_load:
                if (feature, ticker) in panel.columns:
                    reordered_cols.append((feature, ticker))

    panel = panel[reordered_cols]

    return panel


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

    NEW: By default, tries to load from local CSV files first (if USE_LOCAL_CSV is True in config).
    Falls back to yfinance download if local CSV is not available.

    Returns a single DataFrame whose columns are MultiIndex across all tickers.
    """
    from config import USE_LOCAL_CSV, LOCAL_CSV_DIR

    # Try loading from local CSV first if enabled
    if USE_LOCAL_CSV:
        try:
            print(f"[Data Loading] Attempting to load from local CSV: {LOCAL_CSV_DIR}, interval={interval}")
            panel = load_local_csv_panel(LOCAL_CSV_DIR, interval=interval, tickers=tickers)
            print(
                f"[Data Loading] Successfully loaded {len(panel.columns.get_level_values(1).unique())} tickers from local CSV")
            return panel
        except Exception as e:
            print(f"[Data Loading] Failed to load from local CSV ({e}), falling back to yfinance download...")

    # Fallback to yfinance download (original behavior)
    print(f"[Data Loading] Loading from yfinance: {len(tickers)} tickers, interval={interval}")
    frames = []
    for t in tickers:
        frames.append(_ensure_cached_one(t, start, end, interval, cache_dir))
    # Align on index (outer product of columns is fine since they're disjoint per ticker)
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def load_qqq_cached(etf: str, cache_dir: str, start: str, end: str) -> pd.Series:
    """
    Cache and return the Close series for ETF benchmark (e.g., 'QQQ'), as a simple Series.

    NEW: First tries to load from local dataset/etf directory (same format as stock data).
    Falls back to downloading from Yahoo Finance if local file is not available.

    NOTE: QQQ is always downloaded from Yahoo Finance using 1d interval (daily data),
    regardless of the configured interval. This is because QQQ is only used for
    evaluation plotting and comparison, not for training.
    """
    from config import INTERVAL

    # Try loading from local ETF directory first
    etf_dir = Path(__file__).parent.parent / "dataset" / "etf"
    if etf_dir.exists():
        # Look for ETF files matching the configured interval
        # Map interval format: "1_hour" -> "1_hour", "15_mins" -> "15_mins", "1_min" -> "1_min"
        interval_pattern = INTERVAL  # Use the configured interval

        # Find matching ETF file
        etf_files = list(etf_dir.glob(f"{etf}_{interval_pattern}_*.csv"))
        if etf_files:
            etf_file = etf_files[0]  # Take the first matching file
            print(f"[Data Loading] Loading {etf} from local file: {etf_file.name}")

            try:
                # Read the CSV file (same format as stock data)
                df = pd.read_csv(etf_file, header=[0, 1], index_col=0)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)

                # Extract Close column for the ETF
                # The CSV has MultiIndex columns: (feature, ticker)
                close = df[('Close', etf)].astype(float)
                print(f"[Data Loading] Successfully loaded {etf} from local file ({len(close)} rows)")
                return close
            except Exception as e:
                print(f"[Data Loading] Failed to load {etf} from local file ({e}), falling back to cache/download...")

    # Fallback: Check parquet cache
    p = _ticker_cache_path(cache_dir, "1d", f"{etf}_CLOSE")
    if p.exists():
        s = pd.read_parquet(p)
        # stored as DataFrame with a single column; convert to Series
        print(f"[Data Loading] Loaded {etf} from cache (1d interval)")
        return s.iloc[:, 0]

    # Download from yfinance
    print(f"[Data Loading] Downloading {etf} from Yahoo Finance (1d interval)")
    df = yf.download([etf], start=start, end=end, interval="1d", auto_adjust=True).dropna()
    close = df["Close"][etf].astype(float)
    p.parent.mkdir(parents=True, exist_ok=True)
    close.to_frame(name=etf).to_parquet(p)
    return close


# ------------------------- indicators -------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicators across all tickers.
    Input: MultiIndex columns (level=1 is ticker).
    Output: same index with added top-level indicator blocks.
    Note: NaN values are NOT dropped here to preserve data for all tickers.
          NaN rows should only be dropped when creating training env for a specific ticker.
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

    return df.sort_index()


# ------------------------- splitting with history -------------------------

def slice_with_history(df: pd.DataFrame, start: str, end: str, T: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Return df slice including up to T-1 rows *before* the first row >= start,
    so our wrapper can build (T, D) from *true past*. Also return the exact start_at timestamp.

    This works for either a single-ticker view (columns filtered to one ticker)
    or a multi-ticker panel.

    Args:
        df: DataFrame with DatetimeIndex (may be tz-aware or tz-naive)
        start: Start date string
        end: End date string
        T: Number of historical rows to include before start

    Returns:
        Tuple of (df_slice, start_at_timestamp)
    """
    idx = df.index
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Handle timezone consistency between index and timestamps
    start_ts = start_ts.tz_localize(TZ)
    end_ts = end_ts.tz_localize(TZ)

    # backfill: find first index >= start_ts (correct for start boundary)
    # pad: find last index <= end_ts (correct for end boundary)
    i_start = idx.get_indexer([start_ts], method="backfill")[0]
    i_end = idx.get_indexer([end_ts], method="pad")[0]

    # Include T-1 rows before start for history
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
        truncation_period: int | None = None,
):
    """
    Build base TradeEnv (starting at start_at) and wrap with sequence+rules observation for ONE ticker.

    Args:
        df: MultiIndex DataFrame with market data
        ticker: Ticker symbol to trade
        total_asset: Initial cash
        buyin_portion: Portion of cash to use per trade
        reward_look_ahead: Steps to look ahead for reward calculation
        T: Sequence length for observations
        start_at: Starting timestamp
        rule_components: List of rule strategy names to mix
        truncation_period: Number of days after which to truncate episodes (for training).
                          Set to None for no truncation.

    Returns:
        Wrapped TradeEnv for training
    """
    # Filter to this ticker only and drop NaN rows (only for single ticker training)
    ticker_data = df[[c for c in df.columns if c[1] == ticker]].dropna()

    env = TradeEnv(
        data=ticker_data,
        tickers=[ticker],
        cash=total_asset,
        buyin_portion=buyin_portion,
        truncation_period=truncation_period,  # Use configured truncation for training
        reward_look_ahead=reward_look_ahead,
        # start_at=start_at,
        start_at=None,
        eval_mode=False,  # training mode
    )
    strategies = build_strategies_single(df, ticker, rule_components or [])
    wrapped_env = SeqObsWithRulesWrapper(env=env, strategies=strategies, time_window=T)
    # Wrap with Monitor to track episode statistics for SB3 logging
    return Monitor(wrapped_env)


def make_multi_env(
        df: pd.DataFrame,
        tickers: List[str],
        total_asset: float,
        buyin_portion: float,
        reward_look_ahead: int,
        start_at: pd.Timestamp,
        truncation_period: int | None = None,
):
    """
    Multi-ticker `TradeEnv` (NO wrapper; used for evaluation).

    Args:
        df: MultiIndex DataFrame with market data
        tickers: List of ticker symbols to trade
        total_asset: Initial cash
        buyin_portion: Portion of cash to use per trade
        reward_look_ahead: Steps to look ahead for reward calculation
        start_at: Starting timestamp
        truncation_period: Number of days after which to truncate episodes.
                          For evaluation, this should always be None (no truncation).

    Returns:
        TradeEnv for evaluation (no wrapper)
    """
    env = TradeEnv(
        data=df[[c for c in df.columns if c[1] in tickers]],
        tickers=list(tickers),
        cash=total_asset,
        buyin_portion=buyin_portion,
        truncation_period=truncation_period,  # None for evaluation (full period)
        reward_look_ahead=reward_look_ahead,
        # start_at=start_at,
        start_at=None,
        eval_mode=True,  # eval mode
    )
    return env
