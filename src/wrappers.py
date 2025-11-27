# wrappers.py
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trade_env import TradeEnv


# ---------------------------------------------------------------------------
# Public helpers (shared by wrapper and evaluation code)
# ---------------------------------------------------------------------------

def reg_keys_for(obs_space: spaces.Dict, ticker: str, rule_prefix: str = "rule_action_") -> List[str]:
    """
    Return the list of *regular* feature keys (non-rule) for a given ticker,
    in the exact order used by the observation space.
    """
    inner = obs_space.spaces[ticker]
    assert isinstance(inner, spaces.Dict), "Ticker entry must be a Dict of features"
    keys = list(inner.spaces.keys())
    return [k for k in keys if not k.startswith(rule_prefix)]


def flatten_regular(
    raw: Dict[str, Dict[str, np.ndarray]],
    obs_space: spaces.Dict,
    ticker: str,
    reg_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Flatten the *regular* (non-rule) features of a given ticker into a 1D vector (D,).
    We keep a stable order that matches the observation space's inner Dict ordering.
    """
    if reg_keys is None:
        reg_keys = reg_keys_for(obs_space, ticker)
    parts: List[np.ndarray] = []
    for k in reg_keys:
        x = np.asarray(raw[ticker][k], dtype=np.float32).reshape(-1)
        parts.append(x)
    return np.concatenate(parts, axis=0).astype(np.float32)  # (D,)


class RollingSeq:
    """
    Tiny utility to build a right-aligned (T,D) rolling window.

    Usage:
      rs = RollingSeq(T)
      rs.fill_with(vec)   # first frame repeated T times
      rs.push(vec)        # push next frame
      seq = rs.array()    # (T,D) float32
    """
    def __init__(self, T: int):
        self.T = int(T)
        self._buf: deque[np.ndarray] = deque(maxlen=self.T)

    def clear(self) -> None:
        self._buf.clear()

    def fill_with(self, v: np.ndarray) -> None:
        self._buf.clear()
        for _ in range(self.T):
            self._buf.append(v.copy())

    def push(self, v: np.ndarray) -> None:
        if not self._buf:
            self.fill_with(v)
        else:
            self._buf.append(v.copy())

    def array(self) -> np.ndarray:
        # (T, D)
        return np.stack(self._buf, axis=0).astype(np.float32)


def compute_rule_stack(strategies: List, tick: pd.Timestamp, itick: int, portfolio, n_tickers: int) -> Optional[np.ndarray]:
    """
    Execute all strategies once to get their action vectors for *all* tickers.
    Returns:
      None (if no strategies) or a float32 array of shape (K, n_tickers)
    """
    if not strategies:
        return None
    rule_vecs = []
    for strat in strategies:
        vec = strat.trade(tick, itick, portfolio)  # shape: (n_tickers,)
        rule_vecs.append(vec.astype(np.float32))
    return np.stack(rule_vecs, axis=0)  # (K, n_tickers)


# ---------------------------------------------------------------------------
# Single-asset training wrapper (unchanged API; now uses RollingSeq internally)
# ---------------------------------------------------------------------------

class SeqObsWithRulesWrapper(gym.Wrapper):
    """
    Per-step observation to SB3 (single-asset training only):
      Always:
        "seq_regular": (T, D)  # rolling window of flattened regular features (right-aligned)
      If K>0 strategies provided:
        "rule_actions": (K,)    # rule vector at current tick (bypasses LSTM)

    IMPORTANT: No "warm-up" period that forces HOLD.
    We *pre-fill the buffer with real past* by slicing the DataFrame earlier
    (see data_utils.slice_with_history): the underlying TradeEnv is started at `start_at`,
    while our buffer already contains the previous T-1 frames.
    """

    def __init__(
        self,
        env: gym.Env,
        strategies: List,            # List[Strategy], can be empty to disable rule mixing
        time_window: int = 30,
        rule_low: float = -1.0,
        rule_high: float = 1.0,
    ):
        super().__init__(env)
        self.env: TradeEnv
        self.strategies = list(strategies) if strategies is not None else []
        self.T = int(time_window)

        # single training ticker
        assert isinstance(self.env.observation_space, spaces.Dict)
        self.ticker = list(self.env.observation_space.spaces.keys())[0]
        self.reg_keys = reg_keys_for(self.env.observation_space, self.ticker)

        # D = dim of flattened inner features
        self.D = sum(
            spaces.utils.flatdim(self.env.observation_space.spaces[self.ticker].spaces[k])
            for k in self.reg_keys
        )
        self.K = len(self.strategies)

        # Build observation space
        space_dict = {
            "seq_regular": spaces.Box(low=-np.inf, high=np.inf, shape=(self.T, self.D), dtype=np.float32),
        }
        if self.K > 0:
            space_dict["rule_actions"] = spaces.Box(low=rule_low, high=rule_high, shape=(self.K,), dtype=np.float32)
        self.observation_space = spaces.Dict(space_dict)

        # shared rolling buffer via the utility
        self._seq = RollingSeq(self.T)

    # ---------- internal helpers ----------

    def _compute_rule_vec(self) -> Optional[np.ndarray]:
        if self.K == 0:
            return None
        tick = self.env.current_tick
        itick = self.env.i_current_tick
        # strategies return vectors for all tickers (here single-asset ⇒ length 1)
        rule_stack = compute_rule_stack(self.strategies, tick, itick, self.env.portfolio, n_tickers=len(self.env.tickers))
        return rule_stack[:, 0].copy() if rule_stack is not None else None  # (K,)

    def _build_obs(self, raw: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        v = flatten_regular(raw, self.env.observation_space, self.ticker, self.reg_keys)  # (D,)
        if not self._seq._buf:
            self._seq.fill_with(v)
        else:
            self._seq.push(v)
        obs = {"seq_regular": self._seq.array()}
        if self.K > 0:
            rule = self._compute_rule_vec()
            obs["rule_actions"] = rule
        return obs

    # ---------- gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ret = self.env.reset() if options is None else self.env.reset(seed=seed, options=options)
        raw = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret
        self._seq.clear()
        return self._build_obs(raw), {}

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(action)
        # keep strategies' state consistent with env
        for strat in self.strategies:
            strat.last_action = self.env.last_action  # vector for all tickers
        obs = self._build_obs(raw)
        return obs, reward, terminated, truncated, info
