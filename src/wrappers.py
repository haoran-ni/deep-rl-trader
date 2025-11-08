# wrappers.py
from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------- helpers for single-asset nested Dict obs ----------

def _only_ticker(obs_space: spaces.Dict) -> str:
    keys = list(obs_space.spaces.keys())
    assert len(keys) == 1, f"Expected exactly 1 ticker, got {keys}"
    return keys[0]

def _reg_keys(obs_space: spaces.Dict, ticker: str, rule_prefix: str = "rule_action_") -> List[str]:
    inner = obs_space.spaces[ticker]
    assert isinstance(inner, spaces.Dict), "Ticker entry must be a Dict of features"
    keys = list(inner.spaces.keys())
    return [k for k in keys if not k.startswith(rule_prefix)]

def _flat_reg(obs: Dict[str, Dict[str, np.ndarray]], ticker: str, reg_keys: List[str]) -> np.ndarray:
    parts = []
    for k in reg_keys:
        x = np.asarray(obs[ticker][k], dtype=np.float32).reshape(-1)
        parts.append(x)
    return np.concatenate(parts, axis=0).astype(np.float32)  # (D,)

# ---------- wrapper: builds {"seq_regular": (T,D), "rule_actions": (K,)} ----------

class SeqObsWithRulesWrapper(gym.Wrapper):
    """
    Per-step observation to SB3:
      {
        "seq_regular": (T, D),  # rolling window of flattened regular features
        "rule_actions": (K,),   # computed by SMAStrategy at current tick (K=1 for SMA)
      }

    Warm-up: the first `warmup_steps` after reset, incoming action is forced to HOLD (0).
    """

    def __init__(
        self,
        env: gym.Env,
        strategy,                   # SMAStrategy
        time_window: int = 30,
        warmup_steps: Optional[int] = None,
        rule_low: float = -1.0,
        rule_high: float = 1.0,
    ):
        super().__init__(env)
        self.strategy = strategy
        self.T = int(time_window)
        self.warmup_steps = int(warmup_steps) if warmup_steps is not None else self.T - 1

        assert isinstance(self.env.observation_space, spaces.Dict)
        self.ticker = _only_ticker(self.env.observation_space)
        self.reg_keys = _reg_keys(self.env.observation_space, self.ticker)

        # D = total dim of flattened inner features for the one ticker
        self.D = sum(
            spaces.utils.flatdim(self.env.observation_space.spaces[self.ticker].spaces[k])
            for k in self.reg_keys
        )
        self.K = 1  # number of rule sources (currently 1 for SMA)

        self.observation_space = spaces.Dict({
            "seq_regular": spaces.Box(low=-np.inf, high=np.inf, shape=(self.T, self.D), dtype=np.float32),
            "rule_actions": spaces.Box(low=rule_low, high=rule_high, shape=(self.K,), dtype=np.float32),
        })

        self._buf: deque[np.ndarray] = deque(maxlen=self.T)
        self._since_reset = 0

    def _compute_rule_vec(self) -> np.ndarray:
        tick = self.env.current_tick
        actions_vec = self.strategy.trade(tick)  # vector for all tickers
        idx = self.strategy.ticker_to_i[self.ticker]
        a_scalar = float(actions_vec[idx])
        return np.array([a_scalar], dtype=np.float32)  # (K=1,)

    def _build_obs(self, raw: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        v = _flat_reg(raw, self.ticker, self.reg_keys)  # (D,)
        if not self._buf:
            for _ in range(self.T):
                self._buf.append(v.copy())
        else:
            self._buf.append(v.copy())
        seq = np.stack(self._buf, axis=0).astype(np.float32)    # (T, D)
        rule = self._compute_rule_vec()                         # (K,)
        return {"seq_regular": seq, "rule_actions": rule}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ret = self.env.reset() if options is None else self.env.reset(seed=seed, options=options)
        raw = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret
        self._buf.clear()
        self._since_reset = 0
        return self._build_obs(raw), {}  # always return (obs, info)

    def step(self, action):
        if self._since_reset < self.warmup_steps:
            action = np.zeros_like(action, dtype=np.float32)
        raw, reward, terminated, truncated, info = self.env.step(action)
        self._since_reset += 1

        # keep strategy state consistent with env
        self.strategy.last_action = self.env.last_action
        obs = self._build_obs(raw)
        return obs, reward, terminated, truncated, info
