# eval_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import (
    slice_with_history, make_multi_env, build_strategies_multi
)
from wrappers import (
    RollingSeq, reg_keys_for, flatten_regular, compute_rule_stack
)

# ------------------------- small helpers -------------------------

def year_ranges(start: str, end: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    years = []
    y = s.year
    while pd.Timestamp(year=y, month=1, day=1) <= e:
        y_start = max(pd.Timestamp(year=y, month=1, day=1), s)
        y_end = min(pd.Timestamp(year=y, month=12, day=31), e)
        if y_start <= y_end:
            years.append((y_start, y_end))
        y += 1
    return years


def plot_total_value_curves(curves: Dict[str, Tuple[List, List]] , title: str, out_path: str) -> str:
    """
    curves: name -> (dates, values) ; plots only total portfolio value.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5), dpi=150)
    df = pd.DataFrame({
        name: pd.Series(vals, index=pd.to_datetime(dates))
        for name, (dates, vals) in curves.items()
    }).sort_index().ffill()
    ax = plt.gca()
    df.plot(ax=ax, linewidth=1.6)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Total Portfolio Value ($)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------- RL rollout on multi-ticker env (now reusing wrapper helpers) -------------------------

def rollout_multi_total_with_model(
    model,
    env,
    df_ext: pd.DataFrame,
    tickers: List[str],
    T: int,
    strategies,  # list of strategies to compute rule components
) -> Tuple[List, List]:
    """
    Drive a multi-ticker TradeEnv using a SINGLE-ASSET policy by generating one action per ticker.

    DRY: reuses RollingSeq, reg_keys_for, flatten_regular and compute_rule_stack
    to avoid maintaining any ad-hoc buffers here.

    Returns (dates, total_values)
    """
    # Init per-ticker rolling builders + RNN states
    obs_space = env.observation_space
    reg_keys_map = {t: reg_keys_for(obs_space, t) for t in tickers}
    seqs = {t: RollingSeq(T) for t in tickers}
    rnn_states = {t: None for t in tickers}
    episode_flags = {t: np.array([True], dtype=bool) for t in tickers}

    # Reset env → initial raw nested obs; pre-fill each (T,D) with the first frame
    ret = env.reset()
    raw = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret
    for t in tickers:
        v = flatten_regular(raw, obs_space, t, reg_keys_map[t])
        seqs[t].fill_with(v)

    done = False
    while not done:
        # Compute rule stack (K, n_tickers) once per time-step
        rule_stack = compute_rule_stack(
            strategies,
            tick=env.current_tick,
            itick=env.i_current_tick,
            portfolio=env.portfolio,
            n_tickers=len(tickers),
        )

        # Predict one action per ticker
        actions = np.zeros((len(tickers),), dtype=np.float32)
        for i, t in enumerate(tickers):
            obs = {"seq_regular": seqs[t].array()}
            if rule_stack is not None:
                obs["rule_actions"] = rule_stack[:, i]  # (K,)
            act, rnn_states[t] = model.predict(
                obs, state=rnn_states[t], episode_start=episode_flags[t], deterministic=True
            )
            actions[i] = float(np.squeeze(act))
            episode_flags[t][:] = False

        # Step env with the FULL action vector
        raw, reward, terminated, truncated, info = env.step(actions)
        done = bool(terminated) or bool(truncated)

        # Push next frame into each ticker's rolling window
        for t in tickers:
            v = flatten_regular(raw, obs_space, t, reg_keys_map[t])
            seqs[t].push(v)

    # Read portfolio history
    hist = env.portfolio.total_value_history
    dates = hist.index.to_list()
    values = hist["Total Value"].astype(float).to_list()
    return dates, values


def rollout_multi_total_with_rule_only(
    df_ext: pd.DataFrame,
    tickers: List[str],
    total_asset: float,
    buyin_portion: float,
    lookahead: int,
    start_at: pd.Timestamp,
    strategy,
) -> Tuple[List, List]:
    """
    Drive a multi-ticker TradeEnv using a single *rule* strategy only.
    """
    env = make_multi_env(df_ext, tickers, total_asset, buyin_portion, lookahead, start_at)
    _ = env.reset()
    done = False
    while not done:
        actions = strategy.trade(env.current_tick, env.i_current_tick, env.portfolio).astype(np.float32)  # (n_tickers,)
        raw, reward, terminated, truncated, info = env.step(actions)
        strategy.last_action = env.last_action
        done = bool(terminated) or bool(truncated)

    hist = env.portfolio.total_value_history
    dates = hist.index.to_list()
    values = hist["Total Value"].astype(float).to_list()
    return dates, values


# ------------------------- yearly evaluation (multi-ticker) -------------------------

def evaluate_split_yearly_multi_total(
    model,
    df_full: pd.DataFrame,
    tickers: List[str],
    total_asset: float,
    buyin_portion: float,
    lookahead: int,
    T: int,
    split_name: str,
    start: str,
    end: str,
    rule_components: List[str],
    qqq_close_series: pd.Series,  # cached once and reused
    interval: str = "1d",
    log_wandb: bool = True,
) -> Dict:
    """
    For each YEAR in [start, end]:
      - Build one multi-ticker TradeEnv for RL and for each rule strategy (separately)
      - Rollout RL (using per-ticker single-asset policy calls) → total portfolio curve
      - Rollout each rule baseline → curves
      - Plot only TOTAL portfolio value (no per-stock plots)
      - Compute yearly returns and log to W&B
    """
    import wandb

    results = {"per_year": {}, "plots": {}}

    for y_start, y_end in year_ranges(start, end):
        yr = y_start.year

        # slice with T-1 prehistory for ALL tickers
        df_y = df_full[[c for c in df_full.columns if c[1] in tickers]]
        df_ext, start_at = slice_with_history(df_y, str(y_start.date()), str(y_end.date()), T)

        # Strategies for multi-ticker
        strategies = build_strategies_multi(df_ext, tickers, rule_components or [])

        # RL (multi-ticker env, single-asset policy per ticker) — now uses shared helpers
        env_rl = make_multi_env(df_ext, tickers, total_asset, buyin_portion, lookahead, start_at)
        d_rl, v_rl = rollout_multi_total_with_model(model, env_rl, df_ext, tickers, T, strategies)

        # Rule baselines
        curves = {"RL": (d_rl, v_rl)}
        for s in strategies:
            d_rule, v_rule = rollout_multi_total_with_rule_only(
                df_ext, tickers, total_asset, buyin_portion, lookahead, start_at, s
            )
            curves[f"Rule:{s.name}"] = (d_rule, v_rule)

        # QQQ baseline (cached close → normalize to RL start)
        s = qqq_close_series.loc[pd.to_datetime(d_rl[0]): pd.to_datetime(d_rl[-1])]
        v_qqq = (s / s.iloc[0] * float(v_rl[0])).tolist()
        curves["QQQ"] = (s.index.to_list(), v_qqq)

        # Plot single figure (total value only)
        out_png = f"figures/{split_name}_{yr}_total_value.png"
        plot_total_value_curves(curves, f"{split_name.capitalize()} {yr} — Total Value", out_png)

        # Stats
        ret_rl = float(v_rl[-1] / v_rl[0] - 1.0)
        stats = {"rl_return": ret_rl}
        for k, (_, vals) in curves.items():
            if k == "RL":
                continue
            stats[f"{k}_return"] = float(vals[-1] / vals[0] - 1.0)
        results["per_year"][yr] = stats
        results["plots"][yr] = out_png

        if log_wandb:
            wandb.log({f"{split_name}/{yr}/rl_ret": ret_rl})
            for k, v in stats.items():
                if k != "rl_return":
                    wandb.log({f"{split_name}/{yr}/{k}": v})
            wandb.log({f"{split_name}/{yr}/plot_total": wandb.Image(out_png)})

    return results
