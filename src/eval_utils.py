# eval_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TZ
from data_utils import (
    slice_with_history, make_multi_env, build_strategies_multi
)
from wrappers import (
    RollingSeq, reg_keys_for, flatten_regular, compute_rule_stack
)


def plot_total_value_curves(
        curves: Dict[str, Tuple[List, List]],
        title: str,
        out_path: str
) -> str:
    """
    curves: name -> (dates, values); plots only total portfolio value.
    Each curve is plotted separately to avoid tz-naive / tz-aware join issues.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5), dpi=300)
    ax = plt.gca()

    for name, (dates, vals) in curves.items():
        # Convert dates for this curve only
        idx = pd.to_datetime(dates)
        s = pd.Series(vals, index=idx).sort_index().ffill()
        ax.plot(s.index, s.values, label=name, linewidth=1.6)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Total Portfolio Value ($)")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------- RL rollout on multi-ticker env -------------------------

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
    DRY: reuses RollingSeq, reg_keys_for, flatten_regular and compute_rule_stack.
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


# ------------------------- direct period evaluation (multi-ticker) -------------------------

def evaluate_split_multi_total(
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
        rule_baselines: List[str],
        qqq_close_series: pd.Series,
        log_wandb: bool = True,
) -> Dict:
    """
    Evaluate a model on a SINGLE PERIOD (e.g., validation or test) without yearly splitting.
    This is the recommended approach for using specified validation/test periods directly.

    Args:
        model: The trained model to evaluate
        df_full: Full DataFrame with all tickers
        tickers: List of tickers to evaluate
        total_asset: Initial cash
        buyin_portion: Portion of cash to use per trade
        lookahead: Reward look ahead steps
        T: Sequence length
        split_name: Name of the split (e.g., "validation", "test")
        start: Start date of the period
        end: End date of the period
        rule_components: Rule strategies to mix with RL
        rule_baselines: Rule strategies to compare against
        qqq_close_series: QQQ benchmark series
        log_wandb: Whether to log to wandb

    Returns:
        {
            "rl_return": float,
            "Rule:<name>_return": float,
            "QQQ_return": float,
            "plot": "figpath"
        }
    """
    import wandb

    # Slice with T-1 prehistory for ALL tickers
    df_split = df_full[[c for c in df_full.columns if c[1] in tickers]]
    df_ext, start_at = slice_with_history(df_split, start, end, T)

    # Strategies for multi-ticker
    strategies_mix = build_strategies_multi(df_ext, tickers, rule_components or [])
    strategies_baseline = build_strategies_multi(df_ext, tickers, rule_baselines or [])

    # RL (multi-ticker env, single-asset policy per ticker)
    env_rl = make_multi_env(df_ext, tickers, total_asset, buyin_portion, lookahead, start_at, truncation_period=None)
    d_rl, v_rl = rollout_multi_total_with_model(model, env_rl, df_ext, tickers, T, strategies_mix)

    # Rule baselines
    curves = {"RL": (d_rl, v_rl)}
    for s in strategies_baseline:
        d_rule, v_rule = rollout_multi_total_with_rule_only(
            df_ext, tickers, total_asset, buyin_portion, lookahead, start_at, s
        )
        curves[f"Rule:{s.name}"] = (d_rule, v_rule)

    # QQQ baseline (cached close → normalize to RL start)
    start = d_rl[0].tz_convert(TZ)
    end = d_rl[-1].tz_convert(TZ)
    s = qqq_close_series.loc[start:end].copy()

    # 3) Scale and store
    v_qqq = (s / s.iloc[0] * float(v_rl[0])).tolist()
    curves["QQQ"] = (s.index.to_list(), v_qqq)

    # Plot single figure (total value only)
    out_png = f"figures/{split_name}_total_value.png"
    plot_total_value_curves(curves, f"{split_name.capitalize()} — Total Value", out_png)

    # Compute returns
    stats = {"plot": out_png}
    for k, (_, vals) in curves.items():
        ret = float(vals[-1] / vals[0] - 1.0)
        key_name = "rl_return" if k == "RL" else f"{k}_return"
        stats[key_name] = ret

    if log_wandb and getattr(wandb, "run", None) is not None:
        for nm, val in stats.items():
            if nm != "plot":
                wandb.log({f"{split_name}/{nm}": val})
        wandb.log({f"{split_name}/plot_total": wandb.Image(out_png)})

    return stats
