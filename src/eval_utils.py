# eval_utils.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sb3_contrib import RecurrentPPO


def _predict_step(model, obs, recurrent_state, episode_starts, deterministic: bool):
    if isinstance(model, RecurrentPPO):
        action, recurrent_state = model.predict(
            obs, state=recurrent_state, episode_start=episode_starts, deterministic=deterministic
        )
        return action, recurrent_state
    else:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action, None


def rollout_collect(model, env, deterministic: bool = True):
    # reset handling (obs or (obs, info))
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret

    rnn_state = None
    episode_starts = np.array([True], dtype=bool)
    done = False

    dates: List[pd.Timestamp] = []
    portfolio_values: List[float] = []
    shares_list: List[float] = []
    prices_list: List[float] = []

    # discover base env and ticker
    base_env = env.env if hasattr(env, "env") else env
    ticker = getattr(env, "ticker", None)
    if ticker is None and hasattr(base_env, "tickers"):
        ticker = base_env.tickers[0]

    while not done:
        # record before stepping (portfolio history is updated at env.step)
        # use last known total value
        try:
            tv_df = base_env.portfolio.total_value_history
            if len(tv_df) > 0:
                dates.append(tv_df.index[-1])
                portfolio_values.append(float(tv_df.iloc[-1, 0]))
                shares_list.append(float(base_env.portfolio.holdings_info[ticker]["shares"]))
                prices_list.append(float(base_env.data.loc[tv_df.index[-1]]["Close"][ticker]))
        except Exception:
            pass

        action, rnn_state = _predict_step(model, obs, rnn_state, episode_starts, deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        episode_starts[:] = done

    # final record if missing last tick
    try:
        tv_df = base_env.portfolio.total_value_history
        if len(dates) == 0 or tv_df.index[-1] != dates[-1]:
            dates.append(tv_df.index[-1])
            portfolio_values.append(float(tv_df.iloc[-1, 0]))
            shares_list.append(float(base_env.portfolio.holdings_info[ticker]["shares"]))
            prices_list.append(float(base_env.data.loc[tv_df.index[-1]]["Close"][ticker]))
    except Exception:
        pass

    return dates, portfolio_values, shares_list, prices_list


def evaluate_rl_only(model, env):
    """
    Lightweight, RL-only evaluation for periodic callback (no plotting, no downloads).
    Returns (dates, values).
    """
    dates, values, _, _ = rollout_collect(model, env, deterministic=True)
    return dates, values


def eval_rule_baseline(df, ticker: str, total_asset: float, buyin_portion: float, lookahead: int):
    # re-use your TradeEnv/SMAStrategy (no learning)
    from trade_env import TradeEnv
    from rule_based_strategy import SMAStrategy

    env = TradeEnv(
        data=df,
        tickers=[ticker],
        cash=total_asset,
        buyin_portion=buyin_portion,
        reward_look_ahead=lookahead,
    )
    strat = SMAStrategy(short_term_sma=df['SMA_50'][[ticker]], long_term_sma=df['SMA_200'][[ticker]])

    obs = env.reset()
    done = False
    dates: List[pd.Timestamp] = []
    portfolio_values: List[float] = []
    shares_list: List[float] = []

    while not done:
        # record before step
        tv = env.portfolio.total_value_history
        if len(tv) > 0:
            dates.append(tv.index[-1])
            portfolio_values.append(float(tv.iloc[-1, 0]))
            shares_list.append(float(env.portfolio.holdings_info[ticker]["shares"]))

        actions = strat.trade(env.current_tick).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(actions)
        strat.last_action = env.last_action
        done = bool(truncated)

    # final record
    tv = env.portfolio.total_value_history
    if len(dates) == 0 or tv.index[-1] != dates[-1]:
        dates.append(tv.index[-1])
        portfolio_values.append(float(tv.iloc[-1, 0]))
        shares_list.append(float(env.portfolio.holdings_info[ticker]["shares"]))

    return dates, portfolio_values, shares_list


def qqq_curve(start_ts, end_ts, start_value, interval: str = "1d"):
    q = yf.download(["QQQ"], start=start_ts, end=end_ts, interval=interval, auto_adjust=True).dropna()
    ser = q['Close']['QQQ'].astype(float)
    norm = ser / ser.iloc[0] * float(start_value)
    return norm.index.to_list(), norm.to_list()


# ---------- plotting ----------

def plot_value_three(d_rl, v_rl, d_rule, v_rule, d_qqq, v_qqq, title: str, out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(11, 5), dpi=150)
    df = pd.DataFrame({
        "RL":   pd.Series(v_rl,   index=pd.to_datetime(d_rl)),
        "Rule": pd.Series(v_rule, index=pd.to_datetime(d_rule)),
        "QQQ":  pd.Series(v_qqq,  index=pd.to_datetime(d_qqq)),
    }).sort_index().ffill()
    ax = plt.gca()
    df.plot(ax=ax, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Portfolio Value ($)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path


def plot_shares_two(d_rl, s_rl, d_rule, s_rule, title: str, out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(11, 4), dpi=150)
    df = pd.DataFrame({
        "RL_shares":   pd.Series(s_rl,   index=pd.to_datetime(d_rl)),
        "Rule_shares": pd.Series(s_rule, index=pd.to_datetime(d_rule)),
    }).sort_index().ffill()
    ax = plt.gca()
    df.plot(ax=ax, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("# Shares")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path


def plot_price_one(dates, prices, title: str, out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(11, 4), dpi=150)
    s = pd.Series(prices, index=pd.to_datetime(dates)).sort_index()
    ax = plt.gca()
    s.plot(ax=ax, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Close Price ($)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path


# ---------- high-level final evaluation (with QQQ & rule baseline) ----------

def evaluate_one_split(model, env, df_split, ticker: str, total_asset: float, split_name: str):
    """
    Final evaluation:
      - RL curve (value, shares, price)
      - Rule-only baseline curve (value, shares)
      - QQQ benchmark curve
      - Writes 3 plots: value (RL/Rule/QQQ), shares (RL/Rule), price
    """
    import os

    d_rl, v_rl, s_rl, p_rl = rollout_collect(model, env, deterministic=True)
    d_rule, v_rule, s_rule = eval_rule_baseline(
        df_split, ticker, total_asset, env.env.buyin_portion, env.env.reward_look_ahead
    )
    d_qqq, v_qqq = qqq_curve(d_rl[0], d_rl[-1], v_rl[0])

    os.makedirs("figures", exist_ok=True)
    out_curv = plot_value_three(d_rl, v_rl, d_rule, v_rule, d_qqq, v_qqq,
                                f"{split_name.capitalize()} — RL vs Rule vs QQQ",
                                f"figures/{split_name}_curve.png")
    out_shar = plot_shares_two(d_rl, s_rl, d_rule, s_rule,
                               f"{split_name.capitalize()} — Shares (RL vs Rule)",
                               f"figures/{split_name}_shares.png")
    out_pric = plot_price_one(d_rl, p_rl,
                              f"{split_name.capitalize()} — Close Price",
                              f"figures/{split_name}_price.png")

    return {
        "value_curve": (d_rl, v_rl),
        "rule_value_curve": (d_rule, v_rule),
        "qqq_curve": (d_qqq, v_qqq),
        "shares_rl": (d_rl, s_rl),
        "shares_rule": (d_rule, s_rule),
        "price": (d_rl, p_rl),
        "paths": {"curve": out_curv, "shares": out_shar, "price": out_pric},
    }
