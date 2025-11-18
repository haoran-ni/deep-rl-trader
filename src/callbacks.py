# callbacks.py
from __future__ import annotations

from typing import List

from stable_baselines3.common.callbacks import BaseCallback

from data_utils import slice_with_history, make_multi_env, build_strategies_multi
from eval_utils import rollout_multi_total_with_model


class PeriodicMultiYearEval(BaseCallback):
    """
    Periodic evaluation during training on *all validation years* and *all validation tickers*
    in MULTI-ticker mode (no plots; no QQQ; no rule baselines).
    We log total-portfolio final value and return per year.

    Note: This keeps training snappy while still reflecting the requested "all val years".
    """

    def __init__(
        self,
        df_val_full,
        val_tickers: List[str],
        total_asset: float,
        buyin_portion: float,
        lookahead: int,
        T: int,
        start: str,
        end: str,
        rule_components: List[str],
        every_steps: int = 20_000,
        use_wandb: bool = True,
    ):
        super().__init__(verbose=0)
        self.df_val_full = df_val_full
        self.val_tickers = val_tickers
        self.total_asset = total_asset
        self.buyin_portion = buyin_portion
        self.lookahead = lookahead
        self.T = T
        self.start = start
        self.end = end
        self.rule_components = rule_components
        self.every = int(every_steps)
        self.use = bool(use_wandb)

    def _on_step(self) -> bool:
        import wandb
        from eval_utils import year_ranges

        if self.every <= 0:
            return True
        if self.num_timesteps % self.every != 0:
            return True

        for y_start, y_end in year_ranges(self.start, self.end):
            df_y = self.df_val_full[[c for c in self.df_val_full.columns if c[1] in self.val_tickers]]
            df_ext, start_at = slice_with_history(df_y, str(y_start.date()), str(y_end.date()), self.T)
            env = make_multi_env(df_ext, self.val_tickers, self.total_asset, self.buyin_portion, self.lookahead, start_at)
            strategies = build_strategies_multi(df_ext, self.val_tickers, self.rule_components or [])
            d_rl, v_rl = rollout_multi_total_with_model(self.model, env, df_ext, self.val_tickers, self.T, strategies)
            if self.use and getattr(wandb, "run", None) is not None:
                wandb.log({
                    f"val/{y_start.year}/ret": float(v_rl[-1] / v_rl[0] - 1.0),
                    "timesteps": self.num_timesteps
                })
        return True
