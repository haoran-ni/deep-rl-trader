# callbacks.py
from __future__ import annotations
from typing import List

from stable_baselines3.common.callbacks import BaseCallback

from data_utils import slice_with_history, make_multi_env, build_strategies_multi
from eval_utils import rollout_multi_total_with_model, year_ranges


class PeriodicMultiYearEval(BaseCallback):
    """
    Periodic evaluation during training on *all validation years* and *all validation tickers*
    in MULTI-ticker mode (no plots; no QQQ; no rule baselines).
    Logs per-year return and the **average across years** at each eval step.

    Now also runs once at training step 0.
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

    # ---------- shared eval body (used by both start and periodic hooks) ----------
    def _eval_once(self, step: int) -> None:
        import wandb

        if not self.use or getattr(wandb, "run", None) is None:
            return

        yearly_returns = []
        for y_start, y_end in year_ranges(self.start, self.end):
            # slice with history and build a multi-ticker env
            df_y = self.df_val_full[[c for c in self.df_val_full.columns if c[1] in self.val_tickers]]
            df_ext, start_at = slice_with_history(df_y, str(y_start.date()), str(y_end.date()), self.T)
            env = make_multi_env(df_ext, self.val_tickers, self.total_asset, self.buyin_portion, self.lookahead, start_at)

            # strategies (no plots / QQQ in callback eval)
            strategies = build_strategies_multi(df_ext, self.val_tickers, self.rule_components or [])

            # rollout RL (single-asset policy applied per ticker)
            d_rl, v_rl = rollout_multi_total_with_model(self.model, env, df_ext, self.val_tickers, self.T, strategies)
            ret = float(v_rl[-1] / v_rl[0] - 1.0)
            yearly_returns.append(ret)

            wandb.log({f"val/{y_start.year}/rl_return": ret, "timesteps": step})

        if yearly_returns:
            avg_ret = float(sum(yearly_returns) / len(yearly_returns))
            wandb.log({f"val/avg/rl_return": avg_ret, "timesteps": step})

    # ---------- run once at training start (step = 0) ----------
    def _on_training_start(self) -> None:
        # immediate baseline before any updates
        self._eval_once(step=0)

    # ---------- periodic eval during learning ----------
    def _on_step(self) -> bool:
        # run only at multiples of `every`, and skip the duplicate at 0 (handled above)
        if self.every > 0 and self.num_timesteps > 0 and (self.num_timesteps % self.every) == 0:
            self._eval_once(step=self.num_timesteps)
        return True
