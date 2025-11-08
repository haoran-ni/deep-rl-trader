# callbacks.py
from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback
import wandb

from eval_utils import evaluate_rl_only


class EvalAndWB(BaseCallback):
    """
    Periodically run a lightweight evaluation on `val_env` and wandb.log().
    No QQQ / no pure-rule baseline / no plotting during training.
    Final plots are handled outside via evaluate_one_split().
    """

    def __init__(self, val_env, every_steps: int, cfg, df_val, ticker: str):
        super().__init__(verbose=0)
        self.val_env = val_env
        self.every = int(every_steps)

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps % self.every != 0:
            return True
        if getattr(wandb, "run", None) is None:
            return True

        # RL-only quick eval
        dates, values = evaluate_rl_only(self.model, self.val_env)

        log = {"timesteps": self.num_timesteps}
        if len(values) >= 2:
            log["val/final"] = float(values[-1])
            log["val/ret"] = float(values[-1] / values[0] - 1.0)

        # Optional: log mixing prior if available
        mix = getattr(self.model.policy, "get_mixing_prior", None)
        if callable(mix):
            w = self.model.policy.get_mixing_prior()
            if w is not None:
                for j, ww in enumerate(w[0]):
                    tag = "mix_prior/model" if j == 0 else f"mix_prior/rule_{j}"
                    log[tag] = float(ww)

        wandb.log(log, step=self.num_timesteps)
        return True
