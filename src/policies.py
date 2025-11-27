# policies.py
from __future__ import annotations

from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy


# ---------- small helper: ensure extractor sees "use_rule_mixing" ----------
def _ensure_fe_kwargs(kwargs: dict, use_rule_mixing: bool) -> dict:
    fe_kwargs = kwargs.get("features_extractor_kwargs", {}) or {}
    fe_kwargs.setdefault("use_rule_mixing", use_rule_mixing)
    kwargs["features_extractor_kwargs"] = fe_kwargs
    return kwargs


# ---------- shared mixin for rule mixing ----------
class RuleMixingMixin:
    """
    Provides:
      - _setup_rule_mixing(...): build mix_mlp / mix_bias and set n_rules
      - _calc_mixed_mean(latent_pi, model_mean): combine [model_mean, rule...] with softmax weights
      - get_mixing_prior(): bias-only softmax (for logging)
    Assumes:
      - self.features_extractor caches rule vector at ._last_rule (set by your extractors)
      - self.action_space is a 1-D Box with shape (1,)
    """

    def _setup_rule_mixing(
        self,
        observation_space: spaces.Space,
        use_rule_mixing: bool,
        mix_hidden: int,
        mix_activation: Type[nn.Module],
    ) -> None:
        self.use_rule_mixing = bool(use_rule_mixing)
        self.mix_hidden = int(mix_hidden)
        self.mix_activation = mix_activation

        # Determine K from observation space
        rule_space = None
        if isinstance(observation_space, spaces.Dict):
            rule_space = observation_space.spaces.get("rule_actions", None)
        if self.use_rule_mixing and isinstance(rule_space, spaces.Box) and len(rule_space.shape) == 1:
            self.n_rules = int(rule_space.shape[0])
        else:
            self.n_rules = 0

        if self.use_rule_mixing and self.n_rules > 0:
            self.mix_mlp = nn.Sequential(
                nn.LazyLinear(self.mix_hidden, bias=True),
                self.mix_activation(),
                nn.Linear(self.mix_hidden, 1 + self.n_rules, bias=False),
            )
            # Bias to prefer rules at init (model column is index 0)
            self.mix_bias = nn.Parameter(torch.zeros(1 + self.n_rules))
            with torch.no_grad():
                self.mix_bias[0] = -10.0
        else:
            self.mix_mlp = nn.Identity()
            self.mix_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _calc_mixed_mean(self, latent_pi: torch.Tensor, model_mean: torch.Tensor) -> torch.Tensor:
        """
        Mix model_mean (B,1) with rule vector (B,K) using softmax weights over 1+K logits.
        Returns final mean (B,1).
        """
        if not (self.use_rule_mixing and getattr(self, "n_rules", 0) > 0):
            return model_mean

        fe = self.features_extractor  # PreLSTMFeaturesSingle | PreMLPFeaturesSingle
        rule = getattr(fe, "_last_rule", None)
        if rule is None or rule.numel() == 0:
            return model_mean

        sources = torch.cat([model_mean, rule], dim=1).unsqueeze(-1)  # (B,1+K,1)
        logits = self.mix_mlp(latent_pi) + self.mix_bias               # (B,1+K)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)          # (B,1+K,1)
        mean = (sources * weights).sum(dim=1)                          # (B,1)
        return mean

    def get_mixing_prior(self) -> Optional[np.ndarray]:
        if not (self.use_rule_mixing and getattr(self, "n_rules", 0) > 0):
            return None
        with torch.no_grad():
            w = torch.softmax(self.mix_bias, dim=-1).unsqueeze(0)  # (1,1+K)
            return w.cpu().numpy()


# ---------- Recurrent (LSTM) rule-augmented policy ----------
class RuleAugmentedRecurrentPolicySingle(RuleMixingMixin, RecurrentActorCriticPolicy):
    """
    Same mixing as the MLP policy, but on top of sb3-contrib's RecurrentActorCriticPolicy.
    The model branch is tanh-bounded to keep the final mean in [-1, 1].
    """

    def __init__(
        self,
        *args,
        use_rule_mixing: bool = True,
        mix_hidden: int = 64,
        mix_activation: Type[nn.Module] = nn.SiLU,
        **kwargs,
    ):
        kwargs = _ensure_fe_kwargs(kwargs, use_rule_mixing)
        super().__init__(*args, **kwargs)

        # Check single-asset action space
        a_dim = int(np.prod(self.action_space.shape))
        assert a_dim == 1, "single-asset (A=1) expected"

        # Build shared mixing head
        self._setup_rule_mixing(self.observation_space, use_rule_mixing, mix_hidden, mix_activation)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        # Bound model branch in [-1,1]; distribution remains Gaussian (unsquashed)
        model_mean = torch.tanh(self.action_net(latent_pi))  # (B,1)
        mean = self._calc_mixed_mean(latent_pi, model_mean)
        return self.action_dist.proba_distribution(mean_actions=mean, log_std=self.log_std)


# ---------- Non-recurrent (MLP) rule-augmented policy ----------
class RuleAugmentedMLPPolicySingle(RuleMixingMixin, ActorCriticPolicy):
    """
    Same mixing as recurrent, but for standard PPO's MLP policy.
    """

    def __init__(
        self,
        *args,
        use_rule_mixing: bool = True,
        mix_hidden: int = 64,
        mix_activation: Type[nn.Module] = nn.SiLU,
        **kwargs,
    ):
        kwargs = _ensure_fe_kwargs(kwargs, use_rule_mixing)
        super().__init__(*args, **kwargs)

        a_dim = int(np.prod(self.action_space.shape))
        assert a_dim == 1, "single-asset (A=1) expected"

        self._setup_rule_mixing(self.observation_space, use_rule_mixing, mix_hidden, mix_activation)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        model_mean = torch.tanh(self.action_net(latent_pi))  # (B,1)
        mean = self._calc_mixed_mean(latent_pi, model_mean)
        return self.action_dist.proba_distribution(mean_actions=mean, log_std=self.log_std)
