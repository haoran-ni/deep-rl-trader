# features.py
from __future__ import annotations
from typing import Optional, Tuple, Type, Dict

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PreLSTMFeaturesSingle(BaseFeaturesExtractor):
    """
    Input per step (batched):
      obs["seq_regular"]: (B, T, D)
      obs["rule_actions"]: (B, K)
    Only seq_regular goes through LSTM → MLP → features_dim.
    rule_actions is cached for the policy to read (not through LSTM).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        fc_hidden: Tuple[int, ...] = (256, 256),
        use_rule_mixing: bool = True,
        fc_activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__(observation_space, features_dim)
        self.use_rule_mixing = bool(use_rule_mixing)

        seq_space = observation_space.spaces["seq_regular"]
        assert isinstance(seq_space, spaces.Box) and len(seq_space.shape) == 2
        T, D = seq_space.shape
        self.T, self.D = int(T), int(D)

        self.lstm = nn.LSTM(
            input_size=self.D, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True
        )

        layers = []
        last = lstm_hidden
        for h in fc_hidden:
            layers += [nn.Linear(last, h), fc_activation()]
            last = h
        layers += [nn.Linear(last, features_dim), fc_activation()]
        self.post = nn.Sequential(*layers)

        self._last_rule: Optional[torch.Tensor] = None  # (B, K)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = obs["seq_regular"].float()               # (B,T,D)
        out, (h_n, _) = self.lstm(seq)                 # out: (B,T,H), h_n: (L,B,H)
        last_h = h_n[-1]                               # (B,H)
        feats = self.post(last_h)                      # (B,F)

        if self.use_rule_mixing:
            rule = obs["rule_actions"].float()
            if rule.dim() == 1:
                rule = rule.unsqueeze(0).expand(seq.size(0), -1)
            self._last_rule = rule
        else:
            self._last_rule = None
        return feats

class PreMLPFeaturesSingle(BaseFeaturesExtractor):
    """
    Pure MLP feature extractor that uses ONLY the most recent frame:
      obs["seq_regular"][:, -1, :] -> MLP -> features_dim
    rule_actions is cached for the policy to read (not through MLP).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        fc_hidden: Tuple[int, ...] = (256, 256),
        use_rule_mixing: bool = True,
        fc_activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__(observation_space, features_dim)
        self.use_rule_mixing = bool(use_rule_mixing)

        seq_space = observation_space.spaces["seq_regular"]
        assert isinstance(seq_space, spaces.Box) and len(seq_space.shape) == 2
        T, D = seq_space.shape
        self.T, self.D = int(T), int(D)

        layers = []
        last = self.D
        for h in fc_hidden:
            layers += [nn.Linear(last, h), fc_activation()]
            last = h
        layers += [nn.Linear(last, features_dim), fc_activation()]
        self.ff = nn.Sequential(*layers)

        self._last_rule: Optional[torch.Tensor] = None

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = obs["seq_regular"].float()               # (B,T,D)
        x = seq[:, -1, :]                               # (B,D) last frame only
        feats = self.ff(x)                              # (B,F)

        if self.use_rule_mixing:
            rule = obs["rule_actions"].float()
            if rule.dim() == 1:
                rule = rule.unsqueeze(0).expand(seq.size(0), -1)
            self._last_rule = rule
        else:
            self._last_rule = None
        return feats
