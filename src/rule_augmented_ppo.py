# rule_augmented_ppo.py
from __future__ import annotations
import os
import torch
import torch.nn as nn
import wandb

from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data_utils import load_single_ticker_df, slice_by_dates, make_wrapped_env
from features import PreLSTMFeaturesSingle, PreMLPFeaturesSingle
from policies import RuleAugmentedRecurrentPolicySingle, RuleAugmentedMLPPolicySingle
from callbacks import EvalAndWB
from eval_utils import evaluate_one_split

def build_policy_and_kwargs(cfg, obs_space: spaces.Dict):
    """Select (policy_class, algo_class, policy_kwargs) based on feature/policy types."""
    feature_type = cfg["feature_type"]      # "lstm" or "mlp"
    policy_type  = cfg["policy_type"]       # "recurrent" or "mlp"

    # Choose features extractor
    if feature_type == "lstm":
        fe_class = PreLSTMFeaturesSingle
        fe_kwargs = dict(features_dim=256, lstm_hidden=128, lstm_layers=1, fc_hidden=(256, 256),
                         use_rule_mixing=cfg["use_rule_mixing"], fc_activation=nn.SiLU)
    elif feature_type == "mlp":
        fe_class = PreMLPFeaturesSingle
        fe_kwargs = dict(features_dim=256, fc_hidden=(256, 256),
                         use_rule_mixing=cfg["use_rule_mixing"], fc_activation=nn.SiLU)
    else:
        raise ValueError("feature_type must be 'lstm' or 'mlp'")

    # Choose policy + algorithm
    if policy_type == "recurrent":
        policy_class = RuleAugmentedRecurrentPolicySingle
        algo_class = RecurrentPPO
        common_kwargs = dict(
            features_extractor_class=fe_class,
            features_extractor_kwargs=fe_kwargs,
            lstm_hidden_size=cfg["lstm_act_hidden"],
            n_lstm_layers=1,
            shared_lstm=False,
            activation_fn=nn.Tanh,
            ortho_init=False,
            net_arch=[64, 64],  # default
            use_rule_mixing=cfg["use_rule_mixing"],
            mix_hidden=cfg["mix_hidden"],
            mix_activation=nn.SiLU,
        )
    elif policy_type == "mlp":
        policy_class = RuleAugmentedMLPPolicySingle
        algo_class = PPO
        common_kwargs = dict(
            features_extractor_class=fe_class,
            features_extractor_kwargs=fe_kwargs,
            activation_fn=nn.Tanh,
            ortho_init=False,
            net_arch=[64, 64],  # default
            use_rule_mixing=cfg["use_rule_mixing"],
            mix_hidden=cfg["mix_hidden"],
            mix_activation=nn.SiLU,
        )
    else:
        raise ValueError("policy_type must be 'recurrent' or 'mlp'")

    return policy_class, algo_class, common_kwargs

def main():
    cfg = dict(
        # data/splits
        ticker="NVDA",
        interval="1d",
        # env
        total_asset=1_000_000.0,
        buyin_portion=0.2,
        lookahead=20,
        T=30,
        # T=1,
        warmup_steps=None,    # or integer; default uses T-1
        # rule mixing
        use_rule_mixing=True,
        # use_rule_mixing=False,
        # algorithm & nets
        feature_type="lstm",      # "lstm" or "mlp"
        # feature_type="mlp",      # "lstm" or "mlp"
        policy_type="recurrent",  # "recurrent" (RecurrentPPO) or "mlp" (PPO)
        # policy_type="mlp",  # "recurrent" (RecurrentPPO) or "mlp" (PPO)
        lstm_act_hidden=512,
        mix_hidden=64,
        # train
        total_steps=300_000,
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        # eval/log
        eval_every=20_000,
        use_wandb=True,
        project="clstm_ppo_rule_aug",
    )

    # data & splits
    df_all = load_single_ticker_df(cfg["ticker"], "2013-01-01", "2025-01-01", cfg["interval"])
    df_train = slice_by_dates(df_all, "2014-01-01", "2020-12-31")
    df_val   = slice_by_dates(df_all, "2021-01-01", "2022-12-31")
    df_test  = slice_by_dates(df_all, "2023-01-01", "2024-12-31")

    # envs
    warmup = cfg["warmup_steps"] or cfg["T"] - 1
    train_env = make_wrapped_env(df_train, cfg["ticker"], cfg["total_asset"], cfg["buyin_portion"], cfg["lookahead"], cfg["T"], warmup)
    val_env   = make_wrapped_env(df_val,   cfg["ticker"], cfg["total_asset"], cfg["buyin_portion"], cfg["lookahead"], cfg["T"], warmup)
    test_env  = make_wrapped_env(df_test,  cfg["ticker"], cfg["total_asset"], cfg["buyin_portion"], cfg["lookahead"], cfg["T"], warmup)
    vec_train = DummyVecEnv([lambda: train_env])

    # build policy/algo
    policy_class, algo_class, policy_kwargs = build_policy_and_kwargs(cfg, train_env.observation_space)

    # wandb single run
    if cfg["use_wandb"] and getattr(wandb, "run", None) is None:
        wandb.init(project=cfg["project"], config={**cfg}, reinit=False)

    # algorithm
    model = algo_class(
        policy=policy_class,
        env=vec_train,
        learning_rate=3e-4,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        seed=cfg["seed"],
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # callback with periodic val logging
    eval_cb = EvalAndWB(val_env, cfg["eval_every"], cfg, df_val, cfg["ticker"])

    # train
    model.learn(total_timesteps=cfg["total_steps"], callback=eval_cb)
    model.save("clstm_ppo_rule_aug_single_asset.zip")

    # final evaluation & plots
    results_val  = evaluate_one_split(model, val_env,  df_val,  cfg["ticker"], cfg["total_asset"], split_name="val")
    results_test = evaluate_one_split(model, test_env, df_test, cfg["ticker"], cfg["total_asset"], split_name="test")

    # log plots
    if getattr(wandb, "run", None) is not None:
        for split, res in [("val", results_val), ("test", results_test)]:
            wandb.log({
                f"{split}/final_value": float(res["value_curve"][1][-1]),
                f"{split}/return": float(res["value_curve"][1][-1] / res["value_curve"][1][0] - 1.0),
                f"{split}/curve": wandb.Image(res["paths"]["curve"]),
                f"{split}/shares": wandb.Image(res["paths"]["shares"]),
                f"{split}/price": wandb.Image(res["paths"]["price"]),
            })

        wandb.finish()

    print("Saved model → clstm_ppo_rule_aug_single_asset.zip; plots → ./figures/")

if __name__ == "__main__":
    main()
