# rule_augmented_ppo.py
from __future__ import annotations

import os

import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import PeriodicMultiYearEval
from config import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    CACHE_DOWNLOAD_START, CACHE_DOWNLOAD_END, INTERVAL,
    TOTAL_ASSET, SINGLE_BUYIN_PORTION, MULTI_BUYIN_PORTION, REWARD_LOOK_AHEAD, T_SEQUENCE,
    RULE_COMPONENTS, USE_RULE_MIXING,
    FEATURE_TYPE, POLICY_TYPE, LSTM_ACT_HIDDEN, MIX_HIDDEN,
    TOTAL_STEPS_PER_ENV, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF, VF_COEF, MAX_GRAD_NORM, SEED,
    EVAL_EVERY, USE_WANDB, WANDB_PROJECT, CACHE_DIR, ETF_BENCH, LEARNING_RATE,
)
from data_utils import (
    load_panel_cached, add_indicators, slice_with_history,
    make_wrapped_env, load_qqq_cached
)
from eval_utils import evaluate_split_yearly_multi_total
from features import PreLSTMFeaturesSingle, PreMLPFeaturesSingle
from policies import RuleAugmentedRecurrentPolicySingle, RuleAugmentedMLPPolicySingle
from tickers import MULTIPLE_TICKERS, QQQ_TICKERS


# ------------------------- policy builder -------------------------

def build_policy_and_kwargs(cfg):
    """
    Select (policy_class, algo_class, policy_kwargs) based on feature/policy types.
    We keep SB3 internal activation as Tanh, but our custom extractors/mixing MLP use SiLU.
    """
    feature_type = cfg["feature_type"]
    policy_type  = cfg["policy_type"]

    # features extractor
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

    # policy + algorithm
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
    # -------------------- Config --------------------
    cfg = dict(
        # data & splits
        interval=INTERVAL,
        train_tickers=MULTIPLE_TICKERS,
        val_tickers=QQQ_TICKERS,
        test_tickers=QQQ_TICKERS,
        train_start=TRAIN_START, train_end=TRAIN_END,
        val_start=VAL_START,     val_end=VAL_END,
        test_start=TEST_START,   test_end=TEST_END,

        # env / econ
        total_asset=TOTAL_ASSET,
        single_buyin_portion=SINGLE_BUYIN_PORTION,
        multi_buying_portion=MULTI_BUYIN_PORTION,
        lookahead=REWARD_LOOK_AHEAD,
        T=T_SEQUENCE,

        # rule mixing
        use_rule_mixing=USE_RULE_MIXING,
        rule_components=RULE_COMPONENTS,

        # policy/feature
        feature_type=FEATURE_TYPE,
        policy_type=POLICY_TYPE,
        lstm_act_hidden=LSTM_ACT_HIDDEN,
        mix_hidden=MIX_HIDDEN,

        # train
        total_steps_per_env=TOTAL_STEPS_PER_ENV,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF, vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,

        # eval/log
        eval_every=EVAL_EVERY,
        use_wandb=USE_WANDB,
        project=WANDB_PROJECT,
        cache_dir=CACHE_DIR,
        etf_bench=ETF_BENCH,
    )

    # -------------------- Data (cached: 2004..2024 + indicators) --------------------
    all_tickers = sorted(set(cfg["train_tickers"] + cfg["val_tickers"] + cfg["test_tickers"] + [cfg["etf_bench"]]))
    raw_panel = load_panel_cached(all_tickers, cfg["interval"], cfg["cache_dir"],
                                  start=CACHE_DOWNLOAD_START, end=CACHE_DOWNLOAD_END)
    df_all = add_indicators(raw_panel)  # dropna inside

    # Cache QQQ close for later plots
    qqq_close = load_qqq_cached(cfg["etf_bench"], cfg["interval"], cfg["cache_dir"],
                                start=CACHE_DOWNLOAD_START, end=CACHE_DOWNLOAD_END)

    # -------------------- Build train vec envs (one env per train ticker) --------------------
    def make_env_for_split(ticker: str, start: str, end: str):
        df_t = df_all[[c for c in df_all.columns if c[1] == ticker]]
        df_ext, start_at = slice_with_history(df_t, start, end, cfg["T"])
        return make_wrapped_env(df_ext, ticker, cfg["total_asset"], cfg["single_buyin_portion"],
                                cfg["lookahead"], cfg["T"], start_at, cfg["rule_components"])

    train_envs = [
        (lambda t=t: make_env_for_split(t, cfg["train_start"], cfg["train_end"]))
        for t in cfg["train_tickers"]
    ]
    vec_train = DummyVecEnv(train_envs)

    # -------------------- Build policy/algo --------------------
    policy_class, algo_class, policy_kwargs = build_policy_and_kwargs(cfg)

    import wandb
    if cfg["use_wandb"] and getattr(wandb, "run", None) is None:
        wandb.init(project=cfg["project"], config={**cfg}, reinit="return_previous")

    model = algo_class(
        policy=policy_class,
        env=vec_train,
        learning_rate=cfg["learning_rate"],
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

    # Periodic validation: ALL val years in MULTI-ticker mode (no rule/QQQ; no plots)
    df_val_full = df_all[[c for c in df_all.columns if c[1] in cfg["val_tickers"]]]
    eval_cb = PeriodicMultiYearEval(
        df_val_full=df_val_full,
        val_tickers=cfg["val_tickers"],
        total_asset=cfg["total_asset"],
        buyin_portion=cfg["multi_buying_portion"],
        lookahead=cfg["lookahead"],
        T=cfg["T"],
        start=cfg["val_start"], end=cfg["val_end"],
        rule_components=cfg["rule_components"],
        every_steps=cfg["eval_every"],
        use_wandb=cfg["use_wandb"],
    )

    # total timesteps accounts for number of train envs
    total_steps = cfg["total_steps_per_env"] * len(cfg["train_tickers"])
    model.learn(total_timesteps=total_steps, callback=eval_cb)
    model.save("clstm_ppo_rule_aug_single_asset.zip")

    # -------------------- Final evaluation & plots --------------------
    os.makedirs("figures", exist_ok=True)

    # Validation (multi-ticker total value; per-year plots) + log
    val_results = evaluate_split_yearly_multi_total(
        model=model,
        df_full=df_all,
        tickers=cfg["val_tickers"],
        total_asset=cfg["total_asset"],
        buyin_portion=cfg["multi_buying_portion"],
        lookahead=cfg["lookahead"],
        T=cfg["T"],
        split_name="val",
        start=cfg["val_start"], end=cfg["val_end"],
        rule_components=cfg["rule_components"],
        qqq_close_series=qqq_close,
        interval=cfg["interval"],
        log_wandb=cfg["use_wandb"],
    )

    # Test (multi-ticker total value; per-year plots) + log
    test_results = evaluate_split_yearly_multi_total(
        model=model,
        df_full=df_all,
        tickers=cfg["test_tickers"],
        total_asset=cfg["total_asset"],
        buyin_portion=cfg["multi_buying_portion"],
        lookahead=cfg["lookahead"],
        T=cfg["T"],
        split_name="test",
        start=cfg["test_start"], end=cfg["test_end"],
        rule_components=cfg["rule_components"],
        qqq_close_series=qqq_close,
        interval=cfg["interval"],
        log_wandb=cfg["use_wandb"],
    )

    # Log summaries (still include full dict, plus explicit avg convenience keys)
    if cfg["use_wandb"]:
        import wandb
        wandb.log({"val/summary": val_results})
        wandb.log({"test/summary": test_results})
        # Convenience explicit avg keys for quick dashboards
        if "avg" in val_results:
            for k, v in val_results["avg"].items():
                wandb.log({f"val/avg/{k}": v})
        if "avg" in test_results:
            for k, v in test_results["avg"].items():
                wandb.log({f"test/avg/{k}": v})
        wandb.finish()

    print("Saved model → clstm_ppo_rule_aug_single_asset.zip; plots → ./figures/")

    # ---- Return averages for potential Optuna integration
    return {
        "val_avg":  val_results.get("avg", {}),
        "test_avg": test_results.get("avg", {}),
    }


if __name__ == "__main__":
    main()
