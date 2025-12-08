# rule_augmented_ppo.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import PeriodicMultiEval
from config import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    CACHE_DOWNLOAD_START, CACHE_DOWNLOAD_END, INTERVAL,
    TOTAL_ASSET, SINGLE_BUYIN_PORTION, MULTI_BUYIN_PORTION, REWARD_LOOK_AHEAD, T_SEQUENCE,
    MIX_RULE_COMPONENTS, COMPARE_RULE_BASELINES, USE_RULE_MIXING,
    FEATURE_TYPE, POLICY_TYPE, LSTM_ACT_HIDDEN, MIX_HIDDEN,
    TOTAL_STEPS_PER_ENV, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF, VF_COEF, MAX_GRAD_NORM, SEED,
    EVAL_EVERY, USE_WANDB, WANDB_PROJECT, CACHE_DIR, ETF_BENCH, LEARNING_RATE,
    TRUNCATION_PERIOD,
    USE_ALL_DATASET_TICKERS, LOCAL_CSV_DIR,
)
from data_utils import (
    load_panel_cached, add_indicators, slice_with_history,
    make_wrapped_env, load_qqq_cached, get_available_tickers
)
from eval_utils import evaluate_split_multi_total
from features import PreLSTMFeaturesSingle, PreMLPFeaturesSingle
from policies import RuleAugmentedRecurrentPolicySingle, RuleAugmentedMLPPolicySingle
from tickers import MULTIPLE_TICKERS, QQQ_TICKERS, NASDAQ_100_2025_DATASET_TICKERS


# ------------------------- policy builder -------------------------

def build_policy_and_kwargs(cfg):
    """
    Select (policy_class, algo_class, policy_kwargs) based on feature/policy types.
    We keep SB3 internal activation as Tanh, but our custom extractors/mixing MLP use SiLU.
    """
    feature_type = cfg["feature_type"]
    policy_type = cfg["policy_type"]

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
    # -------------------- Parse command-line arguments --------------------
    parser = argparse.ArgumentParser(description="Train or evaluate RL trading model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "continue"],
                        help="Mode: 'train' (train from scratch), 'eval' (evaluate only), 'continue' (continue training from checkpoint)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to saved model (required for 'eval' and 'continue' modes)")
    args = parser.parse_args()

    # Validate arguments
    if args.mode in ["eval", "continue"] and args.load_model is None:
        raise ValueError(f"--load-model is required when mode is '{args.mode}'")

    # -------------------- Determine ticker lists --------------------
    # If USE_ALL_DATASET_TICKERS is enabled, dynamically load all tickers from dataset
    # Otherwise use manually specified lists from tickers.py
    if USE_ALL_DATASET_TICKERS:
        print(f"[Config] USE_ALL_DATASET_TICKERS=True, loading all tickers from dataset...")
        try:
            dataset_tickers = get_available_tickers(LOCAL_CSV_DIR, INTERVAL)
            print(f"[Config] Found {len(dataset_tickers)} tickers in dataset: {dataset_tickers[:10]}...")
            # Use all dataset tickers for training, validation, and testing
            train_tickers = dataset_tickers
            val_tickers = dataset_tickers
            test_tickers = dataset_tickers
        except Exception as e:
            print(
                f"[Config] Failed to load tickers from dataset ({e}), falling back to NASDAQ_100_2025_DATASET_TICKERS")
            train_tickers = NASDAQ_100_2025_DATASET_TICKERS
            val_tickers = NASDAQ_100_2025_DATASET_TICKERS
            test_tickers = NASDAQ_100_2025_DATASET_TICKERS
    else:
        print(f"[Config] USE_ALL_DATASET_TICKERS=False, using manually specified ticker lists")
        train_tickers = NASDAQ_100_2025_DATASET_TICKERS
        val_tickers = QQQ_TICKERS
        test_tickers = QQQ_TICKERS
        # val_tickers = MULTIPLE_TICKERS
        # test_tickers = MULTIPLE_TICKERS

    # -------------------- Config --------------------
    cfg = dict(
        # data & splits
        interval=INTERVAL,
        train_tickers=train_tickers,
        val_tickers=val_tickers,
        test_tickers=test_tickers,
        train_start=TRAIN_START, train_end=TRAIN_END,
        val_start=VAL_START, val_end=VAL_END,
        test_start=TEST_START, test_end=TEST_END,

        # env / econ
        total_asset=TOTAL_ASSET,
        single_buyin_portion=SINGLE_BUYIN_PORTION,
        multi_buying_portion=MULTI_BUYIN_PORTION,
        lookahead=REWARD_LOOK_AHEAD,
        T=T_SEQUENCE,
        truncation_period=TRUNCATION_PERIOD,  # Training truncation in days; None for full episodes

        # rule mixing
        use_rule_mixing=USE_RULE_MIXING,
        mix_rule_components=MIX_RULE_COMPONENTS,
        compare_rule_baselines=COMPARE_RULE_BASELINES,

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
    all_tickers = sorted(set(cfg["train_tickers"] + cfg["val_tickers"] + cfg["test_tickers"]))
    raw_panel = load_panel_cached(all_tickers, cfg["interval"], cfg["cache_dir"],
                                  start=CACHE_DOWNLOAD_START, end=CACHE_DOWNLOAD_END)
    df_all = add_indicators(raw_panel)  # NaN rows preserved for all tickers; dropped per-ticker in make_wrapped_env
    # Cache QQQ close for later plots
    qqq_close = load_qqq_cached(cfg["etf_bench"], cfg["cache_dir"], start=CACHE_DOWNLOAD_START, end=CACHE_DOWNLOAD_END)

    # -------------------- Build train vec envs (one env per train ticker) --------------------
    def make_env_for_split(ticker: str, start: str, end: str):
        df_t = df_all[[c for c in df_all.columns if c[1] == ticker]]
        df_ext, start_at = slice_with_history(df_t, start, end, cfg["T"])
        return make_wrapped_env(
            df_ext, ticker, cfg["total_asset"], cfg["single_buyin_portion"],
            cfg["lookahead"], cfg["T"], start_at, cfg["mix_rule_components"],
            truncation_period=cfg["truncation_period"]  # Use configured truncation for training
        )

    train_envs = [
        (lambda t=t: make_env_for_split(t, cfg["train_start"], cfg["train_end"]))
        for t in cfg["train_tickers"]
    ]
    vec_train = DummyVecEnv(train_envs)

    # -------------------- Build or load model --------------------
    policy_class, algo_class, policy_kwargs = build_policy_and_kwargs(cfg)

    import wandb
    if cfg["use_wandb"] and getattr(wandb, "run", None) is None:
        wandb.init(project=cfg["project"], config={**cfg}, reinit="return_previous")

    # Create models directory if it doesn't exist
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        # Train from scratch
        print("[Model] Creating new model for training from scratch...")
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
    elif args.mode == "continue":
        # Load model and continue training
        print(f"[Model] Loading model from {args.load_model} to continue training...")
        model = algo_class.load(args.load_model, env=vec_train)
        print(f"[Model] Model loaded successfully. Continuing training...")
    elif args.mode == "eval":
        # Load model for evaluation only
        print(f"[Model] Loading model from {args.load_model} for evaluation only...")
        model = algo_class.load(args.load_model, env=vec_train)
        print(f"[Model] Model loaded successfully. Skipping training, proceeding to evaluation...")

    # -------------------- Training (skip if eval mode) --------------------
    if args.mode != "eval":
        # Periodic validation: ALL val years in MULTI-ticker mode (no rule/QQQ; no plots)
        df_val_full = df_all[[c for c in df_all.columns if c[1] in cfg["val_tickers"]]]
        eval_cb = PeriodicMultiEval(
            df_val_full=df_val_full,
            val_tickers=cfg["val_tickers"],
            total_asset=cfg["total_asset"],
            buyin_portion=cfg["multi_buying_portion"],
            lookahead=cfg["lookahead"],
            T=cfg["T"],
            start=cfg["val_start"], end=cfg["val_end"],
            rule_components=cfg["mix_rule_components"],  # <— use MIX list in callback
            every_steps=cfg["eval_every"],
            use_wandb=cfg["use_wandb"],
        )

        # total timesteps accounts for number of train envs
        total_steps = cfg["total_steps_per_env"] * len(cfg["train_tickers"])
        print(f"[Training] Starting training for {total_steps} total timesteps...")
        model.learn(total_timesteps=total_steps, callback=eval_cb)

        # Save model with timestamp-based unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"ppo_model_{cfg['policy_type']}_{cfg['feature_type']}_{timestamp}.zip"
        model_path = models_dir / model_filename
        model.save(str(model_path))
        print(f"[Model] Saved trained model to {model_path}")

        # Also save to legacy name for backwards compatibility
        legacy_path = "clstm_ppo_rule_aug_single_asset.zip"
        model.save(legacy_path)
        print(f"[Model] Also saved to {legacy_path} (legacy format)")
    else:
        print("[Training] Skipping training (eval mode)...")

    # -------------------- Final evaluation & plots --------------------
    Path("figures").mkdir(exist_ok=True)

    # Validation (multi-ticker total value; direct period evaluation without yearly forcing)
    val_results = evaluate_split_multi_total(
        model=model,
        df_full=df_all,
        tickers=cfg["val_tickers"],
        total_asset=cfg["total_asset"],
        buyin_portion=cfg["multi_buying_portion"],
        lookahead=cfg["lookahead"],
        T=cfg["T"],
        split_name="val",
        start=cfg["val_start"], end=cfg["val_end"],
        rule_components=cfg["mix_rule_components"],
        rule_baselines=cfg["compare_rule_baselines"],
        qqq_close_series=qqq_close,
        log_wandb=cfg["use_wandb"],
    )

    # Test (multi-ticker total value; direct period evaluation without yearly forcing)
    test_results = evaluate_split_multi_total(
        model=model,
        df_full=df_all,
        tickers=cfg["test_tickers"],
        total_asset=cfg["total_asset"],
        buyin_portion=cfg["multi_buying_portion"],
        lookahead=cfg["lookahead"],
        T=cfg["T"],
        split_name="test",
        start=cfg["test_start"], end=cfg["test_end"],
        rule_components=cfg["mix_rule_components"],
        rule_baselines=cfg["compare_rule_baselines"],
        qqq_close_series=qqq_close,
        log_wandb=cfg["use_wandb"],
    )

    # Log summaries
    if cfg["use_wandb"]:
        import wandb
        wandb.log({"val/summary": val_results})
        wandb.log({"test/summary": test_results})
        # Log individual metrics (val_results and test_results are now flat dicts with returns and plot)
        for k, v in val_results.items():
            if k != "plot":  # Skip the plot path
                wandb.log({f"val/{k}": v})
        for k, v in test_results.items():
            if k != "plot":  # Skip the plot path
                wandb.log({f"test/{k}": v})
        wandb.finish()

    print("Saved model → clstm_ppo_rule_aug_single_asset.zip; plots → ./figures/")

    # ---- Return results for potential Optuna integration
    # Remove "plot" key from results for cleaner return
    return {
        "val": {k: v for k, v in val_results.items() if k != "plot"},
        "test": {k: v for k, v in test_results.items() if k != "plot"},
    }


if __name__ == "__main__":
    main()
