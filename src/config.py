# config.py
from __future__ import annotations

# ---------- Data ranges ----------
# Defaults requested by you:
TRAIN_START = "2005-01-01"
TRAIN_END   = "2018-12-31"

VAL_START   = "2019-01-01"
VAL_END     = "2020-12-31"

TEST_START  = "2021-01-01"
TEST_END    = "2024-12-31"

# A little buffer for cached downloads so we can pre-pad T-1 history
CACHE_DOWNLOAD_START = "2004-01-01"
CACHE_DOWNLOAD_END   = "2024-12-31"

INTERVAL = "1d"

# ---------- Env / economics ----------
TOTAL_ASSET    = 1_000_000.0
SINGLE_BUYIN_PORTION  = 1.0
MULTI_BUYIN_PORTION  = 0.2
REWARD_LOOK_AHEAD = 0  # 20

# Rolling history length T for (T, D) sequences
T_SEQUENCE = 30  # history length T (no warm-up forced; we pre-slice with real past)

# ---------- Rule-mixing ----------
# Set an empty list [] to disable rule-mixing entirely
RULE_COMPONENTS = ["sma", "rsi_ma"]
USE_RULE_MIXING = False

# ---------- Algo / nets ----------
FEATURE_TYPE = "lstm"       # "lstm" or "mlp"   (feature extractor)
# FEATURE_TYPE = "mlp"       # "lstm" or "mlp"   (feature extractor)
POLICY_TYPE  = "recurrent"  # "recurrent" (RecurrentPPO) or "mlp" (PPO)
# POLICY_TYPE  = "mlp"  # "recurrent" (RecurrentPPO) or "mlp" (PPO)

LSTM_ACT_HIDDEN = 512       # actor/critic recurrent body hidden size (sb3-contrib)
MIX_HIDDEN      = 64        # hidden width for the mixing MLP (latent_pi -> 1+K)

# ---------- PPO ----------
# steps_per_iteration = n_steps * num_envs
# iterations = total_timesteps / steps_per_iteration = total_timesteps_per_env / n_steps
TOTAL_STEPS_PER_ENV = 250_000  # multiplied by number of training envs
N_STEPS     = 128
BATCH_SIZE  = 128
LEARNING_RATE = 3e-4
N_EPOCHS    = 10
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_RANGE  = 0.2
ENT_COEF    = 0.01
VF_COEF     = 0.5
MAX_GRAD_NORM = 0.5
# MAX_GRAD_NORM = 1e8
SEED        = 42

# ---------- Logging / cache ----------
EVAL_EVERY = 40_000     # periodic eval steps
USE_WANDB  = True
WANDB_PROJECT = "clstm_ppo_rule_aug"

CACHE_DIR  = "data_cache"
ETF_BENCH  = "QQQ"       # used for plots; cached too

######## Date Ranges ########

TRAIN_START_DATE = "2005-01-01"
TRAIN_END_DATE = "2018-12-31"

TEST_START_DATE = "2019-01-01"
TEST_END_DATE = "2020-12-31"

TRADE_START_DATE = "2021-01-01"
TRADE_END_DATE = "2025-10-01"


MID_FREQ_TRAIN_START_DATE = "2023-11-20"
MID_FREQ_TRAIN_END_DATE = "2024-12-31"
MID_FREQ_TEST_START_DATE = "2025-01-01"
MID_FREQ_TEST_END_DATE = "2025-06-30"
MID_FREQ_TRADE_START_DATE = "2025-07-01"
MID_FREQ_TRADE_END_DATE = "2025-11-10"


HIGH_FREQ_TRAIN_START_DATE = "2025-09-17"
HIGH_FREQ_TRAIN_END_DATE = "2025-10-31"
HIGH_FREQ_TEST_START_DATE = "2025-11-01"
HIGH_FREQ_TEST_END_DATE = "2025-11-07"
HIGH_FREQ_TRADE_START_DATE = "2025-11-08"
HIGH_FREQ_TRADE_END_DATE = "2025-11-14"



######## Directory Paths ########
DATA_DICT = {
    "NASDAQ100":{
        "1D": "dataset/nasdaq100_1d.csv",
        "1h": "dataset/nasdaq100_1h.csv",
        "30m": "dataset/nasdaq100_30m.csv",
        "15m": "dataset/nasdaq100_15m.csv",
    },
    "DOW30":{
        "1D": "dataset/dow30_1d.csv",
        "1h": "dataset/dow30_1h.csv",
        "30m": "dataset/dow30_30m.csv",
        "15m": "dataset/dow30_15m.csv",
    },
    "QQQ":{
        "1D": "dataset/qqq_1d.csv",
        "1h": "dataset/qqq_1h.csv",
        "30m": "dataset/qqq_30m.csv",
        "15m": "dataset/qqq_15m.csv",
    }
}

import os
DATA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))