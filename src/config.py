# config.py
from __future__ import annotations
from pathlib import Path

# ---------- Data loading ----------
# DEFAULT: Load from local CSV files (faster, no network dependency)
# Set USE_LOCAL_CSV = True to load from local CSV, False to download from yfinance
USE_LOCAL_CSV = True
# USE_LOCAL_CSV = False
LOCAL_CSV_DIR = str(Path(__file__).parent.parent / "dataset" / "nasdaq_100_2025_constituents_processed")

# Automatically use ALL tickers available in the dataset
# If True, will dynamically load all tickers from LOCAL_CSV_DIR at the specified INTERVAL
# If False, will use manually specified ticker lists from tickers.py
# USE_ALL_DATASET_TICKERS = True
USE_ALL_DATASET_TICKERS = False

# ---------- Data ranges ----------
# Note: Many stocks don't have full data from 2015-12, they have various start dates
# The data loading will handle stocks with different lengths automatically
TRAIN_START = "2015-12-07"  # Start from earliest available data
TRAIN_END   = "2023-11-30"  # Train up to end of Nov 2023

VAL_START   = "2023-12-01"  # Validation: Dec 2023 - Nov 2024
VAL_END     = "2024-11-30"

TEST_START  = "2024-12-01"  # Test: Dec 2024 onwards
TEST_END    = "2025-12-01"

# A little buffer for cached downloads so we can pre-pad T-1 history
CACHE_DOWNLOAD_START = "2015-12-01"  # Start from earliest available
CACHE_DOWNLOAD_END   = "2025-12-31"  # Extended to cover test period

# INTERVAL options:
# - For local CSV: "1_hour", "15_mins", "1_min"
# - For yfinance download: "1d", "1h", "30m", "15m"
INTERVAL = "1_hour"  # changed to match local CSV format
# INTERVAL = "1d"  # changed to match local CSV format

TZ = "America/New_York"

# ---------- Env / economics ----------
TOTAL_ASSET    = 1_000_000.0
SINGLE_BUYIN_PORTION  = 1.0
MULTI_BUYIN_PORTION  = 1.0
REWARD_LOOK_AHEAD = 0  # 20

# Rolling history length T for (T, D) sequences
T_SEQUENCE = 30  # history length T (no warm-up forced; we pre-slice with real past)

# Truncation for training (in days). Set to None for no truncation.
# During evaluation, truncation is always disabled (None).
TRUNCATION_PERIOD = None  # Truncate episodes after this many days during training

# ---------- Rule strategies ----------
# (A) Strategies used as *augmenting signals* mixed into the policy during train/eval
MIX_RULE_COMPONENTS = [
    "sma",
    "rsi_ma"
]       # [] disables rule mixing
USE_RULE_MIXING     = True

# (B) Strategies used as *baselines in final plots* (comparison against RL & QQQ)
COMPARE_RULE_BASELINES = [
    "sma",
    "rsi_ma"
]    # [] to disable plotting rule baselines

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
# TOTAL_STEPS_PER_ENV = 250_000  # multiplied by number of training envs
TOTAL_STEPS_PER_ENV = 10_000  # multiplied by number of training envs
# TOTAL_STEPS_PER_ENV = 1  # multiplied by number of training envs
N_STEPS     = 128
BATCH_SIZE  = 128
LEARNING_RATE = 3e-4
N_EPOCHS    = 10
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
# GAE_LAMBDA  = 0.7
CLIP_RANGE  = 0.2
ENT_COEF    = 0.01
# ENT_COEF    = 0.0
VF_COEF     = 0.5
MAX_GRAD_NORM = 0.5
# MAX_GRAD_NORM = 1e8
SEED        = 42

# ---------- Logging / cache ----------
# EVAL_EVERY   = 40_000
EVAL_EVERY   = 120_000
# EVAL_EVERY   = 5_000
USE_WANDB    = True
WANDB_PROJECT = "clstm_ppo_rule_aug_2"

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
        "1D": "massive_dataset/nasdaq100_1d.csv",
        "1h": "massive_dataset/nasdaq100_1h.csv",
        "30m": "massive_dataset/nasdaq100_30m.csv",
        "15m": "massive_dataset/nasdaq100_15m.csv",
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

DATA_ROOT_DIR = str(Path(__file__).parent)


###### API KEYS ########
MASSIVE_API = "SekkXsjUzVVSYEWYytpKll30sX_CCqlJ"
AWS_ACCESS_KEY = "b5e171fa-d7cb-41b5-a6b8-d98d227f47ba"
AWS_SECRET_KEY = "SekkXsjUzVVSYEWYytpKll30sX_CCqlJ"