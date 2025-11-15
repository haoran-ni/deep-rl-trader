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