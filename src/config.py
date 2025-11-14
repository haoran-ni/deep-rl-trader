######## Date Ranges ########

TRAIN_START_DATE = "2005-01-01"
TRAIN_END_DATE = "2018-12-31"

TEST_START_DATE = "2019-01-01"
TEST_END_DATE = "2020-12-31"

TRADE_START_DATE = "2021-01-01"
TRADE_END_DATE = "2025-10-01"


######## Directory Paths ########
DATA_DICT = {
    "NASDAQ100":{
        "1D": "dataset/nasdaq100_1d.csv",
        "1h": "dataset/nasdaq100_1h.csv",
        "30m": "dataset/nasdaq100_30m.csv",
    },
    "DOW30":{
        "1D": "dataset/dow30_1d.csv",
        "1h": "dataset/dow30_1h.csv",
        "30m": "dataset/dow30_30m.csv",
    },
    "QQQ":{
        "1D": "dataset/qqq_1d.csv",
        "1h": "dataset/qqq_1h.csv",
        "30m": "dataset/qqq_30m.csv",
    }
}

import os
DATA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))