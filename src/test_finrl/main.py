import os
from typing import List
from argparse import ArgumentParser
from finrl import config
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    TD3_PARAMS,
    A2C_PARAMS,
    PPO_PARAMS,
    DDPG_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)

from config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    DATA_DICT,
    DATA_ROOT_DIR,
)

from tickers import (
    DOW_30_TICKER,
    NAS_100_TICKER,
    QQQ_TICKERS,
)

# construct environment
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

model_params_map = {
    "sac": SAC_PARAMS,
    "td3": TD3_PARAMS,
    "a2c": A2C_PARAMS,
    "ppo": PPO_PARAMS,
    "ddpg": DDPG_PARAMS,
}

tickers_map = {
    "DOW30": DOW_30_TICKER,
    "NASDAQ100": NAS_100_TICKER,
    "QQQ": QQQ_TICKERS,
}

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="train, download_data, test, trade",
        metavar="MODE",
        choices=["train", "test", "trade", "download_data"],
        default="train",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="Name of the model to use",
        metavar="MODEL_NAME",
        choices=["a2c", "ddpg", "td3", "sac", "ppo"],
        default="ppo",
    )
    parser.add_argument(
        "--time_interval",
        dest="time_interval",
        help="Time interval for the data",
        metavar="TIME_INTERVAL",
        choices=["1D", "1h", "30m"],
        default="1D",
    )
    parser.add_argument(
        "--tickers_group",
        dest="tickers_group",
        help="Group of tickers to use",
        metavar="TICKERS_GROUP",
        choices=["DOW30", "NASDAQ100", "QQQ"],
        default="NASDAQ100",
    )
    return parser


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory of the current file
current_directory = os.path.dirname(current_file_path)
def check_and_make_directories(directories: List[str], root_dir: str = None):
    for directory in directories:
        if not os.path.exists(os.path.join(root_dir or current_directory, directory)):
            os.makedirs(os.path.join(root_dir or current_directory, directory))



def main():
    parser = build_parser()
    options = parser.parse_args()
    
    model_cwd = os.path.join(current_directory, f"{options.time_interval}_{options.tickers_group}")
    if not os.path.exists(model_cwd):
        os.makedirs(model_cwd)
        
    check_and_make_directories([TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR], root_dir=model_cwd)

    
    
    if options.mode == "train":
        from finrl import train

        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=tickers_map[options.tickers_group],
            data_source="yahoofinance",
            time_interval=options.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name=options.model_name,
            cwd=os.path.join(model_cwd, TRAINED_MODEL_DIR, options.model_name),
            agent_params=model_params_map[options.model_name], #Stable_Baselines3 params
            erl_params=ERL_PARAMS,
            break_step=1e5,
            if_vix=False,
            data_source_file=os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group][options.time_interval]),
            tensorboard_log_dir=os.path.join(model_cwd,TENSORBOARD_LOG_DIR),
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from finrl import test
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

        account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=tickers_map[options.tickers_group],
            data_source="yahoofinance",
            time_interval=options.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name=options.model_name,
            cwd=os.path.join(model_cwd, TRAINED_MODEL_DIR, options.model_name),
            net_dimension=512,
            if_vix=False,
            data_source_file=os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group][options.time_interval]),
            kwargs=kwargs,
        )
        # Add QQQ baseline
        df_baseline = yf.download("QQQ", start=TEST_START_DATE, end=TEST_END_DATE, interval=options.time_interval, auto_adjust=True)
        baseline_normalized_value = df_baseline['Close'] / df_baseline['Close'].iloc[0]
        totalasset_normalized_value = account_value_erl / account_value_erl[0]
        if np.abs(len(baseline_normalized_value) - len(totalasset_normalized_value)) == 1:
            # align lengths by dropping the last entry of the longer series
            if len(baseline_normalized_value) > len(totalasset_normalized_value):
                baseline_normalized_value = baseline_normalized_value[:-1]
            else:
                totalasset_normalized_value = totalasset_normalized_value[:-1]
        assert len(baseline_normalized_value) == len(totalasset_normalized_value), f"baseline ({len(baseline_normalized_value)}) and test ({len(totalasset_normalized_value)}) account value length not equal!"
        totalasset_normalized_value = pd.DataFrame(totalasset_normalized_value, index=baseline_normalized_value.index, columns=[f'ERL_{options.model_name}'])
        
        # Plot the account value change curve
        import matplotlib.pyplot as plt
        plt.plot(totalasset_normalized_value, label=f'ERL_{options.model_name}')
        plt.plot(baseline_normalized_value, label='QQQ')
        plt.xlabel('Time Step')
        plt.ylabel('Total Asset Percentage')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_cwd,RESULTS_DIR, f'account_value_erl_{options.model_name}.png'))
        
    elif options.mode == "trade":
        # from finrl import trade
        # env = StockTradingEnv
        # kwargs = {}
        # trade(
        #     start_date=TRADE_START_DATE,
        #     end_date=TRADE_END_DATE,
        #     ticker_list=DOW_30_TICKER,
        #     data_source="yahoofinance",
        #     time_interval=options.time_interval,
        #     technical_indicator_list=INDICATORS,
        #     drl_lib="stable_baselines3",
        #     env=env,
        #     model_name=options.model_name,
        #     API_KEY=ALPACA_API_KEY,
        #     API_SECRET=ALPACA_API_SECRET,
        #     API_BASE_URL=ALPACA_API_BASE_URL,
        #     trade_mode='backtesting',
        #     if_vix=True,
        #     kwargs=kwargs,
        # )
        from finrl import test
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

        account_value_erl = test(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=tickers_map[options.tickers_group],
            data_source="yahoofinance",
            time_interval=options.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name=options.model_name,
            cwd=os.path.join(model_cwd, TRAINED_MODEL_DIR, options.model_name),
            net_dimension=512,
            if_vix=False,
            data_source_file=os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group][options.time_interval]),
            kwargs=kwargs,
        )
        # Add QQQ baseline
        df_baseline = yf.download("QQQ", start=TRADE_START_DATE, end=TRADE_END_DATE, interval=options.time_interval, auto_adjust=True)
        baseline_normalized_value = df_baseline['Close'] / df_baseline['Close'].iloc[0]
        totalasset_normalized_value = account_value_erl / account_value_erl[0]
        if np.abs(len(baseline_normalized_value) - len(totalasset_normalized_value)) == 1:
            # align lengths by dropping the last entry of the longer series
            if len(baseline_normalized_value) > len(totalasset_normalized_value):
                baseline_normalized_value = baseline_normalized_value[:-1]
            else:
                totalasset_normalized_value = totalasset_normalized_value[:-1]
        assert len(baseline_normalized_value) == len(totalasset_normalized_value), f"baseline ({len(baseline_normalized_value)}) and test ({len(totalasset_normalized_value)}) account value length not equal!"
        totalasset_normalized_value = pd.DataFrame(totalasset_normalized_value, index=baseline_normalized_value.index, columns=[f'ERL_{options.model_name}'])
        
        # Plot the account value change curve
        import matplotlib.pyplot as plt
        plt.plot(totalasset_normalized_value, label=f'ERL_{options.model_name}')
        plt.plot(baseline_normalized_value, label='QQQ')
        plt.xlabel('Time Step')
        plt.ylabel('Total Asset Percentage')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_cwd,RESULTS_DIR, f'account_value_TRADE_{options.model_name}.png'))
        
    elif options.mode == "download_data":
        time_interval = options.time_interval
        from finrl.meta.data_processor import DataProcessor
        dp = DataProcessor(data_source="wrds")
        data = dp.download_data(
            ticker_list=tickers_map[options.tickers_group],
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE,
            time_interval=time_interval
        )
        data = dp.add_technical_indicator(data, INDICATORS)
        data = dp.add_vix(data)
        data = dp.add_turbulence(data)
        # Save data
        data.to_csv(os.path.join(DATA_ROOT_DIR,DATA_SAVE_DIR, f"{options.tickers_group}_{time_interval}.csv"))
    else:
        raise ValueError("Wrong mode.")


## Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()