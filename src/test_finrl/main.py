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
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    TD3_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)

# construct environment
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="train, download_data, test, trade",
        metavar="MODE",
        default="train",
    )
    return parser


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory of the current file
current_directory = os.path.dirname(current_file_path)
def check_and_make_directories(directories: List[str]):
    for directory in directories:
        if not os.path.exists(os.path.join(current_directory, directory)):
            os.makedirs(os.path.join(current_directory, directory))



def main():
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    if options.mode == "train":
        from finrl import train

        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="td3",
            cwd="./test_td3",
            agent_params=TD3_PARAMS, #Stable_Baselines3 params
            erl_params=ERL_PARAMS,
            break_step=1e5,
            if_vix=False,
            data_source_file=os.path.join(current_directory,DATA_SAVE_DIR, "dow_30_data_1D.csv"),
            tensorboard_log_dir=os.path.join(current_directory,TENSORBOARD_LOG_DIR),
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from finrl import test
        import yfinance as yf
        import pandas as pd
        
        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

        account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            net_dimension=512,
            if_vix=False,
            data_source_file=os.path.join(current_directory,DATA_SAVE_DIR, "dow_30_data_1D.csv"),
            kwargs=kwargs,
        )
        # Add QQQ baseline
        df_baseline = yf.download("QQQ", start=TEST_START_DATE, end=TEST_END_DATE, interval="1D", auto_adjust=True)
        baseline_normalized_value = df_baseline['Close'] / df_baseline['Close'].iloc[0]
        totalasset_normalized_value = account_value_erl / account_value_erl[0]
        assert len(baseline_normalized_value) == len(totalasset_normalized_value), "baseline and test account value length not equal!"
        totalasset_normalized_value = pd.DataFrame(totalasset_normalized_value, index=baseline_normalized_value.index, columns=['ERL_PPO'])
        
        # Plot the account value change curve
        import matplotlib.pyplot as plt
        plt.plot(totalasset_normalized_value, label='ERL_PPO')
        plt.plot(baseline_normalized_value, label='QQQ')
        plt.xlabel('Time Step')
        plt.ylabel('Total Asset Percentage')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(current_directory,RESULTS_DIR, f'account_value_erl_ppo.png'))
        
    elif options.mode == "trade":
        from finrl import trade
        env = StockTradingEnv
        kwargs = {}
        trade(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            API_KEY=ALPACA_API_KEY,
            API_SECRET=ALPACA_API_SECRET,
            API_BASE_URL=ALPACA_API_BASE_URL,
            trade_mode='backtesting',
            if_vix=True,
            kwargs=kwargs,
        )
    elif options.mode == "download_data":
        time_interval = "1D"
        from finrl.meta.data_processor import DataProcessor
        dp = DataProcessor(data_source="yahoofinance")
        data = dp.download_data(
            ticker_list=DOW_30_TICKER,
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE,
            time_interval=time_interval
        )
        data = dp.add_technical_indicator(data, INDICATORS)
        data = dp.add_vix(data)
        data = dp.add_turbulence(data)
        # Save data
        data.to_csv(os.path.join(current_directory,DATA_SAVE_DIR, f"dow_30_data_{time_interval}.csv"))
    else:
        raise ValueError("Wrong mode.")


## Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()