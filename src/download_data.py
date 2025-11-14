import os
from typing import List
from argparse import ArgumentParser

from config import (
    TRAIN_START_DATE,
    TRADE_END_DATE,
    DATA_DICT,
    DATA_ROOT_DIR
)

from tickers import NAS_100_TICKER, DOW_30_TICKER, QQQ_TICKERS
from finrl.config import INDICATORS


ticker_groups = {"NASDAQ100": NAS_100_TICKER, "DOW30": DOW_30_TICKER, "QQQ": QQQ_TICKERS}  # DOW_30_TICKER can be added similarly

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--tickers_group",
        dest="tickers_group",
        help="Group of tickers to download data for",
        metavar="TICKERS_GROUP",
        choices=["DOW30", "NASDAQ100", "QQQ"],
        default="NASDAQ100",
    )
    parser.add_argument(
        "--time_interval",
        dest="time_interval",
        help="Time interval for the data",
        metavar="TIME_INTERVAL",
        choices=["1D", "1h", "30m"],
        default="1D",
    )
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    time_interval = options.time_interval
    from finrl.meta.data_processor import DataProcessor
    dp = DataProcessor(data_source="yahoofinance")
    data = dp.download_data(
        ticker_list=ticker_groups[options.tickers_group.upper()],
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE, # use TRADE_END_DATE to cover full range
        time_interval=time_interval
    )
    data = dp.add_technical_indicator(data, INDICATORS)
    data = dp.add_vix(data)
    data = dp.add_turbulence(data)
    # Save data
    save_path = os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group.upper()][time_interval])
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    data.to_csv(save_path)
    print(f"Data saved to {save_path}")
    
if __name__ == "__main__":
    main()