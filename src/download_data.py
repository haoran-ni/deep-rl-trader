import os
from typing import List
from argparse import ArgumentParser
from tqdm import tqdm
from config import (
    TRAIN_START_DATE,
    TEST_START_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    DATA_DICT,
    DATA_ROOT_DIR,
    MID_FREQ_TRAIN_START_DATE,
    MID_FREQ_TRAIN_END_DATE,
    MID_FREQ_TEST_START_DATE,
    MID_FREQ_TEST_END_DATE,
    MID_FREQ_TRADE_START_DATE,
    MID_FREQ_TRADE_END_DATE,
    HIGH_FREQ_TRAIN_START_DATE,
    HIGH_FREQ_TRAIN_END_DATE,
    HIGH_FREQ_TEST_START_DATE,
    HIGH_FREQ_TEST_END_DATE,
    HIGH_FREQ_TRADE_START_DATE,
    HIGH_FREQ_TRADE_END_DATE,
    MASSIVE_API
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
        choices=["1D", "1h", "30m","15m"],
        default="1D",
    )
    parser.add_argument(
        "--data_source",
        dest="data_source",
        help="Data source to download data from",
        metavar="DATA_SOURCE",
        choices=["yahoofinance", "massive"],
        default="massive",
    )
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    time_interval = options.time_interval
    data_source = options.data_source
    
    def save_data(data, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    
    if data_source == "yahoofinance":
        from finrl.meta.data_processor import DataProcessor
        dp = DataProcessor(data_source="yahoofinance")
        if time_interval == "1D":
            data = dp.download_data(
                ticker_list=ticker_groups[options.tickers_group.upper()],
                start_date=TRAIN_START_DATE,
                end_date=TRADE_END_DATE, # use TRADE_END_DATE to cover full range
                time_interval=time_interval,
                train_test_split_date=TEST_START_DATE,
                test_trade_split_date=TRADE_START_DATE,
            )
        elif time_interval in ["1h", "2h"]:
            data = dp.download_data(
                ticker_list=ticker_groups[options.tickers_group.upper()],
                start_date=MID_FREQ_TRAIN_START_DATE,
                end_date=MID_FREQ_TRADE_END_DATE,
                time_interval=time_interval,
                train_test_split_date=MID_FREQ_TEST_START_DATE,
                test_trade_split_date=MID_FREQ_TRADE_START_DATE,
            )
        elif time_interval in ["30m", "15m"]:
            data = dp.download_data(
                ticker_list=ticker_groups[options.tickers_group.upper()],
                start_date=HIGH_FREQ_TRAIN_START_DATE,
                end_date=HIGH_FREQ_TRADE_END_DATE,
                time_interval=time_interval,
                train_test_split_date=HIGH_FREQ_TEST_START_DATE,
                test_trade_split_date=HIGH_FREQ_TRADE_START_DATE,
            )
        else:
            raise ValueError(f"Unsupported time interval: {time_interval}")
        data = dp.add_technical_indicator(data, INDICATORS)
        data = dp.add_vix(data)
        data = dp.add_turbulence(data)
    elif data_source == "massive":
        from massive import RESTClient
        import pandas as pd
        time_freq = {"1D": "day", "1h": "hour", "30m": "minute", "15m": "minute"}
        ticker_list = ticker_groups[options.tickers_group.upper()]
        ticker = options.ticker_name if hasattr(options, 'ticker_name') else None
        client = RESTClient(MASSIVE_API)
        if ticker in ticker_list:
            ticker_list = [ticker]
        for ticker in ticker_list:
            try:
                data = pd.DataFrame()
                print(f"Downloading data for {ticker}...")
                aggs = []
                for a in tqdm(client.list_aggs(
                    ticker,
                    1,
                    time_freq[time_interval],
                    TRADE_START_DATE,
                    TRADE_END_DATE,
                    adjusted="true",
                    sort="asc",
                    limit=120,
                )):
                    if a is None:
                        continue
                    aggs.append(a)
                    ticker_data = pd.DataFrame([a.__dict__ for a in aggs])
                    ticker_data["ticker"] = ticker
                    data = pd.concat([data, ticker_data], ignore_index=True)
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
                data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
                data.sort_index(inplace=True)
            except Exception as e:
                print(f"Download interrupted by {e}. Saving data collected so far...")
                # Save data:
                save_data(data, os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group.upper()][time_interval]+f"_{ticker}.csv"))
            finally:
                save_data(data, os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group.upper()][time_interval]+f"_{ticker}.csv"))
        return   
    # Save data
    save_path = os.path.join(DATA_ROOT_DIR, DATA_DICT[options.tickers_group.upper()][time_interval])
    save_data(data, save_path)
    
if __name__ == "__main__":
    main()