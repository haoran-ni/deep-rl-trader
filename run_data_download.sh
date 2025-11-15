#!/bin/bash
# Script to run data download for experiments

tickers=("NASDAQ100" "DOW30" "QQQ")
time_intervals=("1D" "1h" "15m" "30m")

for ticker in "${tickers[@]}"; do
  for interval in "${time_intervals[@]}"; do
    echo "Downloading data for $ticker with interval $interval"
    python src/download_data.py --tickers_group $ticker --time_interval $interval
  done
done
