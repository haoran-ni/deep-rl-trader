# Script to run FinRL baselines

python src/test_finrl/main.py --mode train --model_name ppo --time_interval 1D --tickers_group NASDAQ100
python src/test_finrl/main.py --mode test --model_name ppo --time_interval 1D --tickers_group NASDAQ100

python src/test_finrl/main.py --mode train --model_name a2c --time_interval 1D --tickers_group NASDAQ100
python src/test_finrl/main.py --mode test --model_name a2c --time_interval 1D --tickers_group NASDAQ100

python src/test_finrl/main.py --mode train --model_name sac --time_interval 1D --tickers_group NASDAQ100
python src/test_finrl/main.py --mode test --model_name sac --time_interval 1D --tickers_group NASDAQ100

python src/test_finrl/main.py --mode train --model_name td3 --time_interval 1D --tickers_group NASDAQ100
python src/test_finrl/main.py --mode test --model_name td3 --time_interval 1D --tickers_group NASDAQ100