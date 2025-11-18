import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradeEnv(gym.Env):
    
    def __init__(self,
                 data: pd.DataFrame,
                 tickers: list,
                 cash: float = 1000000,
                 buyin_portion: float = 0.2,
                 reward_look_ahead: int = 20,
                 start_at: pd.Timestamp | str | None = None,
                 fee_rate: float = 0.001):
        
        super().__init__()
        
        # initialization
        self.data = data.fillna(0.0).astype(np.float32) # fill NA values with 0.0 # TODO: This affects the price!
        self.index = data.index
        self.tickers = tickers
        self.buyin_portion = buyin_portion
        # This is used to record the last action for each ticker, used by the strategy outside this env
        self.last_action = np.zeros(len(tickers), dtype=np.float32)
        self.fee_rate = fee_rate

        # reward related
        self.reward_look_ahead = reward_look_ahead

        # action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(tickers),), dtype=np.float32)

        # observation space
        # hint: use gymnasium.spaces.utils.flatten to quickly convert to a feature vector
        self.observation_space = spaces.Dict({
            ticker: spaces.Dict({
                # basic market data
                "Open": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "Close": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                # Volume
                "RVOL14": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "RVOL50": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "RVOL200": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                # SMA
                "SMA_14": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "SMA_50": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "SMA_200": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                # RSI
                "RSI_14": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "RSI_50": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "RSI_200": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "RSI_14_MA_7": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                # portfolio related
                "cash": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "total_value": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "shares": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "average_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "returns": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }) for ticker in tickers
        })

        # trading related
        self.initial_cash = cash
        self.portfolio = Portfolio(tickers, cash=cash, fee_rate=fee_rate)
        
        # progression related
        if start_at is None:
            self.start_time = self.index[0]
            self.i_start = 0
        else:
            self.start_time = pd.to_datetime(start_at)
            # i_start = exact index (raise if missing to avoid drift)
            pos = np.searchsorted(self.index.values, np.datetime64(self.start_time))
            assert pos < len(self.index) and self.index[pos] == self.start_time, \
                f"start_at {self.start_time} not in data.index"
            self.i_start = int(pos)

        self.current_tick = self.start_time # pd.Timestamp
        self.i_current_tick = self.i_start
    
    def reset(self):
        # TODO: reset will randomly sample a ticker during training
        self.current_tick = self.start_time
        self.i_current_tick = self.i_start
        self.portfolio = Portfolio(self.tickers, cash=self.initial_cash)
        return self._get_observation(self.current_tick), {}

    def step(self, actions: np.ndarray):
        # NOTE: this function should be called strictly based on time sequence because portfolio is time dependent
        
        # NOTE: actions comes from one strategy, not a list of strategies
        
        # quantize actions to 1 decimal place
        actions = np.round(actions, decimals=1)
        self.last_action = actions.copy()
        
        # actions act on the Open price of the tick
        # the returned state / reward is based on the Close price of the tick
        self.i_current_tick += 1
        self.current_tick = self.index[self.i_current_tick]

        open_prices = self.data.loc[self.current_tick]["Open"]
        close_prices = self.data.loc[self.current_tick]["Close"]

        for i, ticker in enumerate(self.tickers):
            open_price = open_prices[ticker]
            close_price = close_prices[ticker]
            if actions[i] > 0: # e.g. action = 0.5, buy in with 0.5*BUYIN_UNIT
                # buy in at the open price of the next tick
                buyin_price = open_price
                buyin_money = self.buyin_portion * self.portfolio.cash
                # only buy in integer shares
                shares, actual_buyin_money = self._buyin_shares(buyin_money, buyin_price)
                self.portfolio.update_cash(-actual_buyin_money)
                if self.portfolio.cash_enough:
                    self.portfolio.update_trade_history(self.current_tick, 'BUY', ticker, shares, buyin_price)
                    self.portfolio.update_holdings(ticker, shares, buyin_price)
                else:
                    # action not executed, set last action to hold
                    self.last_action[i] = 0.0
                    
            elif actions[i] < 0: # e.g. action = -0.5, sell half of the shares
                # sell at the open price of the next tick
                sell_price = open_price
                # shares is a positive integer number
                holding_shares = self.portfolio.holdings_info[ticker]["shares"]
                if holding_shares > 0:
                    shares = max(1.0, np.floor(holding_shares * (-actions[i])))
                    revenue = self._sell_shares(shares, sell_price)  # revenue after fees
                    self.portfolio.update_cash(revenue)
                    self.portfolio.update_trade_history(self.current_tick, 'SELL', ticker, shares, sell_price)
                    self.portfolio.update_holdings(ticker, -shares, sell_price)
                else:
                    # action not executed, set last action to hold
                    self.last_action[i] = 0.0

        # update portfolio data at the close price
        self.portfolio.step(self.current_tick, close_prices)
        
        obs = self._get_observation(self.current_tick)
        reward, truncated = self._get_reward() # reward.shape = (len(tickers),)
        
        return obs, reward, False, truncated, {}

    def _get_observation(self, tick: pd.Timestamp) -> dict:
        # NOTE: this function should be called strictly based on time sequence because portfolio is time dependent
        obs = {}
        for ticker in self.tickers:
            obs[ticker] = {
                "Open": np.array([self.data.loc[tick]["Open"][ticker]], dtype=np.float32),
                "Close": np.array([self.data.loc[tick]["Close"][ticker]], dtype=np.float32),
                # Volume
                "RVOL14": np.array([self.data.loc[tick]["RVOL14"][ticker]], dtype=np.float32),
                "RVOL50": np.array([self.data.loc[tick]["RVOL50"][ticker]], dtype=np.float32),
                "RVOL200": np.array([self.data.loc[tick]["RVOL200"][ticker]], dtype=np.float32),
                # SMA
                "SMA_14": np.array([self.data.loc[tick]["SMA_14"][ticker]], dtype=np.float32),
                "SMA_50": np.array([self.data.loc[tick]["SMA_50"][ticker]], dtype=np.float32),
                "SMA_200": np.array([self.data.loc[tick]["SMA_200"][ticker]], dtype=np.float32),
                # RSI
                "RSI_14": np.array([self.data.loc[tick]["RSI_14"][ticker]], dtype=np.float32),
                "RSI_50": np.array([self.data.loc[tick]["RSI_50"][ticker]], dtype=np.float32),
                "RSI_200": np.array([self.data.loc[tick]["RSI_200"][ticker]], dtype=np.float32),
                "RSI_14_MA_7": np.array([self.data.loc[tick]["RSI_14_MA_7"][ticker]], dtype=np.float32),
                # portfolio related
                "cash": self.portfolio.cash,
                "total_value": self.portfolio.total_value,
                "shares": self.portfolio.holdings_info[ticker]["shares"],
                "average_price": self.portfolio.holdings_info[ticker]["average_price"],
                "returns": self.portfolio.holdings_info[ticker]["returns"],
            }
            
        return obs

    def _get_reward(self) -> float:
        # TODO: add drawback penalty
        # reward is the return of the current ticker in the next 20 ticks, for example
        reward = np.zeros(len(self.tickers), dtype=np.float32)
        truncated = False
        
        # sanity check
        if self.i_current_tick + self.reward_look_ahead >= len(self.index) - 1:
            truncated = True
            # use the last price available
            future_price = self.data.loc[self.index[-1]]["Close"]
        else:
            future_price = self.data.loc[self.index[self.i_current_tick + self.reward_look_ahead]]["Close"]
        
        for i, ticker in enumerate(self.tickers):
            if self.portfolio.holdings_info[ticker]["shares"] == 0.0:
                reward[i] = 0.0
            else:
                scale = self.portfolio.holdings_info[ticker]["shares"] / self.portfolio.holdings_info[ticker]["initial_shares"]
                average_price = self.portfolio.holdings_info[ticker]["average_price"]
                reward[i] = scale * (future_price[ticker] - average_price) / average_price

        return reward, truncated

    def _buyin_shares(self, buyin_money, price):
        # only buyin integer shares
        price_with_fee = price * (1 + self.fee_rate)
        shares = max(1.0, np.floor(buyin_money / price_with_fee))
        actual_buyin_money = shares * price_with_fee
        return shares, actual_buyin_money

    def _sell_shares(self, shares, price):
        price_after_fee = price * (1 - self.fee_rate)
        revenue = shares * price_after_fee
        return revenue


class Portfolio:
    def __init__(self, tickers, cash=1000000.0, fee_rate=0.001):
        # initialize holdings for each ticker
        self.tickers = tickers
        # initial_shares is used to calculate rewards later
        # we will scale the rewards based on shares/initial_shares
        # e.g. if the model decides to add position on a ticker, the rewards will be scaled up accordingly
        # e.g. if the model decides to reduce position on a ticker, the rewards will be scaled down accordingly
        # Using this, we hope the model can learn to adjust its positions
        self.holdings_info = {ticker: {"initial_shares": 0.0,
                                       "shares": 0.0,
                                       "total_cost": 0.0,
                                       "average_price": 0.0,
                                       "returns": 0.0} for ticker in tickers}
        self.cash = cash
        self.total_value = cash
        self.total_value_history = pd.DataFrame(columns=['Total Value'])
        self.trade_history = {ticker: [] for ticker in tickers}
        self.cash_enough = True
        self.fee_rate = fee_rate
    
    def step(self, tick, current_prices):
        self.total_value = self.cash
        for ticker, current_price in current_prices.items():
            if current_price > 0:
                self.update_returns(ticker, current_price)
                self.update_total_value(ticker, current_price)
        self.total_value_history.loc[tick, 'Total Value'] = self.total_value

    def update_returns(self, ticker, current_price):
        if self.holdings_info[ticker]["shares"] > 0:
            shares = self.holdings_info[ticker]["shares"]
            average_price = self.holdings_info[ticker]["average_price"]
            self.holdings_info[ticker]["returns"] = (current_price - average_price) / average_price

    def update_cash(self, amount):
        self.cash_enough = False if self.cash + amount < 0 else True
        if self.cash_enough:
            self.cash += amount
    
    def update_holdings(self, ticker, shares, price):
        if self.holdings_info[ticker]["shares"] == 0.0:
            self.holdings_info[ticker]["initial_shares"] = shares
        
        self.holdings_info[ticker]["shares"] += shares
        if self.holdings_info[ticker]["shares"] > 0.0:
            self.holdings_info[ticker]["total_cost"] += (price*(1+self.fee_rate)) * shares
            self.holdings_info[ticker]["average_price"] = self.holdings_info[ticker]["total_cost"] / self.holdings_info[ticker]["shares"]
        elif self.holdings_info[ticker]["shares"] == 0.0:
            # reset holdings info if all shares are sold
            self.holdings_info[ticker]["total_cost"] = 0.0
            self.holdings_info[ticker]["average_price"] = 0.0
            self.holdings_info[ticker]["returns"] = 0.0
            self.holdings_info[ticker]["initial_shares"] = 0.0
        else:
            raise ValueError(f"Shares of {ticker} cannot be negative.")
    
    def update_trade_history(self, date, action, ticker, shares, price):
        realized = None
        if action == 'SELL':
            realized = (price*(1-self.fee_rate)-self.holdings_info[ticker]["average_price"]) * shares
        self.trade_history[ticker].append({"date": date,
                                           "action": action,
                                           "shares": shares,
                                           "price": price,
                                           "realized" : realized})
    
    def update_total_value(self, ticker, current_price):
        # note that for the total value, transaction fees are not considered because
        # the fee is only applied when TRANSACTION happens
        self.total_value += self.holdings_info[ticker]["shares"] * current_price