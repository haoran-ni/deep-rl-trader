import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils import Portfolio

class RuleBasedTradeEnv(gym.Env):
    
    def __init__(self,
                 data: pd.DataFrame,
                 tickers: list,
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp,
                 rollout_time_length: int,
                 cash: float = 1000000):
        
        super().__init__()
        
        self.data = data
        self.tickers = tickers
        self.start_time = start_time
        self.end_time = end_time
        self.rollout_time_length = rollout_time_length # determines how long the environment runs in units of ticks
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(tickers),), dtype=np.float32)
        self.observation_space = spaces.Dict({ ticker:{
            "open_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "close_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "sma_50": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "sma_200": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "rsi_14": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "shares_held": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "average_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "returns": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        } for ticker in tickers
        })
        self.initial_cash = cash
        self.portfolio = Portfolio(tickers, cash=cash)
        self.current_tick = start_time # pd.Timestamp
        self.i_current_tick = 0
    
    
    def reset(self):
        self.current_tick = self.start_time
        self.i_current_tick = 0
        self.portfolio = Portfolio(self.tickers, cash=self.initial_cash)
        return self._get_observation(self.current_tick, self.portfolio)


    def _get_observation(self, tick: pd.Timestamp, portfolio: dict) -> dict:
        obs = {}
        for ticker in self.tickers:
            obs[ticker] = {
                "open_price": self.data.loc[tick, (ticker, "open")],
                "close_price": self.data.loc[tick, (ticker, "close")],
                "sma_50": self.data.loc[tick, (ticker, "sma_50")],
                "sma_200": self.data.loc[tick, (ticker, "sma_200")],
                "rsi_14": self.data.loc[tick, (ticker, "rsi_14")],
                "shares_held": portfolio[ticker]["shares"],
                "average_price": portfolio[ticker]["average_price"],
                "returns": portfolio[ticker]["returns"]
            }
        return obs