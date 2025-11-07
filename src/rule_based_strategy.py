from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    # strategy is expected to run on a batch of tickers
    def __init__(self):
        self.buy = 1.0
        self.sell = -1.0
        self.hold = 0.0
        self.last_action = self.hold
        self.tickers = []
        self.ticker_to_i = {}
    
    @abstractmethod
    def trade(self, i):
        """Decide the action for each ticker at time index i based on the **CLOSE** price"""
        return
    
    @abstractmethod
    def reset(self):
        """Reset the strategy state"""
        return


class SMAStrategy(Strategy):
    def __init__(self, short_term_sma, long_term_sma):
        super().__init__()
        assert short_term_sma.columns.to_list() == long_term_sma.columns.to_list(), \
            "Dataframes must have the same tickers"
            
        self.short_term_sma = short_term_sma
        self.long_term_sma = long_term_sma
        self.tickers = short_term_sma.columns.to_list()
        self.ticker_to_i = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.trade_history = pd.DataFrame(float("nan"), index=short_term_sma.index, columns=self.tickers)
        self.last_action = np.zeros(len(self.tickers))

    def trade(self, tick):
        # tick is the date index
        actions = np.zeros(len(self.tickers), dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            if self.short_term_sma[ticker].loc[tick] > self.long_term_sma[ticker].loc[tick] and self.last_action[i] != self.buy:
                self.trade_history.loc[tick, ticker] = self.buy
                self.last_action[i] = self.buy
                actions[i] = self.buy
            elif self.short_term_sma[ticker].loc[tick] < self.long_term_sma[ticker].loc[tick] and self.last_action[i] != self.sell:
                self.trade_history.loc[tick, ticker] = self.sell
                self.last_action[i] = self.sell
                actions[i] = self.sell
            else:
                self.trade_history.loc[tick, ticker] = self.hold

        # action for each ticker
        return actions

    def reset(self):
        self.trade_history = pd.DataFrame(float("nan"), index=self.short_term_sma.index, columns=self.tickers)
        self.last_action = np.zeros(len(self.tickers))


class SMAStrategySingleStock(Strategy):
    
    # for one ticker only
    # short_term_sma and long_term_sma are pd.Series
    # e.g. short_term_sma = df['SMA_50']['NVDA']
    
    def __init__(self, short_term_sma, long_term_sma):
        super().__init__()
        
        assert type(short_term_sma) == pd.Series and type(long_term_sma) == pd.Series, \
            "short_term_sma and long_term_sma must be pd.Series"
            
        self.short_term_sma = short_term_sma
        self.long_term_sma = long_term_sma
        self.trade_history = pd.DataFrame(float("nan"), index=short_term_sma.index, columns=["Action"])
        self.last_action = self.hold

    def trade(self, tick):
        # tick is the date index
        if self.short_term_sma.loc[tick] > self.long_term_sma.loc[tick] and self.last_action != self.buy:
            self.trade_history.loc[tick] = self.buy
            self.last_action = self.buy
            action = self.buy
        elif self.short_term_sma.loc[tick] < self.long_term_sma.loc[tick] and self.last_action != self.sell:
            self.trade_history.loc[tick] = self.sell
            self.last_action = self.sell
            action = self.sell
        else:
            self.trade_history.loc[tick] = self.hold

        # action for each ticker
        return action
    
    def reset(self):
        self.trade_history = pd.DataFrame(float("nan"), index=self.short_term_sma.index, columns=["Action"])
        self.last_action = self.hold