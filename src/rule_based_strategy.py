from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    # strategy is expected to run on a batch of tickers
    def __init__(self):
        self.buy = 1
        self.sell = -1
        self.hold = 0
        self.last_action = self.hold
    
    @abstractmethod
    def trade(self, i):
        """Decide action at time index i"""
        pass


class SMAStrategy(Strategy):
    def __init__(self, short_term_sma, long_term_sma):
        super().__init__()
        assert short_term_sma.columns.to_list() == long_term_sma.columns.to_list(), \
            "Dataframes must have the same tickers"
            
        self.short_term_sma = short_term_sma
        self.long_term_sma = long_term_sma
        self.tickers = short_term_sma.columns.to_list()
        self.trade_history = pd.DataFrame(float("nan"), index=short_term_sma.index, columns=self.tickers)
        self.last_action = np.zeros(len(self.tickers))

    def trade(self, tick):
        # tick is the date index
        actions = np.zeros(len(self.tickers))
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

        # if self.short_term_sma.loc[tick] > self.long_term_sma.loc[tick] and self.last_action != self.buy:
        #     self.trade_history.append((tick, 'BUY'))
        #     self.last_action = self.buy
        #     return self.buy

        # elif self.short_term_sma.loc[tick] < self.long_term_sma.loc[tick] and self.last_action != self.sell:
        #     self.trade_history.append((tick, 'SELL'))
        #     self.last_action = self.sell
        #     return self.sell
        
        # return self.hold