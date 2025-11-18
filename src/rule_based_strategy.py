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
        self.name = None
    
    @abstractmethod
    def trade(self, tick, itick, portfolio):
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
            f"Dataframes must have the same tickers, got {short_term_sma.columns.to_list()} and {long_term_sma.columns.to_list()}"
            
        self.short_term_sma = short_term_sma
        self.long_term_sma = long_term_sma
        self.tickers = short_term_sma.columns.to_list()
        self.ticker_to_i = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.trade_history = pd.DataFrame(float("nan"), index=short_term_sma.index, columns=self.tickers)
        self.last_action = np.zeros(len(self.tickers))
        self.name = 'sma'

    def trade(self, tick, itick, portfolio):
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
    

class RsiMAStrategy(Strategy):
    def __init__(self, rsi, rsi_ma, rsi_buy_threshold=30.0, rsi_sell_threshold=70.0):
        super().__init__()
        assert rsi.columns.to_list() == rsi_ma.columns.to_list(), \
            f"Dataframes must have the same tickers, got {rsi.columns.to_list()} and {rsi_ma.columns.to_list()}"

        self.rsi = rsi
        self.rsi_ma = rsi_ma
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.tickers = rsi.columns.to_list()
        self.ticker_to_i = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.trade_history = pd.DataFrame(float("nan"), index=rsi.index, columns=self.tickers)
        self.last_action = np.zeros(len(self.tickers))
        self.name = 'rsi_ma'

    def trade(self, tick, itick, portfolio):
        actions = np.zeros(len(self.tickers), dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            stop_loss = (portfolio.holdings_info[ticker]['returns'] < -0.06)
            oversell_crossover = i > 0 and \
                                 (self.rsi[ticker].loc[tick] > self.rsi_ma[ticker].loc[tick]) and \
                                 (self.rsi[ticker].iloc[itick-1] < self.rsi_ma[ticker].iloc[itick-1]) and \
                                 (self.rsi[ticker].loc[tick] < self.rsi_buy_threshold) and \
                                 (self.rsi_ma[ticker].loc[tick] < self.rsi_buy_threshold) and \
                                 (self.last_action[i] != self.buy)
            overbought_crossover = i > 0 and \
                                   (self.rsi[ticker].loc[tick] < self.rsi_ma[ticker].loc[tick]) and \
                                   (self.rsi[ticker].iloc[itick-1] > self.rsi_ma[ticker].iloc[itick-1]) and \
                                   (self.rsi[ticker].loc[tick] > self.rsi_sell_threshold) and \
                                   (self.rsi_ma[ticker].loc[tick] > self.rsi_sell_threshold) and \
                                   (self.last_action[i] != self.sell)
            if oversell_crossover:
                self.trade_history.loc[tick, ticker] = self.buy
                self.last_action[i] = self.buy
                actions[i] = self.buy
            elif overbought_crossover or stop_loss:
                self.trade_history.loc[tick, ticker] = self.sell
                self.last_action[i] = self.sell
                actions[i] = self.sell
            else:
                self.trade_history.loc[tick, ticker] = self.hold

        # action for each ticker
        return actions

    def reset(self):
        self.trade_history = pd.DataFrame(float("nan"), index=self.rsi.index, columns=self.tickers)
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
        self.name = 'sma_single'

    def trade(self, tick, itick, portfolio):
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