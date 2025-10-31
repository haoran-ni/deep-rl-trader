import pandas as pd

def buyin_shares(buyin_money, price):
    # currently no fee considered
    # modify this function if fees are needed
    shares = buyin_money / price
    return shares

def sell_shares(shares, price):
    # currently no fee considered
    # modify this function if fees are needed
    revenue = shares * price
    return revenue


class Portfolio:
    def __init__(self, tickers, cash=1000000):
        # initialize holdings for each ticker
        self.tickers = tickers
        self.holdings_info = {ticker: {"shares": 0.0,
                                       "total_cost": 0.0,
                                       "average_price": 0.0,
                                       "returns": 0.0} for ticker in tickers}
        self.cash = cash
        self.total_value = cash
        self.total_value_history = pd.DataFrame(columns=['Total Value'])
        self.trade_history = {ticker: [] for ticker in tickers}
        self.cash_enough = True
    
    def step(self, tick, current_prices):
        self.total_value = self.cash
        for ticker, current_price in current_prices.items():
            # Note that there will be a bug here if current_price is NaN (unlisted stock)
            # During training, please always ensure all stocks are listed
            self.update_returns(ticker, current_price)
            self.update_total_value(ticker, current_price)
        self.total_value_history.loc[tick, 'Total Value'] = self.total_value

    def update_returns(self, ticker, current_price):
        if self.holdings_info[ticker]["shares"] > 0:
            shares = self.holdings_info[ticker]["shares"]
            average_price = self.holdings_info[ticker]["average_price"]
            self.holdings_info[ticker]["returns"] = (current_price - average_price) * shares

    def update_cash(self, amount):
        self.cash_enough = False if self.cash + amount < 0 else True
        if self.cash_enough:
            self.cash += amount
    
    def update_holdings(self, ticker, shares, price):
        # note that the price here does not include fees
        self.holdings_info[ticker]["shares"] += shares
        if self.holdings_info[ticker]["shares"] != 0.0:
            self.holdings_info[ticker]["total_cost"] += price * shares
            self.holdings_info[ticker]["average_price"] = self.holdings_info[ticker]["total_cost"] / self.holdings_info[ticker]["shares"]
        else:
            # reset holdings info if all shares are sold
            self.holdings_info[ticker]["total_cost"] = 0.0
            self.holdings_info[ticker]["average_price"] = 0.0
            self.holdings_info[ticker]["returns"] = 0.0
    
    def update_trade_history(self, date, action, ticker, shares, price):
        if action == 'SELL':
            realized = (price-self.holdings_info[ticker]["average_price"]) * shares
        self.trade_history[ticker].append({"date": date,
                                           "action": action,
                                           "shares": shares,
                                           "price": price,
                                           "realized" : realized if action == 'SELL' else None})
    
    def update_total_value(self, ticker, current_price):
        self.total_value += self.holdings_info[ticker]["shares"] * current_price