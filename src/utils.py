def buyin_shares(money, price, fee=0.001):
    cost = price * (1 + fee)
    shares = money // cost
    return shares, cost

def sell_shares(shares, price, fee=0.001):
    revenue = shares * price * (1 - fee)
    return revenue


class Portfolio:
    def __init__(self, cash=1000000):
        self.tickers = []
        self.holdings_info = {}
        self.cash = cash
        self.trade_history = []
        self.total_value = cash
    
    def step(self, ticker, current_price):
        self.update_returns(ticker, current_price)
        self.update_total_value(ticker, current_price)


    def update_returns(self, ticker, current_price):
        if ticker in self.tickers and self.holdings_info[ticker]["shares"] > 0:
            shares = self.holdings_info[ticker]["shares"]
            average_price = self.holdings_info[ticker]["average_price"]
            self.holdings_info[ticker]["returns"] = (current_price - average_price) * shares

    def update_cash(self, amount):
        self.cash += amount
    
    def update_holdings(self, ticker, shares, price):
        # note that the price here does not include fees
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.holdings_info[ticker] = {"shares": shares,
                                          "total_cost": shares * price,
                                          "average_price": price,
                                          "returns": 0.0}
        else:
            self.holdings_info[ticker]["shares"] += shares
            if self.holdings_info[ticker]["shares"] != 0:
                self.holdings_info[ticker]["total_cost"] += price * shares
                self.holdings_info[ticker]["average_price"] = self.holdings_info[ticker]["total_cost"] / self.holdings_info[ticker]["shares"]
            else:
                self.holdings_info[ticker]["total_cost"] = 0.0
                self.holdings_info[ticker]["average_price"] = 0.0
                self.holdings_info[ticker]["returns"] = 0.0
    
    def update_trade_history(self, date, action, ticker, shares, price):
        self.trade_history.append({"date": date,
                                   "action": action,
                                   "ticker": ticker,
                                   "shares": shares,
                                   "price": price})
    
    def update_total_value(self, ticker, current_price):
        self.total_value = self.cash
        self.total_value = self.cash + shares * current_price