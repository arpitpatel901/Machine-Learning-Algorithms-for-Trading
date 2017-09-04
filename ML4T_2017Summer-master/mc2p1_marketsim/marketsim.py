"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    ##build the date framr
    df = pd.read_csv(orders_file, index_col='Date',parse_dates=True, na_values=['nan']) ## Read the data
    df = df.sort_index() ## sort dates in increasing order
    start_date = df.index[0]
    end_date = df.index[-1]
    symbols = df['Symbol'].unique().tolist()  ## get a list of all symbols
    prices_all = get_data(symbols, pd.date_range(start_date, end_date)) ## get data of those symbols from start to end date
    prices = prices_all[symbols]  # only portfolio symbols
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')

    ##Initialize cash array
    cash = np.zeros(len(prices))
    cash[0] = start_val
    stockTracker = {}
    for ticker in symbols:
        stockTracker["Val_{0}".format(ticker)] = np.zeros(len(prices))
        stockTracker["Shares_{0}".format(ticker)] = np.zeros(len(prices))

    TransDateIndexes = [] ## initiate temporary index(index of all matching dates)

    for row in df.itertuples():## for each row in transaction csv
        TransDate = row[0] ## time stamp for that row
        TransSymb = row[1] ## symbol for that transaction 'GOOG'
        TransSum = 0
        absTransSum = 0
        NumOfShares = np.zeros(len(symbols)) ## track number of stock of each stock tickerer('GOOG',...) at the end/start of each transaction
        TransDateIndexes.append(list(np.where(prices.index == TransDate)[0])) ## match transaction date to date in the dataframe and get index
        if row[2] == 'BUY':
            buyCost = 1.005 * prices[TransSymb][TransDateIndexes[-1]].values * row[3] + 9.95
            for i, ticker in enumerate(symbols):
                if ticker == TransSymb: ## if the symbol matches
                    NumOfShares[i] = (stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1]] + row[3]) * prices[ticker][TransDateIndexes[-1]]
                else: ## if symbol doesnt match
                    NumOfShares[i] = stockTracker["Shares_{0}".format(ticker)][TransDateIndexes[-1]] * prices[ticker][TransDateIndexes[-1]] ## price of 'GOOG(as per df)
                TransSum = TransSum + NumOfShares[i]
                absTransSum = absTransSum + abs(NumOfShares[i])
            if TransDateIndexes[-1][0] == 0: ## if date matches
                leverage = absTransSum / (TransSum + cash[TransDateIndexes[-1]] - buyCost)
            else: ## if date doesnt match
                leverage = absTransSum / (TransSum + cash[[TransDateIndexes[-2]]] - buyCost)
            if leverage < 2:
                if TransDateIndexes[-1][0] == 0:
                    cash[TransDateIndexes[-1][0]:] = cash[TransDateIndexes[-1]] - buyCost
                    stockTracker["Shares_{0}".format(TransSymb)][:] = stockTracker["Shares_{0}".format(TransSymb)][0] + row[3]
                else:
                    cash[TransDateIndexes[-1][0]:] = cash[TransDateIndexes[-2]] - buyCost
                    stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1][0]:] = stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1]] + row[3]

        else:  # SELL
            sellrev = .995 * prices[TransSymb][TransDateIndexes[-1]].values * row[3] - 9.95
            for i, ticker in enumerate(symbols):
                if ticker == TransSymb:
                    NumOfShares[i] = (stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1]] - row[3]) * prices[ticker][TransDateIndexes[-1]]
                else:
                    NumOfShares[i] = stockTracker["Shares_{0}".format(ticker)][TransDateIndexes[-1]] * prices[ticker][TransDateIndexes[-1]]
                TransSum += NumOfShares[i]
                absTransSum += abs(NumOfShares[i])
            if TransDateIndexes[-1][0] == 0:
                leverage = absTransSum / (TransSum + cash[TransDateIndexes[-1]] + sellrev)
            else:
                leverage = absTransSum / (TransSum + cash[[TransDateIndexes[-2]]] + sellrev)
            if leverage < 2:
                if TransDateIndexes[-1][0] == 0:
                    cash[TransDateIndexes[-1][0]:] = cash[0] + sellrev
                    stockTracker["Shares_{0}".format(TransSymb)][:] = stockTracker["Shares_{0}".format(TransSymb)][0] - row[3]
                else:
                    cash[TransDateIndexes[-1][0]:] = cash[TransDateIndexes[-2]] + sellrev
                    stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1][0]:] = stockTracker["Shares_{0}".format(TransSymb)][TransDateIndexes[-1]] - row[3]
    portvals = pd.DataFrame(index=prices.index)
    portval = np.zeros(len(prices))
    for x in range(len(prices)):
        for ticker in symbols:
            stockTracker["Val_{0}".format(ticker)][x] = stockTracker["Shares_{0}".format(ticker)][x] * prices[ticker][x]
            portval[x] += stockTracker["Val_{0}".format(ticker)][x]
    portvals["portvals"] = (portval + cash).tolist()
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

def author():
    return 'apatel380'


if __name__ == "__main__":
    test_code()
