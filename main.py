import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import statistics

# Given a m x n matrix  of m closing prices for n equities, returns an 1 x n 
# array of average closing prices
def compute_average(closing_prices_for_multiple_equities):
    sums = [0] * len(closing_prices_for_multiple_equities[0])
    for row in closing_prices_for_multiple_equities:
        for i in range(len(row)):
            sums[i] += row[i]

    avgs = [x/len(closing_prices_for_multiple_equities) for x in sums]
    return avgs

# Given an m x n matrix of m closing prices for n equities and n averages for n 
# equities, returns all of the difference of the closing prices and their averages
# essentially substracts mean to make the new mean of the data set 0 for each column
def compute_stocks_demeaned(closing_prices_matrix, average_closing_prices):
  m = len(closing_prices_matrix)
  n = len(closing_prices_matrix[0])
  demeaned = [[0 for i in range(n)] for j in range(m)]
  for i in range(len(closing_prices_matrix)):
    for j in range(len(closing_prices_matrix[i])):
      demeaned[i][j] = closing_prices_matrix[i][j] - average_closing_prices[j]

  return demeaned

# Given an m x n matrix of the demeaned closing prices for n equities, returns
# the associated covariance matrix calculated by demeaned transpose x demeaned.
def compute_covariance_matrix(demeaned):
    s_minus_m = np.array(demeaned)
    s_minus_m_t = s_minus_m.transpose()
    return np.matmul(s_minus_m_t, s_minus_m) / len(demeaned)
    

def estimated_portfolio_risk_based_on_stddev (asset_weights, covariance_matrix):
    Wt = np.asmatrix(asset_weights)
    W = Wt.transpose()
    return np.sqrt(Wt.dot(covariance_matrix).dot(W))


def expected_portfolio_return(average_closing_prices, asset_weights):
    M = np.array(average_closing_prices)
    W = np.array(asset_weights)
    return M.dot(W)


def sharpe_ratio(expected_return, portfolio_std_dev, risk_free_rate):
    return (expected_return - risk_free_rate) / portfolio_std_dev
    

# Given array of equities, returning dataframe with historical data dating back 
def get_closing_price_historical_data(equities, period, interval):
    data = {}
    for x in equities:
        ticker = yf.Ticker(x)
        history = ticker.history(period= period, interval= interval)['Close']
        data[x] = history
    return data

def __main__():
    # Gather Portfolio Data
    trimmed_senbet = pd.read_excel('Senbet Portfolio.xlsx')

    # Beautify Data and the df
    trimmed_senbet.columns = ['Ticker', 'Allocation', 'Shares']
    trimmed_senbet['Allocation'] *= 100
    # Certain Equities had .O .K and extra spaces, applied this lambda to clean up the ticker symbols
    trimmed_senbet['Ticker'] = trimmed_senbet.apply(lambda x: x['Ticker'].replace('.O', '').replace('.K', '').replace(' ', ''), axis = 1)
    asset_weights = list(trimmed_senbet['Allocation'] / 100)
    
    # Gather historical closing price data
    historical_data = get_closing_price_historical_data(trimmed_senbet['Ticker'], '179d', '1d')

    return 0

if __name__ == "__main__":
    __main__()