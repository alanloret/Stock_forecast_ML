#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import datetime

""" This part is about plotting some graphs on the index, the returns and the variance / covariance matrix 
    to have the correlations between the features we will use in this projet. """


""" Collecting our data from Yahoo website """

df_CAC40 = web.DataReader('^FCHI', 'yahoo', datetime.datetime(2009, 1, 1), datetime.date.today())\
              .rename(columns={'Adj Close': 'Adj_Close'})
df_APPLE = web.DataReader('AAPL', 'yahoo', datetime.datetime(2009, 1, 1), datetime.date.today())\
              .rename(columns={'Adj Close': 'Adj_Close'})

#print(df_CAC40.columns)
#print(df_CAC40.shape)

# Plots
df_CAC40.Close.plot(title='Share ^FCHI : 2006-01-01 / ' + str(datetime.date.today()), x='Date', y='Index',
                    figsize=(15, 9))
plt.show()
df_APPLE.Close.plot(title='Share APPL : 2006-01-01 / ' + str(datetime.date.today()), x='Date', y='Index',
                    figsize=(15, 9))
plt.show()


# Sorting and creating a dataframe with the target variable
df_CAC40 = df_CAC40.sort_index()
data = df_CAC40.copy()


# Histogram of the daily price change percent of 'Last' price
data['Close'].pct_change().plot.hist(bins=50)
plt.title('Daily Price: 1-Day Percent Change')
plt.show()

# Crating features
data['1d_MinMax'] = data.High - data.Low
data['1d_diff'] = data.Open - data.Close
data['returns'] = data.Close.pct_change()
data['change'] = data.Close.diff()
data['gain'] = data.change.mask(data.change < 0, 0.0)
data['loss'] = -data.change.mask(data.change > 0, -0.0)
data['3d_pct'] = data.Close.pct_change(3)
data['3d_future_pct'] = data.Close.shift(-3).pct_change(3)
for depth in [15, 20, 30]:
    data['ma' + str(depth)] = data.Close.rolling(depth).mean()
    data['rsi' + str(depth)] = 100 - 100/(1 + data.gain.rolling(depth).mean()/data.loss.rolling(depth).mean())

# Target values :
data['target'] = data['Close'].shift(-1)

# Drop all na values
data.dropna(axis=0, inplace=True)


""" Few results interesting for our project """

list_features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj_Close', '1d_MinMax', '1d_diff', 'returns', 'change',
                 'gain', 'loss', '3d_pct', '3d_future_pct', 'ma15', 'rsi15', 'ma20', 'rsi20', 'ma30', 'rsi30']

# Calculate the correlation matrix between the 5d close percentage changes (current and future)
corr = data[list_features].corr()

# As we can seen there is usually no or a tiny correlation between previous and upcoming pct_changes
plt.clf()
plt.figure(figsize=(7, 7))
plt.scatter(data['3d_pct'], data['3d_future_pct'])
plt.title('Current vs. Future 3-Day % Change')
plt.show()

plt.clf()
plt.figure(figsize=(13, 13))
sns.heatmap(corr, annot=True)
plt.show()



