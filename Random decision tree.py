#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import datetime

# machine learning packages
from sklearn.ensemble import ExtraTreesRegressor

# measure results
from sklearn.metrics import mean_squared_error


""" Collecting our data from Yahoo website """

df_CAC40 = web.DataReader('^FCHI', 'yahoo', datetime.datetime(2009, 1, 1), datetime.datetime(2020, 1, 1))\
              .rename(columns={'Adj Close': 'Adj_Close'})

""" Decision tree :
    Basically use a random decision tree to get predictions """

# Sorting and creating a dataframe with the target variable
df_CAC40 = df_CAC40.sort_index()
data = df_CAC40.copy()

# Crating features
data['1d_MinMax'] = data.High - data.Low
data['1d_diff'] = data.Open - data.Close
data['returns'] = data.Close.pct_change()
data['change'] = data.Close.diff()
data['gain'] = data.change.mask(data.change < 0, 0.0)
data['loss'] = -data.change.mask(data.change > 0, -0.0)
data['3d_pct'] = data.Close.pct_change(3)
for depth in [14, 18, 30]:
    data['ma' + str(depth)] = data.Close.rolling(depth).mean()
    data['rsi' + str(depth)] = 100 - 100/(1 + data.gain.rolling(depth).mean()/data.loss.rolling(depth).mean())

# Target values :
data['target'] = data['change'].shift(-1)

# Drop all na values
data.dropna(axis=0, inplace=True)

list_features = ['High', 'Low', 'Open', 'Close', 'Adj_Close', 'Volume', '1d_MinMax', '1d_diff', 'returns', 'change',
                 'gain', 'loss', '3d_pct', 'ma14', 'rsi14', 'ma18', 'rsi18', 'ma30', 'rsi30']

# Best features for the random tree regressor
features = ['1d_MinMax', '1d_diff', 'gain', 'loss', 'rsi14', 'ma18', 'rsi18', 'ma30', 'rsi30', 'target']
df = data[features].copy()

# Train-test split
valid_rows = int(data.shape[0] * 0.1)
X_train, y_train = df[:-2 * valid_rows].loc[:, df.columns != 'target'], df.target[:-2 * valid_rows]
X_valid, y_valid = df[-2 * valid_rows:-valid_rows].loc[:, df.columns != 'target'], \
                      df.target[-2 * valid_rows:-valid_rows]
X_test, y_test = df[-valid_rows:].loc[:, df.columns != 'target'], df.target[-valid_rows:]


# Define the model
model = ExtraTreesRegressor(max_depth=30, random_state=0)

# Train the model using the training sets
model.fit(X_train, y_train)

# Get predictions
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)

# Get accuracy rates
prediction = pd.DataFrame(y_pred_test, index=y_test.index, columns=['change'])
results = pd.concat([prediction, y_test], axis=1)

accuracy = results.apply(lambda row: row.change * row.target >= 0, axis=1).value_counts()[True]
nbr_growth = results.apply(lambda row: row.target >= 0, axis=1).value_counts()[True]
nbr_buy = results.apply(lambda row: row.change >= 0, axis=1).value_counts()[True]
profit = results.apply(lambda row: (row.change >= 0) and (row.target >= 0), axis=1).value_counts()[True]
nbr_sell = results.apply(lambda row: row.change < 0, axis=1).value_counts()[True]
leave = results.apply(lambda row: (row.change < 0) and (row.target < 0), axis=1).value_counts()[True]

# Results
# Errors
print('Accuracy of the model\n')
print('RSME error valid data : ', np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
print('RSME error test data : ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

print('\nWell predicted the next variation : {:.2f} %'.format(100 * accuracy / valid_rows))
print('Accuracy when we buy : {:.2f} %'.format(100 * profit / nbr_buy))
print('Accuracy when we leave : {:.2f} %'.format(100 * leave / nbr_sell))
print('Lost opportunities : {:.2f} %'.format(100 * (1 - profit / nbr_growth)))

# Graphs
plt.clf()
plt.figure(figsize=(11, 7))
plt.title('Random decision tree Regressor')
plt.scatter(y_valid, y_pred_valid, label='valid', color='r')
plt.scatter(y_test, y_pred_test, label='test', color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

