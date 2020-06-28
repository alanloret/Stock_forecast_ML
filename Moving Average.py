#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import datetime

# measure results
from sklearn.metrics import r2_score, mean_squared_error


""" Collecting our data from Yahoo website """

df_CAC40 = web.DataReader('^FCHI', 'yahoo', datetime.datetime(2009, 1, 1), datetime.datetime(2020, 1, 1)) \
    .rename(columns={'Adj Close': 'Adj_Close'})


"""  Moving average strategy  :
    The goal here is to predict the share's value with the mean of the past days. """

# Sorting and creating a dataframe with the target variable
df_CAC40 = df_CAC40.sort_index()
features = ['Close']
data = df_CAC40[features]


# Train-test split (There is no training nor validation data because there are not necessary)
test_rows = int(data.shape[0] * 0.1)
y_valid = data.Close[-2 * test_rows:-test_rows].copy()
y_test = data.Close[-test_rows:].copy()


# Define the model
def moving_average(df, depth, num_rows):
    """ This function predicts the share's value for the 'num_rows' upcoming days :
        it's calculating the average of the previous 'depth' values."""

    predictions = pd.DataFrame(df[-num_rows - depth:].rolling(depth).mean())\
                    .rename(columns={'Close': 'target'})
    predictions = predictions.shift()
    predictions.dropna(axis=0, inplace=True)

    return predictions


# Get predictions
y_pred_valid = moving_average(data, 1, 2*test_rows).target[:test_rows]
y_pred_test = moving_average(data, 1, test_rows).target


# Results
# Calculate errors
print('RMSE error valid data : ', np.sqrt(mean_squared_error(y_pred_valid, y_valid)))
print('RMSE error test data : ', np.sqrt(mean_squared_error(y_pred_test, y_test)))

# Graphs
nbr_SMA = 200
rmse_rate = [np.sqrt(mean_squared_error(moving_average(data, i, test_rows).target, y_test)) for i in range(1, nbr_SMA)]
variance_rate = [r2_score(moving_average(data, i, test_rows).target, y_test) for i in range(1, nbr_SMA)]

plt.clf()
plt.plot([i for i in range(1, nbr_SMA)], rmse_rate, color='b')
plt.title('RMSE of Moving Average models')
plt.xlabel('MA depth')
plt.ylabel('RMSE')
plt.show()

plt.clf()
plt.plot([i for i in range(1, nbr_SMA)], variance_rate, color='b')
plt.title('R2 score of Moving Average models')
plt.xlabel('MA depth')
plt.ylabel('Variance')
plt.show()

plt.clf()
plt.figure(figsize=(11, 7))
plt.title('Moving Average')
plt.scatter(y_valid, y_pred_valid, label='valid', color='r')
plt.scatter(y_test, y_pred_test, label='test', color='blue')
plt.plot(y_test, y_test, color='black', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
