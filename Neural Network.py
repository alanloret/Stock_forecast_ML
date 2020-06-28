#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as web
import datetime

# machine learning packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# measure results
from sklearn.metrics import mean_squared_error


""" Collecting our data from Yahoo website """

df_CAC40 = web.DataReader('^FCHI', 'yahoo', datetime.datetime(2000, 1, 1), datetime.datetime(2020, 1, 1))\
              .rename(columns={'Adj Close': 'Adj_Close'})


""" Neural Network model :
    We will build an artificial neural network based on LSTM but we will only use the previous change
    of the 60 days as feature to predict the change of the next day """

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

# Select the features we want to use
features = ['target']
df = data[features].copy()
depth = 60

# Select our data for train/valid/test
valid_rows = 277  # 277 days just to compare with the same test sample of other models
train = df[:-2 * valid_rows].values
valid = df[-2 * valid_rows - depth:-valid_rows].values
test = df[-valid_rows - depth:].values

# Scale our data : the scale is per column
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
valid = scaler.transform(valid)
test = scaler.transform(test)

# Train-test split
X_train = np.array([train[i-depth:i, 0] for i in range(depth, len(train))])
y_train = np.array([train[i] for i in range(depth, len(train))])
X_valid = np.array([valid[i-depth:i, 0] for i in range(depth, len(valid))])
y_valid = np.array([valid[i] for i in range(depth, len(valid))])
X_test = np.array([test[i-depth:i, 0] for i in range(depth, len(test))])
y_test = np.array([test[i] for i in range(depth, len(test))])

# Reshape our model in 3 dimensions for the LSTM neural network
X_train = X_train.reshape((*X_train.shape, 1))
X_valid = X_valid.reshape((*X_valid.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

# Define the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and optimize
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid), verbose=2,
                    callbacks=[], shuffle=False)

# Get predictions
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)

y_pred_valid = scaler.inverse_transform(y_pred_valid)
y_pred_test = scaler.inverse_transform(y_pred_test)
y_valid = scaler.inverse_transform(y_valid)
y_test = scaler.inverse_transform(y_test)

# Get accuracy rates
prediction = pd.DataFrame(y_pred_test, index=df[-valid_rows:].index, columns=['change'])
results = pd.concat([prediction, df[-valid_rows:]], axis=1)

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
plt.figure(figsize=(11, 8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title('Neural Network loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.clf()
plt.figure(figsize=(11, 7))
plt.title('Neural Network')
plt.scatter(y_valid, y_pred_valid, label='valid', color='r')
plt.scatter(y_test, y_pred_test, label='test', color='blue')
plt.plot([0, 0], [-5, 5], color='black')
plt.plot(y_test, [0] * len(y_test), color='black')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()



