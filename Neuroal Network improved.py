#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
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
    We will build an artificial neural network based on LSTM and use 13 features and the last 5 days 
    to predict the change of the next day """

# Sorting and creating a dataframe with the target variable
df_CAC40 = df_CAC40.sort_index()
data = df_CAC40.copy()

# Crating features
data['1d_MinMax'] = data.High - data.Low
data['1d_diff'] = data.Open - data.Close
data['returns'] = data.Close.pct_change()
data['change'] = data.Close.diff()
data['gain'] = data.change.mask(data.change < 0, 0.0)
data['loss'] = - data.change.mask(data.change > 0, -0.0)
data['3d_pct'] = data.Close.pct_change(3)
for depth in [12, 24]:
    data['ma' + str(depth)] = data.Close.rolling(depth).mean()
for depth in [14, 20, 30]:
    data['rsi' + str(depth)] = 100 - 100/(1 + data.gain.rolling(depth).mean()/data.loss.rolling(depth).mean())

# Target values :
data['target'] = data['change'].shift(-1)

# Drop all na values
data.dropna(axis=0, inplace=True)

list_features = ['High', 'Low', 'Open', 'Close', 'Adj_Close', 'Volume', '1d_MinMax', '1d_diff', 'returns', 'change',
                 'gain', 'loss', '3d_pct', 'ma12', 'ma24', 'rsi14', 'rsi20', 'rsi30']

# Select the features we want to use
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'returns', 'change', 'ma12', 'ma24',
            'rsi14', 'rsi20', 'rsi30', 'target']
df = data[features].copy()
depth = 5

# Select our data for train/valid/test
valid_rows = 200
train = df[:-2 * valid_rows]
valid = df[-2 * valid_rows - depth:-valid_rows]
test = df[-valid_rows - depth:]

# Scale our data : the scale is per column
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
valid = scaler.transform(valid)
test = scaler.transform(test)

# Train-test split
X_train = np.array([train[i+1-depth:i+1, :-1] for i in range(depth, len(train))])
X_valid = np.array([valid[i+1-depth:i+1, :-1] for i in range(depth, len(valid))])
X_test = np.array([test[i+1-depth:i+1, :-1] for i in range(depth, len(test))])

y_train = train[depth:, -1].reshape((train.shape[0]-depth, 1))
y_valid = valid[depth:, -1].reshape((valid.shape[0]-depth, 1))
y_test = test[depth:, -1].reshape((test.shape[0]-depth, 1))
# y_test = df.target[-valid_rows:].values.reshape((test.shape[0]-depth, 1))

# Define our model
model = Sequential()
model.add(LSTM(300, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(Dropout(0.5))
model.add(LSTM(200, return_sequences=False, input_shape=X_train.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1))

# Compile and fit our model
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_valid, y_valid), shuffle=False)

# Get predictions
y_pred_valid = model.predict(X_valid)
y_pred_test = model.predict(X_test)

# Unscale our data
y_pred_valid = np.concatenate((valid[depth:, :-1], y_pred_valid), axis=1)
y_pred_test = np.concatenate((test[depth:, :-1], y_pred_test), axis=1)
y_valid = np.concatenate((valid[depth:, :-1], y_valid), axis=1)
y_test = np.concatenate((test[depth:, :-1], y_test), axis=1)

y_pred_valid = scaler.inverse_transform(y_pred_valid)[:, -1]
y_pred_test = scaler.inverse_transform(y_pred_test)[:, -1]
y_valid = scaler.inverse_transform(y_valid)[:, -1]
y_test = scaler.inverse_transform(y_test)[:, -1]

# Results
# Errors
print('RSME error valid data : ', np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
print('RSME error test data : ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Graphs
plt.clf()
plt.figure(figsize=(11, 7))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title('Neural Network improved loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.clf()
plt.figure(figsize=(11, 7))
plt.title('Neural Network improved')
plt.scatter(y_valid, y_pred_valid, label='valid', color='r')
plt.scatter(y_test, y_pred_test, label='test', color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()



