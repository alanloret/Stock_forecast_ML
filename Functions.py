#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:05:16 2020

@authors: alanloret
"""

# import packages
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import datetime

# machine learning packages
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# measure results
from sklearn.metrics import mean_squared_error

""" Useful functions """


class PossibleDataLeakages(Exception):
    """ This error class is raised when you input a date that may lead to data leakages :
        when you want to make predictions on the training data with a start date earlier
        than the end date of training dataset """
    pass


def my_data(start, end, share, days=0):
    """ Collecting our data from Yahoo website and returns a dataframe
        with the data from the start date to the end date.
        Note : here we collect our date futher than the input dates because due to moving average and rsi features
        if we don't do this NaN values will appear in our data. Moreover we have to be careful with the closing days."""

    data = web.DataReader(share, 'yahoo', datetime.datetime(*start)-datetime.timedelta(days=days),
                          datetime.datetime(*end)+datetime.timedelta(days=4)).rename(columns={'Adj Close': 'Adj_Close'})

    return data


def collect_data(start, end, *shares, **kwargs):
    """ Collecting our data and creating the features we want.
        Note :
        start -> tuple with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> you can input * 'depths' for different depths for moving average and rsi features """

    stocks = {}
    depths = kwargs.get('depths', [14, 18, 20, 30])

    for share in shares:
        df_share = my_data(start, end, share, 2 * max(depths) + 1)
        df_share = df_share.sort_index()
        data = df_share.copy()

        # Crating our features
        data['1d_MinMax'] = data.High - data.Low
        data['1d_diff'] = data.Open - data.Close
        data['returns'] = data.Close.pct_change()
        data['change'] = data.Close.diff()
        data['gain'] = data.change.mask(data.change < 0, 0.0)
        data['loss'] = -data.change.mask(data.change > 0, -0.0)
        data['3d_pct'] = data.Close.pct_change(3)
        for depth in depths:
            data['ma' + str(depth)] = data.Close.rolling(depth).mean()
            data['rsi' + str(depth)] = 100 - 100 / (
                        1 + data.gain.rolling(depth).mean() / data.loss.rolling(depth).mean())

        # Target
        data['target'] = data['change'].shift(-1)

        # Drop all NaN values and select the useful data
        data.dropna(axis=0, inplace=True)
        data = data[datetime.datetime(*start) <= data.index]
        stocks[share] = data[datetime.datetime(*end) >= data.index].copy()

    return stocks


def moving_average(start, end, *shares, **kwargs):
    """ This function uses moving average to predict the share's value for the date you've input
        Notes :
        start/end -> tuples with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> you can input * 'depths' number of days for the moving average """

    predictions = {}

    depths = kwargs.get('depths', [14, 18, 30])
    features = ['ma' + str(depth) for depth in depths]

    for stock, df_stock in collect_data(start, end, *shares, **kwargs).items():
        df = df_stock[features].copy()
        for feature in features:
            predictions[stock + '_' + feature] = pd.DataFrame(df[feature], index=df[feature].index) \
                .rename(columns={feature: 'target'})

    return predictions


def linear_regression(start, end, *shares, **kwargs):
    """ Does a Linear Regression of the stocks and predicts the share's value for the date you've input
        Notes :
        start/end -> tuples with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> you can input * 'features' for the training data
                                * 'depths' for different depths for moving average and rsi features
                                * 'training_share' to select the same share to train all the models """

    predictions = {}

    depths = kwargs.get('depths', [14, 18, 30])

    # Select the best features for the linear regression
    best_features = ['Low', 'Close', 'change', 'returns', 'gain', 'loss'] + ['rsi' + str(d) for d in depths] + \
                    ['target']
    features = kwargs.get('features', best_features)

    for stock, df_stock in collect_data(start, end, *shares, **kwargs).items():
        training_share = kwargs.get('training_share', stock)

        if datetime.datetime(2017, 10, 26) > datetime.datetime(*start) and stock == training_share:
            raise PossibleDataLeakages('Your start date prediction is before the end date of the training data'
                                       'for the training share')

        # Train data
        train = collect_data((2009, 1, 1), (2017, 10, 26), training_share, **kwargs)[training_share]
        train_data = train[features].copy()
        X_train, y_train = train_data.loc[:, train_data.columns != 'target'], train_data.target

        # Define the model
        model = LinearRegression()

        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Test data
        df = df_stock[features].copy()
        X_test, y_test = df.loc[:, df.columns != 'target'], df.target

        # Make predictions
        y_pred_test = model.predict(X_test)
        y_pred_test = pd.Series(y_pred_test, index=y_test.index)

        predictions[stock] = pd.DataFrame(y_pred_test + df_stock.Close, columns=['target'])
        predictions[stock + '_mse'] = np.sqrt(mean_squared_error(y_pred_test, y_test))

    return predictions


def random_decision_tree(start, end, *shares, **kwargs):
    """ Creates a random tree regressor on the stocks to predict the share's value for the date you've input
        Notes :
        start/end -> tuples with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> you can input * 'features' for the training data,
                                * 'depths' for different depths for moving average and rsi features
                                * 'training_share' to select the share you want to train with """

    predictions = {}

    depths = kwargs.get('depths', [14, 18, 30])

    # Select the best features for the random tree regressor
    best_features = ['ma' + str(d) for d in depths[1:]] + ['rsi' + str(d) for d in depths] + \
                    ['1d_MinMax', '1d_diff', 'gain', 'loss', 'returns', 'target']

    features = kwargs.get('features', best_features)

    for stock, df_stock in collect_data(start, end, *shares, **kwargs).items():
        training_share = kwargs.get('training_share', stock)

        if datetime.datetime(2017, 10, 26) > datetime.datetime(*start) and stock == training_share:
            raise PossibleDataLeakages('Your start date prediction is before the end date of the training data '
                                       'for the training share')

        # Train data
        train = collect_data((2009, 1, 1), (2017, 10, 26), training_share, **kwargs)[training_share]
        train_data = train[features].copy()
        X_train, y_train = train_data.loc[:, train_data.columns != 'target'], train_data.target

        # Define the model
        model = ExtraTreesRegressor(max_depth=9, random_state=0)

        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Test data
        df = df_stock[features].copy()
        X_test, y_test = df.loc[:, df.columns != 'target'], df.target

        # Make predictions
        y_pred_test = model.predict(X_test)
        y_pred_test = pd.Series(y_pred_test, index=y_test.index)

        predictions[stock] = pd.DataFrame(y_pred_test + df_stock.Close, columns=['target'])
        predictions[stock + '_mse'] = np.sqrt(mean_squared_error(y_pred_test, y_test))

    return predictions


def k_nearest_neighbors(start, end, *shares, **kwargs):
    """ Uses the K nearest neighbors on the stocks to predict the share's value for the date you've input
            Notes :
            start/end -> tuples with (yyyy, mm, dd)
            shares -> strings made of shares name
            kwargs -> you can input * 'features' for the training data,
                                    * 'depths' for different depths for moving average and rsi features
                                    * 'training_share' to select the share you want to train with
                                    * 'neighbors' to select the number of neighbors """

    predictions = {}
    depths = kwargs.get('depths', [20])

    # Select the best features for the K_nearest_neighbors
    best_features = ['High', 'Low', 'Open', 'Close', '1d_diff', 'change']+['rsi' + str(d) for d in depths]+['target']

    features = kwargs.get('features', best_features)

    neighbors = kwargs.get('neighbors', 105)

    for stock, df_stock in collect_data(start, end, *shares, **kwargs).items():
        training_share = kwargs.get('training_share', stock)

        if datetime.datetime(2017, 10, 26) > datetime.datetime(*start) and stock == training_share:
            raise PossibleDataLeakages('Your start date prediction is before the end date of the training data'
                                       'for the training share')

        # Train data
        train = collect_data((2009, 1, 1), (2017, 10, 26), training_share, **kwargs)[training_share]
        train_data = train[features].copy()
        X_train, y_train = train_data.loc[:, train_data.columns != 'target'], train_data.target

        df = df_stock[features].copy()
        X_test, y_test = df.loc[:, df.columns != 'target'], df.target

        # Scale our data
        mean, std = y_train.mean(), y_train.std()
        y_train_scaled = (y_train - mean) / std

        # Modify our data to apply k nearest neighbors : the target can not be a float
        encoder = LabelEncoder()
        y_train_enc = (pd.DataFrame(encoder.fit_transform(y_train_scaled))).values.ravel()

        # Define the model
        model = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')

        # Train the model using the training sets
        model.fit(X_train, y_train_enc)

        # Make predictions
        y_pred_test_enc = model.predict(X_test)
        y_pred_test_enc = pd.Series(y_pred_test_enc, index=y_test.index)

        # Revert the encoding and unscale the prediction
        y_pred_test_scaled = encoder.inverse_transform(y_pred_test_enc)
        y_pred_test = (y_pred_test_scaled * std) + mean

        predictions[stock] = pd.DataFrame(y_pred_test + df_stock.Close, columns=['target'])
        predictions[stock + '_mse'] = np.sqrt(mean_squared_error(y_pred_test, y_test))

    return predictions


def neural_data(start, end, *shares, **kwargs):
    """ Collecting our data and creating the features we want.
        Note :
        start -> tuple with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> just to make sur we get the previous 60 predictions"""

    stocks = {}
    memory = kwargs.get('memory', 60)

    for share in shares:
        df_share = my_data(start, end, share, 2 * (memory + 1))  # we go further and will return only the predictions
                                                                 # we asked for
        df_share = df_share.sort_index()
        data = df_share.copy()

        # Target : we just use the Close.diff of the previous memory days to predict the next day Close.diff
        df = pd.DataFrame(data.Close.diff().shift(-1)).rename(columns={'Close': 'target'})
        df['Close'] = data.Close

        # Drop all NaN values and select the useful data
        df.dropna(axis=0, inplace=True)
        stocks[share] = df[datetime.datetime(*end) >= df.index].copy()

    return stocks


def neural_network(start, end, *shares, **kwargs):
    """ Uses a neural network based on LSTM to predict the share's value for the date you've input
        Notes :
        start/end -> tuples with (yyyy, mm, dd)
        shares -> strings made of shares name
        kwargs -> you can input * 'memory' to select the number of days in the LSTM
                                * 'training_share' to select the share you want to train with
                                * 'training_start' to select the start date of the training
                                * 'epochs' to select the epochs of the model
                                * 'batch_size' to select the batch_size of the model """

    predictions = {}

    features = ['target']
    memory = kwargs.get('memory', 60)
    training_start = kwargs.get('training_start', (2000, 1, 1))
    epochs = kwargs.get('epochs', 100)
    batch_size = kwargs.get('batch_size', 32)

    for stock, df_stock in neural_data(start, end, *shares, **kwargs).items():
        training_share = kwargs.get('training_share', stock)

        if datetime.datetime(2017, 10, 26) > datetime.datetime(*start) and stock == training_share:
            raise PossibleDataLeakages('Your start date prediction is before the end date of the training data '
                                       'for the training share')

        # Train data
        train = neural_data(training_start, (2017, 10, 26), training_share, **kwargs)[training_share]
        train = train.target.values.reshape((train.shape[0], 1))

        # Scale our data : the scale is per column
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train)

        X_train = np.array([train[i - memory:i, 0] for i in range(memory, len(train))])
        y_train = np.array([train[i] for i in range(memory, len(train))])

        # Reshape our model in 3 dimensions for the LSTM neural network
        X_train = X_train.reshape((*X_train.shape, 1))

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
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)

        # Test data
        df = df_stock[features].copy()
        test = df.values
        test = scaler.transform(test)
        X_test = np.array([test[i - memory:i, 0] for i in range(memory, len(test))])
        y_test = np.array([test[i] for i in range(memory, len(test))])
        X_test = X_test.reshape((*X_test.shape, 1))

        # Get predictions
        y_pred_test = model.predict(X_test)
        y_pred_test = scaler.inverse_transform(y_pred_test)
        y_test = scaler.inverse_transform(y_test)

        predictions[stock + '_mse'] = np.sqrt(mean_squared_error(y_pred_test, y_test))

        y_pred_test = np.concatenate([[[0]] * memory, y_pred_test], axis=0)
        prediction = pd.DataFrame(y_pred_test, index=df.index, columns=['preds'])
        pred_index = pd.DataFrame(prediction.preds + df_stock.Close, columns=['target'])
        
        predictions[stock] = pred_index[datetime.datetime(*start) <= pred_index.index]

    return predictions


