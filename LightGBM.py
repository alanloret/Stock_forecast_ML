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
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor

# measure results
from sklearn.metrics import mean_squared_error, r2_score


def light_gbm(data_split_scaled,data_split_unscaled,data,data_scaled ,i=1): 
    dtrain = lgb.Dataset(data_split_scaled[0], label= data_split_scaled[1])

    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'rmse', 'seed': 7}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=10, verbose_eval=False)

    est_scaled = bst.predict(data_split_scaled[i+1])
    data[i]['est_scaled'] = est_scaled
    data[i]['est'] = data[i]['est_scaled'] * data[i]['adj_close_std'] + data[i]['adj_close_mean']
    

# Calculate RMSE
    rmse_bef_tuning = np.sqrt(mean_squared_error(data_split_unscaled[2*i +1], data[i]['est']))
    #print("RMSE on dev set = %0.3f" % rmse_bef_tuning)

# Calculate MAPE
    mape_bef_tuning = xgb.get_mape(data_split_unscaled[2*i +1], data[i]['est'])
    print('Variance score test data: ', r2_score(data_split_unscaled[2*i + 1], data[i]['est']))
    #print("MAPE on dev set = %0.3f%%" % mape_bef_tuning)
    return rmse_bef_tuning,mape_bef_tuning

