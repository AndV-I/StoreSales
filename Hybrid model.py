# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:06:42 2021

@author: Andrei
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

store_sales = pd.read_csv('train.csv',  
                         usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                         dtype={'store_nbr': 'category', 'family': 'category', 'sales': 'float32',}, 
                         parse_dates=['date'], 
                         infer_datetime_format=True)

family_sales = (store_sales.groupby(['family', 'date']).mean().unstack('family').loc['2017'])
y = family_sales.loc[:, 'sales']

# Class Model definition
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
    
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns)
       
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() 
        self.model_2.fit(X_2, y_resid)
    
        self.y_columns = y.columns

    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns)
        y_pred = y_pred.stack().squeeze()  #
        y_pred += self.model_2.predict(X_2)
        
        return y_pred.unstack()

# X1 Features for Time Series
dp = DeterministicProcess(index=y.index, order=1)
X1 = dp.in_sample()

# X2 Features for Regression
X2 = family_sales.drop('sales', axis=1).stack()  # onpromotion feature
le = LabelEncoder()  
X2 = X2.reset_index('family')
X2['family'] = le.fit_transform(X2['family'])
X2["day"] = X2.index.day

# Train boosted hybrid
model = BoostedHybrid(model_1=Ridge(), model_2=KNeighborsRegressor())
model.fit(X1, X2, y)
y_pred = model.predict(X1, X2)
y_pred = y_pred.clip(0.0)

# Metrics
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
print(f'Model R2 score:{r2_score(y, y_pred): .2f}')
MSE = mean_squared_error(y, y_pred)
print(f"Model RMSE score:{np.sqrt(MSE) : .3f}")

# Plots
y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X1[: "2017-07-01"], X1["2017-07-02" :]
X2_train, X2_valid = X2.loc[:"2017-07-01"], X2.loc["2017-07-02":]

model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), alpha=0.5)
_ = y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)