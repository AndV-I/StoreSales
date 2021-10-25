# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:09:04 2021

@author: Andrei
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

store_sales = pd.read_csv('train.csv',  
                         usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                         dtype={'store_nbr': 'category', 'family': 'category', 'sales': 'float32',}, 
                         parse_dates=['date'], 
                         infer_datetime_format=True)
holidays_events = pd.read_csv('holidays_events.csv',
                                                     dtype={'type': 'category',
                                                            'locale': 'category',
                                                            'locale_name': 'category',
                                                            'description': 'category',
                                                            'transferred': 'bool'},
                                                     parse_dates=['date'],
                                                     infer_datetime_format=True)
df_test = pd.read_csv('test.csv',
                      dtype={'store_nbr': 'category', 'family': 'category', 'onpromotion': 'uint32'},
                      parse_dates=['date'],
                      infer_datetime_format=True)

holidays_events = holidays_events.set_index('date').to_period('D')
store_sales['date'] = store_sales.date.dt.to_period('D')
df_test['date'] = df_test.date.dt.to_period('D')

df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

y = store_sales.drop('onpromotion', axis=1).unstack(['store_nbr', 'family']).loc["2017"]

holidays = (holidays_events
            .query("locale in ['National', 'Regional']")
            .loc['2017':'2017-08-15', ['description']]
            .assign(description=lambda x: x.description.cat.remove_unused_categories())
)
X_holidays = pd.get_dummies(holidays)

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
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X1 = dp.in_sample()
X1['NewYear'] = (X1.index.dayofyear == 1)
X1 = X1.join(X_holidays, on='date').fillna(0.0)

# X2 Features for Regression
X2 = store_sales.drop('sales', axis=1).unstack(['store_nbr', 'family']).loc["2017"]  
X2 = X2.stack() # onpromotion feature
le = LabelEncoder()  
X2 = X2.reset_index('family')
X2['family'] = le.fit_transform(X2['family'])
X2["day"] = X2.index.day


# Train boosted hybrid
model = BoostedHybrid(model_1=Ridge(), model_2=RandomForestRegressor())
model.fit(X1, X2, y)
y_pred = model.predict(X1, X2)
y_pred = y_pred.clip(0.0)

# Metrics
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
print(f'Model R2 score:{r2_score(y, y_pred): .2f}')
MSE = mean_squared_error(y, y_pred)
print(f"Model RMSE score:{np.sqrt(MSE) : .3f}")

RMSLE = np.sqrt(mean_squared_log_error(y, y_pred))
print(f"Model RMSLE score:{RMSLE : .3f}")# 

# Plots
y.columns
filtr = ('sales', '1', 'AUTOMOTIVE')
ax = y[filtr].plot(alpha=0.5, title="Sales", ylabel="items sold")
ax = y_pred[filtr].plot(ax=ax, label="Prediction")
ax.legend();

# Submission
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)
X_test = X_test.join(X_holidays, on='date').fillna(0.0)

X2_test = df_test.drop('id', axis=1).unstack(['store_nbr', 'family'])
X2_test = X2_test.stack() # onpromotion feature
X2_test = X2_test.reset_index('family')
X2_test['family'] = le.fit_transform(X2_test['family'])
X2_test["day"] = X2_test.index.day

y_submit = pd.DataFrame(model.predict(X_test, X2_test), index=X_test.index, columns=y.columns).clip(0.0)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission2.csv', index=False)
