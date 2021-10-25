# -*- coding: utf-8 -*-
"""
"""
#data = 'https://www.kaggle.com/c/store-sales-time-series-forecasting/data/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

with zipfile.ZipFile('store-sales-time-series-forecasting.zip', 'r' ) as sales_zip:
     train = pd.read_csv(sales_zip.extract('train.csv'),  
                         usecols=['store_nbr', 'family', 'date', 'sales'],
                         dtype={'store_nbr': 'category', 'family': 'category', 'sales': 'float32',}, 
                         parse_dates=['date'], 
                         infer_datetime_format=True)
     holidays_events = pd.read_csv(sales_zip.extract('holidays_events.csv'),
                                                     dtype={'type': 'category',
                                                            'locale': 'category',
                                                            'locale_name': 'category',
                                                            'description': 'category',
                                                            'transferred': 'bool'},
                                                     parse_dates=['date'],
                                                     infer_datetime_format=True)
     df_test = pd.read_csv(sales_zip.extract('test.csv'),
                           dtype={'store_nbr': 'category', 'family': 'category'},
                           parse_dates=['date'],
                           infer_datetime_format=True)
     
holidays_events = holidays_events.set_index('date').to_period('D')
train['date'] = train.date.dt.to_period('D')
df_test['date'] = df_test.date.dt.to_period('D')

df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
train = train.set_index(['store_nbr', 'family', 'date']).sort_index()

holidays = (holidays_events
            .query("locale in ['National', 'Regional']")
            .loc['2017':'2017-08-15', ['description']]
            .assign(description=lambda x: x.description.cat.remove_unused_categories())
)
X_holidays = pd.get_dummies(holidays)

# Model
y = train.unstack(['store_nbr', 'family']).loc["2017"]

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)

X = X.join(X_holidays, on='date').fillna(0.0)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns).clip(0.0)

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

y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)