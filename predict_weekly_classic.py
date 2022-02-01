from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingTreeRegressor, iSOUPTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_dt(col):
    col = str(col)
    dt = col[:4]+'-'+col[4:6]+'-'+col[6:]
    return dt

# import data
df = pd.read_csv('covid_slovenia.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'NewCases']]

# preprocess data on measures
measures = pd.read_csv('OxCGRT_latest.csv')
slovenia = measures[measures['CountryName'] == 'Slovenia']
slovenia['date'] = pd.to_datetime(slovenia['Date'].apply(create_dt))
slovenia = slovenia[['date', 'StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']]
slovenia.fillna(method='ffill', inplace=True)

# transform data on weekly, merge
df = df.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')
slovenia = slovenia.resample('W-Sat', on='date').mean().reset_index().sort_values(by='date')
df = df.merge(slovenia, how='inner', on=['date'])


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

X, y = df[['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']], df['NewCases']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=50000).fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))