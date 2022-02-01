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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('covid_slovenia.csv')

# # train/test split
train = df[:]
X_train, y_train = train.loc[:, 'sin_day_of_week':'cos_month'], train['NewCases']

test = df[600:]
X_test, y_test = test.loc[:, 'sin_day_of_week':'cos_month'], test['NewCases']

# train = df[:600]
# X_train, y_train = train[['day_of_week', 'month']], train['NewCases']
#
# test = df[600:]
# X_test, y_test = test[['day_of_week', 'month']], test['NewCases']

# Naive mean approach
print('MEAN', np.sqrt(mean_squared_error(y_test, [np.mean(y_train)]*len(y_test))))

# KNN
reg_knn = KNeighborsRegressor(n_neighbors=7)
reg_knn.fit(X_train, y_train)

y_pred_kNN = reg_knn.predict(X_test)

print('kNN', np.sqrt(mean_squared_error(y_test, y_pred_kNN)))

# RF
reg_rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=7)
reg_rf.fit(X_train, y_train)

y_pred_RF = reg_rf.predict(X_test)

print('RandomForest', np.sqrt(mean_squared_error(y_test, y_pred_RF)))
print('RandomForest;RRMSE', np.sqrt(mean_squared_error(y_test, y_pred_RF))/(np.max(y_test)-np.min(y_test)))
print('RandomForest;RRMSE', np.sqrt(mean_squared_error(y_test, y_pred_RF))/(np.mean(y_test)))

# Linear regression
reg_lr = LinearRegression()
reg_lr.fit(X_train, y_train)

y_pred_LR = reg_lr.predict(X_test)

print('LR', np.sqrt(mean_squared_error(y_test, y_pred_LR)))

# MLP
reg_mlp = MLPRegressor(max_iter=1000)
reg_mlp.fit(X_train, y_train)

y_pred_MLP = reg_mlp.predict(X_test)

print('MLP', np.sqrt(mean_squared_error(y_test, y_pred_MLP)))

# put into table and process by week
true_test_pred = pd.DataFrame({
    'true': y_test,
    'pred_RF': y_pred_RF,
    'pred_kNN': y_pred_kNN,
    'pred_LR': y_pred_LR,
    'pred_MLP': y_pred_MLP,
    'date': pd.to_datetime(test['date'])
})

weekly_sums = true_test_pred.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')

# add baseline
weekly_sums['pred_baseline'] = [np.nan] + list(weekly_sums['true'])[:-1]
weekly_sums = weekly_sums.iloc[1:]

np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_baseline']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_RF']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_kNN']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_LR']))


true_train_pred = pd.DataFrame({
    'true': y_train,
    'pred_RF': reg_rf.predict(X_train),
    'pred_kNN': reg_knn.predict(X_train),
    'pred_LR': reg_lr.predict(X_train),
    'pred_MLP': reg_mlp.predict(X_train),
    'date': pd.to_datetime(train['date'])
})

weekly_sums = true_train_pred.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')

# add baseline
weekly_sums['pred_baseline'] = [np.nan] + list(weekly_sums['true'])[:-1]
weekly_sums = weekly_sums.iloc[1:]

np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_baseline']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_RF']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_kNN']))
np.sqrt(mean_squared_error(weekly_sums['true'], weekly_sums['pred_LR']))

# Adaptive Models
ht_arf = AdaptiveRandomForestRegressor(n_estimators=100, random_state=42, memory_estimate_period=21, grace_period=3, drift_detection_method=None, warning_detection_method=None)
# ht_arf = HoeffdingTreeRegressor(memory_estimate_period=21, grace_period=2)
# ht_arf = KNNRegressor()

# Auxiliary variables to control loop and track performance
n_samples = 0
max_samples = len(train)
y_pred = np.zeros(max_samples)
y_true = np.zeros(max_samples)

# Run test-then-train loop for max_samples and while there is data
for idx in range(max_samples):
    X, y = X_train.iloc[idx], y_train.iloc[idx]
    X, y = X.values.reshape(1, 6), np.array(y).reshape(1, 1)  # when you get error, check number of features in a dataset
    y_true[n_samples] = y
    y_pred[n_samples] = ht_arf.predict(X)[0]
    ht_arf.partial_fit(X, y)
    n_samples += 1

y_pred_arf = []

for example in X_test.values:
    pred = ht_arf.predict([example])
    y_pred_arf.append(pred[0])

print('AdaptiveRandomForestRegressor', np.sqrt(mean_squared_error(y_test, y_pred_arf)))

plt.plot(list(range(len(y_test))), y_test)
plt.plot(list(range(len(y_test))), y_pred_arf)
plt.title('AdaptiveRandomForest')
plt.savefig(f'AdaptiveRandomForest_prediction.png')
plt.show()
plt.clf()

plt.plot(list(range(len(y_true))), y_true)
plt.plot(list(range(len(y_true))), y_pred)
plt.title('AdaptiveRandomForest')
plt.savefig(f'AdaptiveRandomForest_prediction.png')
plt.show()
plt.clf()