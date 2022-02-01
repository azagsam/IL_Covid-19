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

# import data
df = pd.read_csv('covid_slovenia.csv')
df['date'] = pd.to_datetime(df['date'])

measures = pd.read_csv('OxCGRT_latest.csv')
slovenia = measures[measures['CountryName'] == 'Slovenia']

# transform data for experiments on weekly scale
FEATURES = 2
X_train, y_train = df.loc[:, 'sin_day_of_week':'cos_day_of_week'], df['NewCases']

# Adaptive Models
models = {
    'ARFR': AdaptiveRandomForestRegressor(n_estimators=10, random_state=42, memory_estimate_period=7, grace_period=2, drift_detection_method=None, warning_detection_method=None),
    'KNN': KNNRegressor(max_window_size=7, n_neighbors=2)  # best results 3 / 1
}

next_week_predictions = {}

for model_name, model in models.items():
    print(model_name)
    # Auxiliary variables to control loop and track performance
    n_samples = 0
    max_samples = len(X_train)
    y_pred = np.zeros(max_samples)
    y_true = np.zeros(max_samples)
    next_week_pred = [0.0] # predict zero for first week

    # Run test-then-train loop for max_samples and while there is data
    for idx in range(max_samples):
        X, y = X_train.iloc[idx], y_train.iloc[idx]
        X, y = X.values.reshape(1, FEATURES), np.array(y).reshape(1, 1)  # when you get error, check number of features in a dataset

        # predict for next week
        if df.iloc[idx]['day_of_week'] == 6 and idx < 710:
            next_week = []
            for i in range(7):
                X = X_train.iloc[idx+i]
                X = X.values.reshape(1, FEATURES)
                next_week.append(model.predict(X)[0])
            next_week_pred.append(sum(next_week))
        # predict for last week
        if idx == 710:
            next_week = []
            for i in range(6):  # last week has only 6 days
                X = X_train.iloc[idx+i]
                X = X.values.reshape(1, FEATURES)
                next_week.append(model.predict(X)[0])
            next_week_pred.append(sum(next_week))

        y_true[n_samples] = y
        y_pred[n_samples] = model.predict(X)[0]
        model.partial_fit(X, y)

        n_samples += 1

    next_week_predictions[model_name] = next_week_pred

print('Results on DAILY scale: ')

# transform to weekly sums to get prediction for next week
weekly_sums = df.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')

# add baseline models and predictions from both ML models
weekly_sums['1_week_baseline'] = [0] + list(weekly_sums['NewCases'])[:-1]
weekly_sums['4_week_baseline'] = [0] + list(weekly_sums['NewCases'].rolling(window=4, min_periods=1).mean()[:-1])
weekly_sums['KNN'] = next_week_predictions['KNN']
weekly_sums['ARFR'] = next_week_predictions['ARFR']

for model in ['1_week_baseline', '4_week_baseline', 'ARFR', 'KNN']:
    plt.plot(list(range(len(weekly_sums['NewCases']))), weekly_sums['NewCases'])
    plt.plot(list(range(len(weekly_sums['NewCases']))), weekly_sums[model])
    plt.title(model)
    plt.savefig(f'{model}.png')
    plt.show()
    plt.clf()

print('\nNon-relative RMSE:')
print('1_week_baseline', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['1_week_baseline'])))
print('4_week_baseline', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['4_week_baseline'])))
print('ARFR', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['ARFR'])))
print('kNN', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['KNN'])))

weekly_sums.to_csv('test_predictions.csv', columns=['NewCases', '1_week_baseline', '4_week_baseline', 'ARFR', 'KNN'])

print('\nRelative RMSE:')
baseline = np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['4_week_baseline']))
print('1_week_baseline', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['1_week_baseline'])) / baseline)
print('ARFR', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['ARFR'])) / baseline)
print('kNN', np.sqrt(mean_squared_error(weekly_sums['NewCases'], weekly_sums['KNN'])) / baseline)

# else:
#     print('Results on WEEKLY scale: ')
#     # add baseline models
#     df['pred_baseline'] = [0] + list(df['NewCases'])[:-1]
#     df['4_week_baseline'] = df['NewCases'].rolling(window=4, min_periods=1).mean()
#
#     for model in ['pred_baseline', '4_week_baseline', 'ARFR', 'KNN']:
#         plt.plot(list(range(len(df['NewCases']))), df['NewCases'])
#         plt.plot(list(range(len(df['NewCases']))), df[model])
#         plt.title(model)
#         plt.savefig(f'Weekly-{WEEKLY}-{model}.png')
#         plt.show()
#         plt.clf()
#
#     print('\nNon-relative RMSE:')
#     print('Baseline', np.sqrt(mean_squared_error(df['NewCases'], df['pred_baseline'])))
#     print('4_week_baseline', np.sqrt(mean_squared_error(df['NewCases'], df['4_week_baseline'])))
#     print('AEFR', np.sqrt(mean_squared_error(df['NewCases'], df['ARFR'])))
#     print('kNN', np.sqrt(mean_squared_error(df['NewCases'], df['KNN'])))
#
#     print('\nRelative RMSE:')
#     baseline = np.sqrt(mean_squared_error(df['NewCases'], df['4_week_baseline']))
#     print('Baseline', np.sqrt(mean_squared_error(df['NewCases'], df['pred_baseline'])) / baseline)
#     print('AEFR', np.sqrt(mean_squared_error(df['NewCases'], df['ARFR'])) / baseline)
#     print('kNN', np.sqrt(mean_squared_error(df['NewCases'], df['KNN'])) / baseline)
