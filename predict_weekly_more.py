from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor


def create_dt(col):
    col = str(col)
    dt = col[:4] + '-' + col[4:6] + '-' + col[6:]
    return dt


def save_to_file(predictions_list, expected_len, filename="predictions"):
    """
    Saves the file to TXT and pads the leading zeros.
    :param predictions_list: list of predictions
    :param expected_len: desired output length of the list (for padding)
    :param filename: filename to use for saving the prediction (default: predicitons)
    """
    while len(predictions_list) < expected_len:
        predictions_list.insert(0, 0)
    textfile = open("{}.txt".format(filename), "w")
    for element in predictions_list:
        textfile.write(str(int(element)) + "\n")
    textfile.close()


# import data
df = pd.read_csv('covid_more_final.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'NewCases', 'NewDeaths', 'location_name']]
df = df.pivot(columns='location_name', values=['NewCases', 'NewDeaths'], index='date')
df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
df = df.reset_index()

# preprocess data on measures
measures = pd.read_csv('OxCGRT_latest_final.csv')
slovenia = measures[measures['CountryName'] == 'Slovenia']
slovenia['date'] = pd.to_datetime(slovenia['Date'].apply(create_dt))
slovenia = slovenia[
    ['date', 'StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']]
slovenia.fillna(method='ffill', inplace=True)

# # # selext only certain measures
# measures = pd.read_csv('OxCGRT_latest.csv')
# slovenia = measures[measures['CountryName'] == 'Slovenia']
# slovenia['date'] = pd.to_datetime(slovenia['Date'].apply(create_dt))
#
# slovenia = slovenia.filter(regex=("C[0-9].*|date|E[0-9].*|H[0-9].*"))
# slovenia = slovenia[slovenia.columns.drop(list(slovenia.filter(regex='Flag')))]
# slovenia.fillna(method='ffill', inplace=True)
#
# slovenia.isnull().values.any()

# transform data on weekly, merge
df = df.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')
df.rename(columns={'Slovenia': 'NewCases'}, inplace=True)
slovenia = slovenia.resample('W-Sat', on='date').mean().reset_index().sort_values(by='date')
df = df.merge(slovenia, how='inner', on=['date'])

# # create new variables and shift information
df['PrevCases_Slovenia'] = [0] + list(df['NewCases_Slovenia'])[:-1]
df['PrevDeaths_Slovenia'] = [0] + list(df['NewDeaths_Slovenia'])[:-1]

df['PrevCases_Austria'] = [0] + list(df['NewCases_Austria'])[:-1]
df['PrevDeaths_Austria'] = [0] + list(df['NewDeaths_Austria'])[:-1]

df['PrevCases_Croatia'] = [0] + list(df['NewCases_Croatia'])[:-1]
df['PrevDeaths_Croatia'] = [0] + list(df['NewDeaths_Croatia'])[:-1]

df['PrevCases_Hungary'] = [0] + list(df['NewCases_Hungary'])[:-1]
df['PrevDeaths_Hungary'] = [0] + list(df['NewDeaths_Hungary'])[:-1]
# df['PrevCases'] = df['NewCases'].shift(periods=1).fillna(method='bfill').astype('int')
# df.drop(columns=['date']).corr()

# plt.plot(list(range(len(df['NewCases']))), df['Austria'], label='Austria')
# plt.plot(list(range(len(df['NewCases']))), df['Croatia'], label='Croatia')
# plt.plot(list(range(len(df['NewCases']))), df['Hungary'], label='Hungary')
# # plt.plot(list(range(len(df['NewCases']))), df['Italy'], label='Italy')
# plt.plot(list(range(len(df['NewCases']))), df['NewCases'], label='Slovenia')
# # plt.plot(list(range(len(df['NewCases']))), df[model])
# plt.title('New weekly cases')
# plt.legend(loc='upper left')
# # plt.savefig(f'oxford{model}.png')
# plt.show()
# plt.clf()

# EXP 2 - all 5 features
# df['dummy'] = [1]*len(df)  # dummy is used because fit method doesnt work on single feature data
X_train, y_train = df[[
                       'PrevCases_Austria', 'PrevCases_Croatia', 'PrevCases_Hungary',
                       'PrevDeaths_Austria', 'PrevDeaths_Croatia', 'PrevDeaths_Hungary',
                       'PrevCases_Slovenia', 'PrevDeaths_Slovenia',
                       'StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex',
                       # 'dummy',
                       ]], \
                   df['NewCases_Slovenia']

# X_train['prevalent_variant'] = [0]*102 + [1]*4

FEATURES = len(X_train.columns)
# Models
# Adaptive Models
models = {
    'ARFR': AdaptiveRandomForestRegressor(n_estimators=100,
                                          random_state=45,
                                          # memory_estimate_period=100,
                                          # grace_period=100,
                                          drift_detection_method=None,
                                          warning_detection_method=None
                                          ),
    'KNN': KNNRegressor(max_window_size=4, n_neighbors=1),
    # 'HTR': HoeffdingTreeRegressor()
}

for model_name, model in models.items():
    # print(model_name)
    # Auxiliary variables to control loop and track performance
    n_samples = 0
    max_samples = len(X_train)
    y_pred = np.zeros(max_samples)
    y_true = np.zeros(max_samples)

    # Run test-then-train loop for max_samples and while there is data
    for idx in range(max_samples):
        X, y = X_train.iloc[idx], y_train.iloc[idx]
        if FEATURES == 1:
            X, y = np.array(X).reshape(1, 1), np.array(y).reshape(1, 1)
        else:
            X, y = X.values.reshape(1, FEATURES), np.array(y).reshape(1, 1)
        y_true[n_samples] = y
        y_pred[n_samples] = model.predict(X)[0]
        model.partial_fit(X, y)
        n_samples += 1

    df[model_name] = y_pred
    print(f'{model_name}, RMSE (all):', np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f'{model_name}, RMSE (last 10-weeks):', np.sqrt(mean_squared_error(y_true[-10:], y_pred[-10:])))

# add baselines
df['1_week_baseline'] = [0] + list(df['NewCases_Slovenia'])[:-1]
print('1_week_baseline, RMSE (all):', np.sqrt(mean_squared_error(df['NewCases_Slovenia'], df['1_week_baseline'])))
print('1_week_baseline, RMSE (last 10-weeks):',
      np.sqrt(mean_squared_error(df['NewCases_Slovenia'][-10:], df['1_week_baseline'][-10:])))
df['4_week_baseline'] = [0] + list(df['NewCases_Slovenia'].rolling(window=4, min_periods=1).mean()[:-1])
print('4_week_baseline, RMSE (all):', np.sqrt(mean_squared_error(df['NewCases_Slovenia'], df['4_week_baseline'])))
print('4_week_baseline, RMSE (last 10-weeks):',
      np.sqrt(mean_squared_error(df['NewCases_Slovenia'][-10:], df['4_week_baseline'][-10:])))

for model in ['1_week_baseline', '4_week_baseline', 'ARFR', 'KNN']:
    plt.plot(list(range(len(df['NewCases_Slovenia']))), df['NewCases_Slovenia'])
    plt.plot(list(range(len(df['NewCases_Slovenia']))), df[model])
    plt.title(model)
    plt.savefig(f'oxford{model}.png')
    plt.show()
    plt.clf()

# MLP partial fit
results_all, results_10 = [], []
for _ in range(30):
    model = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.001)

    n_samples = 0
    max_samples = len(X_train)
    y_pred = np.zeros(max_samples)
    y_true = np.zeros(max_samples)

    keep = 1  # 1 works the best
    for idx in range(1, max_samples + 1):
        if idx < keep:
            X, y = X_train.iloc[:idx], y_train.iloc[:idx]
        else:
            X, y = X_train.iloc[idx - keep:idx], y_train.iloc[idx - keep:idx]

        if idx < keep:
            X, y = X.values.reshape(idx, FEATURES), np.array(y).reshape(idx, )
        else:
            X, y = X.values.reshape(keep, FEATURES), np.array(y).reshape(keep, )

        y_true[n_samples] = y[-1]
        if idx != 1:
            single_prediction = model.predict(X[-1].reshape(1, -1))[0]
            if single_prediction < 0:
                y_pred[n_samples] = 0
            else:
                y_pred[n_samples] = single_prediction

        # fit model a number of times
        for n in range(50):
            indices = list(range(len(X)))
            shuffle(indices)
            X, y = X[indices], y[indices]
            model.partial_fit(X, y)
        n_samples += 1

    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_10 = mean_squared_error(y_true[-10:], y_pred[-10:], squared=False)
    print('NN, RMSE (all):', rmse_all)
    print(f'NN, RMSE: (last 10-weeks)', rmse_10)
    results_10.append(rmse_10)
    results_all.append(rmse_all)

    plt.plot(list(range(len(y_true))), y_true)
    plt.plot(list(range(len(y_true))), y_pred)
    plt.title('MLP')
    plt.savefig(f'oxford-mlp.png')
    plt.show()
    plt.clf()


import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

print(mean_confidence_interval(results_all))
print(mean_confidence_interval(results_10))


