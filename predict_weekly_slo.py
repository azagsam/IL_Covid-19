from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler
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
from random import shuffle

def create_dt(col):
    col = str(col)
    dt = col[:4]+'-'+col[4:6]+'-'+col[6:]
    return dt

def save_to_file(predictions_list, expected_len, filename="predictions"):
    """
    Saves the file to TXT and pads the leading zeros.
    :param predictions_list: list of predictions
    :param expected_len: desired output length of the list (for padding)
    :param filename: filename to use for saving the prediction (default: predicitons)
    """
    while len(predictions_list) < expected_len:
        predictions_list.insert(0,0)
    textfile = open("{}.txt".format(filename), "w")
    for element in predictions_list:
        textfile.write(str(int(element)) + "\n")
    textfile.close()


# import data
df = pd.read_csv('covid_slovenia_final.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'NewCases', 'NewDeaths']]

# preprocess data on measures
measures = pd.read_csv('OxCGRT_latest_final.csv')
slovenia = measures[measures['CountryName'] == 'Slovenia']
slovenia['date'] = pd.to_datetime(slovenia['Date'].apply(create_dt))
slovenia = slovenia[['date', 'StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']]
slovenia.fillna(method='ffill', inplace=True)

# # selext only certain measures
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
slovenia = slovenia.resample('W-Sat', on='date').mean().reset_index().sort_values(by='date')
df = df.merge(slovenia, how='inner', on=['date'])

# create new variables
df['PrevCases'] = [0] + list(df['NewCases'])[:-1]
df['PrevDeaths'] = [0] + list(df['NewDeaths'])[:-1]
df.drop(columns=['date']).corr()

# df['variant'] = [0]*25 + [1]*25 + [2]*25 + [3]*(len(df)-75)

# plots
plt.plot(list(range(len(df['NewCases']))), df['NewCases'])
# plt.plot(list(range(len(df['NewCases']))), df[model])
# plt.title(model)
# plt.savefig(f'oxford{model}.png')
plt.show()
plt.clf()

# #  splits
# # EXP 1 - only previous cases
# df['dummy'] = [1]*len(df)  # dummy is used because fit method doesnt work on single feature data
# X_train, y_train = df[['PrevCases', 'dummy']], df['NewCases']

# EXP 2 - all 5 features
X_train, y_train = df[['PrevCases',
                       'PrevDeaths',
                       'StringencyIndex',
                       'GovernmentResponseIndex',
                       'ContainmentHealthIndex',
                       'EconomicSupportIndex',]], \
                   df['NewCases']

# # EXP 2 - all 5 features with time lag two week
# X_train, y_train = df[['PrevCases', 'StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']], df['NewCases']
# lagging_period = -1
# lagged_features = X_train[['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']].shift(periods=lagging_period).fillna(method='ffill')
# X_train[['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']] = lagged_features[['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex', 'EconomicSupportIndex']]

# EXP 3
# df['avg_measures'] = (df['StringencyIndex'] + df['GovernmentResponseIndex'] + df['ContainmentHealthIndex'] + df['EconomicSupportIndex']) / 4

# # # EXP 4
# # # EXP 2 - only sertain measures
# X_train, y_train = df.drop(columns=['NewCases', 'date']), df['NewCases']

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
df['1_week_baseline'] = [0] + list(df['NewCases'])[:-1]
print('1_week_baseline, RMSE (all):', np.sqrt(mean_squared_error(df['NewCases'], df['1_week_baseline'])))
print('1_week_baseline, RMSE (last 10-weeks):', np.sqrt(mean_squared_error(df['NewCases'][-10:], df['1_week_baseline'][-10:])))
df['4_week_baseline'] = [0] + list(df['NewCases'].rolling(window=4, min_periods=1).mean()[:-1])
print('4_week_baseline, RMSE (all):', np.sqrt(mean_squared_error(df['NewCases'], df['4_week_baseline'])))
print('4_week_baseline, RMSE (last 10-weeks):', np.sqrt(mean_squared_error(df['NewCases'][-10:], df['4_week_baseline'][-10:])))

for model in ['1_week_baseline', '4_week_baseline', 'ARFR', 'KNN']:
    plt.plot(list(range(len(df['NewCases']))), df['NewCases'])
    plt.plot(list(range(len(df['NewCases']))), df[model])
    plt.title(model)
    plt.savefig(f'oxford{model}.png')
    plt.show()
    plt.clf()


# MLP partial fit
model = MLPRegressor(hidden_layer_sizes=(100, ), learning_rate_init=0.001)

n_samples = 0
max_samples = len(X_train)
y_pred = np.zeros(max_samples)
y_true = np.zeros(max_samples)

keep = 1  # 1 works the best
for idx in range(1, max_samples+1):
    if idx < keep:
        X, y = X_train.iloc[:idx], y_train.iloc[:idx]
    else:
        X, y = X_train.iloc[idx-keep:idx], y_train.iloc[idx-keep:idx]

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
    for n in range(15):
        indices = list(range(len(X)))
        shuffle(indices)
        X, y = X[indices], y[indices]
        model.partial_fit(X, y)
    n_samples += 1

print('NN, RMSE (all):', np.sqrt(mean_squared_error(y_true, y_pred)))
print(f'NN, RMSE: (last 10-weeks)', mean_squared_error(y_true[-10:], y_pred[-10:], squared=False))

plt.plot(list(range(len(y_true))), y_true)
plt.plot(list(range(len(y_true))), y_pred)
plt.title('MLP')
plt.savefig(f'oxford-mlp.png')
plt.show()
plt.clf()