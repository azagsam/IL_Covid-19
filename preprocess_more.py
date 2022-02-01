import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# Load the data
url_new_cases = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Cases.csv"
url_new_deaths = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Deaths.csv"

df_deaths = pd.read_csv(url_new_deaths)
df_cases = pd.read_csv(url_new_cases)

# Merge into a single dataframe
df_all = df_cases.merge(df_deaths, how='inner', on=["location_name", "date"])

# Extract data for Slovenia only
l = ['Slovenia', 'Austria', 'Italy', 'Hungary', 'Croatia']
df_slo = df_all[df_all['location_name'].isin(l)].reset_index(drop=True)
df_slo['date'] = pd.to_datetime(df_slo['date'])
df_slo['day_of_week'] = df_slo['date'].dt.dayofweek
df_slo['month'] = df_slo['date'].dt.month
df_slo['day_of_month'] = df_slo['date'].dt.day


df_slo['sin_day_of_week'] = np.sin(2*np.pi*df_slo['day_of_week']/7)
df_slo['cos_day_of_week'] = np.cos(2*np.pi*df_slo['day_of_week']/7)

df_slo['sin_day_of_month'] = np.sin(2*np.pi*df_slo['day_of_month']/31)
df_slo['cos_day_of_month'] = np.cos(2*np.pi*df_slo['day_of_month']/31)

df_slo['sin_month'] = np.sin(2*np.pi*df_slo['month']/12)
df_slo['cos_month'] = np.cos(2*np.pi*df_slo['month']/12)

# Save to CSV
new_columns = {
    'value_x': 'NewCases',
    'value_y': 'NewDeaths'
}

df_slo = df_slo.rename(columns=new_columns)
df_slo.to_csv("covid_more_final.csv", index=True)