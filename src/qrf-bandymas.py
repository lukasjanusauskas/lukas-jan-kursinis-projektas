import pandas as pd
import numpy as np
import quantile_forest as qrf
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from src.data_preparation import feature_engineering, prepare_time_series

# Load
X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')

X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

X_test = np.load('X_test_final.npy')
y_test = np.load('y_test_final.npy')

n_timesteps = X_train.shape[1]
n_inp_features = X_train.shape[2]
n_out_features = y_train.shape[2]

reshape_params_x = (-1, n_inp_features*n_timesteps)
reshape_params_y = (-1, n_out_features*n_timesteps)

X_train = X_train.reshape(reshape_params_x)
X_val = X_val.reshape(reshape_params_x)
X_test = X_test.reshape(reshape_params_x)

y_train = y_train.reshape(reshape_params_y)
y_val = y_val.reshape(reshape_params_y)
y_test = y_test.reshape(reshape_params_y)


# Anomaly data
df_anom = pd.read_csv(
    'data/anomaly_meteo.csv', 
    index_col=['Unnamed: 0'],
    parse_dates=['position_timestamp']
)

df_anom.drop(columns=['windWaveDirection', 'windWaveHeight'], inplace=True)

df_anom = df_anom.rename(columns={
    'course': 'COG',
    'heading': 'Heading',
    'position_timestamp': '# Timestamp',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'speed': 'SOG'
})
df_anom['MMSI'] = 414270000

dfs = []
above_interpolation_limit = 0

df_anom['Heading'] = np.clip( df_anom['Heading'], 0, 360 )
df_anom['COG'] = np.clip( df_anom['COG'], 0, 360 )

df_anom['hour'] = df_anom['# Timestamp'].dt.round('h')
df_anom = df_anom.drop(columns=['hour'])

df_anom, timestamps = feature_engineering(df_anom, return_timestamps=True)
df_anom.drop(columns=['Latitude', 'Longitude'], inplace=True)

min_max_scale_cols = [
    # 'Latitude',
    # 'Longitude',
    'SOG',
    'currentSpeed',
    'gust',
    'swellHeight',
    'waveHeight',
    'windSpeed'
]

with open('output/min-max-scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df_anom[min_max_scale_cols] = scaler.fit_transform(
    df_anom[min_max_scale_cols].values
)

df_anom.dropna(inplace=True)
df_anom.drop(columns=['time'], inplace=True)

X_arr, y_arr = prepare_time_series(df_anom)
X_arr = X_arr.reshape(reshape_params_x)
y_arr = y_arr.reshape(reshape_params_y)

params = {
    'max_depth': 16,
    'max_samples': 0.75,
    'max_samples_leaf': 1,
    'max_features': 0.75
}

qrf_model = qrf.RandomForestQuantileRegressor(
    **params,
    n_jobs=-1,
    verbose=True
)

qrf_model.fit(X_train, y_train)
with open('output/qrf-final.pkl', 'wb+') as f:
    pickle.dump(qrf_model, f)

y_pred = qrf_model.predict(
    X_arr.reshape(reshape_params_x),
    quantiles=[0.025, 0.975]
)
y_low = y_pred[:, :, 0]
y_high = y_pred[:, :, 1]

res = np.mean(
    (y_low[450:, :] < y_arr[450:, :]) &
    (y_high[450:, :] > y_arr[450:, :])
)
print('coverage after and including raising:', res)

res = np.mean(
    (y_low[550:, :] < y_arr[550:, :]) &
    (y_high[550:, :] > y_arr[550:, :])
)
print('coverage after raising:', res)

res = np.mean(
    (y_low[:450, :] < y_arr[:450, :]) &
    (y_high[:450, :] > y_arr[:450, :])
)
print('coverage before raising:', res)

# validation
y_pred_val = qrf_model.predict(X_val, quantiles=[0.025, 0.975])
y_low_val = y_pred_val[:, :, 0]
y_high_val = y_pred_val[:, :, 1]

in_pi_mask = (
    (y_val < y_high_val) &
    (y_val > y_low_val)
)

val_mmsis = np.load('val-mmsis-test.npy', allow_pickle=True)
val_days = np.load('val-days-test.npy', allow_pickle=True)

group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [ in_pi_mask[ix] ]
    else:
        group_results[(mmsi, day)].append( in_pi_mask[ix] )

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks)\
        .reshape((-1, n_timesteps, n_out_features))\
        .mean(axis=1)\
        .mean(axis=0)

    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means[0],
        'delta_dif': means[1],
    })

df = pd.DataFrame(df)
thresh_delta_cog = np.quantile(df['delta_cog'], 0.05)
thresh_delta_dif = np.quantile(df['delta_dif'], 0.05)

print( 'delta_cog limit:', np.quantile(df['delta_cog'], 0.05) )
print( 'delta_dif liimt:', np.quantile(df['delta_dif'], 0.05) )

in_pi_mask.dump('output/in_pi_mask_val.npy')

# test
y_pred_test = qrf_model.predict(X_test, quantiles=[0.025, 0.975])
y_low_test = y_pred_test[:, :, 0]
y_high_test = y_pred_test[:, :, 1]

in_pi_mask = (
    (y_test < y_high_test) &
    (y_test > y_low_test)
)

in_pi_mask.dump('output/in_pi_mask_test.npy')
