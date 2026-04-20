import sys
import pickle
import time
import pandas as pd # type: ignore
import random
import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # type: ignore

from scipy.spatial import cKDTree # type: ignore
from scipy.stats import boxcox
from itertools import product

import warnings
from pandas.errors import PerformanceWarning # type: ignore

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Coordinate grid
from src.util import (
    LAT_MIN, LAT_MAX,
    LON_MIN, LON_MAX
)
from src.meteorological import construct_coordinate_grid

# Constants for time step configuration

STEP_BACK_DEFAULT = 25
STEP_FORW_DEFAULT = 25
ROLLING_WINDOW_SIZE = 25

# Interpolation limit of two hours is set because it is standard in literature
INTERPOLATION_LIMIT = 2

# Will be required for: calculating components (sine, cosine) and interpolation clipping
ANGLE_COLS = [
    'COG',
    'currentDirection',
    'swellDirection',
    'waveDirection',
    'windDirection'
]

SMOOTHING = True if sys.argv[1] == 'y' else False
print('Smoothing:', SMOOTHING)

def merge_with_meteo(
    row: pd.Series,
    weather_df: pd.DataFrame,
    coordinate_grid: list,
    coordinate_tree: cKDTree
) -> pd.Series:

    # query returns distances (d) and indices (i)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html
    _, indices = coordinate_tree.query(
        (row['Latitude'], row['Longitude']), k=1
    )
    lat, lon = coordinate_grid[indices]

    time = row['hour']
    weather_row = weather_df.loc[(time, lat, lon), :]

    data = np.concatenate( 
        [row.values, weather_row.values[0]]
    )

    columns = list(row.index) + list(weather_df.columns)
    return pd.Series(data=data, index=columns)


def feature_engineering(df: pd.DataFrame, return_timestamps: bool = False) -> pd.DataFrame:

    # Calculate diff
    df['dif'] = ( df['Heading'] - df['COG']  + 180) % 360 - 180
    df.drop( columns=['Heading'], inplace=True )

    for col in ANGLE_COLS:
        df[f'{col}_vec_x'] = np.cos(df[col])
        df[f'{col}_vec_y'] = np.sin(df[col])

    # Drop all angle columns apart from  COG
    drop_angle = np.setdiff1d(ANGLE_COLS, ['COG'])
    df.drop(columns=drop_angle, inplace=True)

    # Do min-max scaling to [-1, 1]
    df['COG'] = 180 - np.abs(180 - df['COG'])
    df['COG'] = (df['COG'] - 90) / 90
    df['COG-dif'] = df['COG'].diff(1) / 2

    df['dif'] = df['dif'] / 180
    df['delta_dif'] = df['dif'].diff(1) / 2

    df.drop(columns=['dif'], inplace=True)

    columns_to_drop = ['# Timestamp', 'MMSI', 'Latitude', 'Longitude']

    if return_timestamps:
        timestamps = df['# Timestamp'].values
        df.drop(columns=columns_to_drop, inplace=True)
        return df, timestamps

    else:
        df.drop(columns=columns_to_drop, inplace=True)
        return df


def prepare_time_series(
    df: pd.DataFrame,
    n_lags: int = STEP_BACK_DEFAULT,
    n_future_lag: int = STEP_FORW_DEFAULT
):
    outputs = [ 'COG-dif', 'delta_dif' ]

    n_features = df.shape[1]

    # so the input would be in accordance with:
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM#call-arguments
    X_arr = np.empty( (df.shape[0], n_lags, n_features) )
    y_arr = np.empty( (df.shape[0], n_future_lag, len(outputs)) )

    for ix, col in enumerate( df.columns ):
        for lag in range( n_lags ):
            X_arr[:, lag, ix] = df[col].shift(-1 * lag)

    for ix, col in enumerate( outputs ):
        for lag in range( 1, n_lags+1 ):
            y_arr[:, lag-1, ix] = df[col].shift(lag)

    return X_arr[n_lags:-n_lags, :, :], y_arr[n_lags:-n_lags, :, :]


def partition_days(days: list):

    days_shuffled = days.copy()
    random.shuffle( days_shuffled )

    n = len(days_shuffled)
    n_train = int(0.7 * n)
    n_val = int( 0.85 * n )

    return (
        days_shuffled[:n_train],
        days_shuffled[n_train:n_val],
        days_shuffled[n_val:]
    )


if __name__ == "__main__":

    meteo_df = pd.read_csv(
        'data/weather_df.csv',
        parse_dates=['time']
    )

    latitudes = meteo_df['Latitude'].unique()
    longitudes = meteo_df['Longitude'].unique()
    lat_lon_grid = list(product(latitudes, longitudes))
    lat_lon_tree = cKDTree(lat_lon_grid)

    meteo_df = meteo_df\
        .set_index( ['time', 'Latitude', 'Longitude'])\
        .drop(columns=['Unnamed: 0'])

    raw_df = pd.read_csv(
        'data/ais_dataset.csv',
        parse_dates=['# Timestamp'],
        dayfirst=True
    ).drop(columns=['Unnamed: 0'])
    print('read')

    print( 'AIS signals', raw_df.shape[0] )

    # We will groupby day also, so that we do not have huge gaps to interpolate
    raw_df['day'] = raw_df['# Timestamp'].dt.normalize()

    grouped_df = raw_df\
        .set_index('# Timestamp')\
        .groupby(['MMSI', 'day'])

    dfs = []
    all_time_diffs = []
    missing_count = 0
    above_interpolation_limit = 0

    for (mmsi, day), group_df in grouped_df:

        group_df = group_df.drop(columns=['day'])

        # We define a track to be at least 50 measurements, 
        #   and does not have a gap bigger than 2 hours 

        time_diffs = group_df.reset_index()\
            ['# Timestamp']\
            .diff(1)

        time_diffs = time_diffs / timedelta(minutes=1)
        all_time_diffs.extend( list(time_diffs) )

        max_time_diff = max(all_time_diffs)
        if max_time_diff > INTERPOLATION_LIMIT:
            continue

        # interpolate
        group_df = group_df\
            .resample('15s')\
            .median()

        if group_df.dropna().shape[0] < 60:
            continue

        missing_count += np.sum( group_df.isnull() )

        group_df = group_df\
            .interpolate('pchip')\
            .reset_index()

        # To correct interpolation errors
        # https://numpy.org/doc/2.1/reference/generated/numpy.clip.html
        group_df['Heading'] = np.clip( group_df['Heading'], 0, 360 )
        group_df['COG'] = np.clip( group_df['COG'], 0, 360 )

        # Source: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe
        group_df['hour'] = group_df['# Timestamp'].dt.round('h')

        # merge with meteorological data
        group_df = group_df.apply(
            lambda row: merge_with_meteo(row, meteo_df, lat_lon_grid, lat_lon_tree),
            axis='columns'
        )
        group_df = group_df.drop(columns=['hour']) 

        # create features such as dif and delta dif
        group_df = feature_engineering(group_df)
        group_df.dropna(inplace=True)

        group_df['MMSI'] = int(mmsi)
        group_df['day'] = day
        dfs.append( group_df )

    df = pd.concat( dfs, ignore_index=True )
    df.to_csv('data/df-prepared.csv', index=False)

    print('Interpolated percentage:', missing_count / df.shape[0] * 100)
    np.save('output/gaps.npy', np.array(all_time_diffs))

    min_max_scale_cols = [
        'SOG',
        'currentSpeed',
        'gust',
        'swellHeight',
        'waveHeight',
        'windSpeed',
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    df[min_max_scale_cols] = scaler.fit_transform(
        df[min_max_scale_cols].values
    )

    print(df.max())
    print(df.min())

    # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
    with open('output/min-max-scaler.pkl', 'wb+') as f:
        pickle.dump( scaler, f)

    # Prepare time series and concatenate
    # initialize arrays as null
    X_train, y_train, X_val, y_val, X_test, y_test = [None] * 6

    total_mmsis_final = 0

    train_days, val_days, test_days = partition_days( df['day'].unique() )
    print( train_days, val_days, test_days )
    train_counts, val_counts, test_counts = 0, 0, 0

    for (mmsi, day), mmsi_df in df.groupby(['MMSI', 'day']):

        mmsi_df.dropna(inplace=True)
        mmsi_df.drop( columns=['MMSI', 'day'], inplace=True )

        if SMOOTHING:
            mmsi_df = mmsi_df\
                .rolling(ROLLING_WINDOW_SIZE)\
                .mean()

            mmsi_df.dropna(inplace=True)

        X_arr, y_arr = prepare_time_series(mmsi_df)

        if not X_arr.shape[0] >= 50:
            continue

        total_mmsis_final += 1

        if day in train_days:

            train_counts += 1

            if X_train is None:
                X_train, y_train = X_arr, y_arr
            else:
                X_train = np.concatenate([X_train, X_arr], axis=0)
                y_train = np.concatenate([y_train, y_arr], axis=0)

        elif day in test_days:

            val_counts += 1

            if X_val is None:
                X_val, y_val = X_arr, y_arr
            else:
                X_val = np.concatenate([X_val, X_arr], axis=0)
                y_val = np.concatenate([y_val, y_arr], axis=0)

        else:

            test_counts += 1

            if X_test is None:
                X_test, y_test = X_arr, y_arr
            else:
                X_test = np.concatenate([X_test, X_arr], axis=0)
                y_test = np.concatenate([y_test, y_arr], axis=0)

    name_appendix = 'COGandDifoutput'

    print( X_train.min() )
    print( X_train.max() )
    print( y_train.min() )
    print( y_train.max() )

    np.save(f'X_train_{name_appendix}.npy', X_train)
    np.save(f'y_train_{name_appendix}.npy', y_train)
    np.save(f'X_val_{name_appendix}.npy', X_val)
    np.save(f'y_val_{name_appendix}.npy', y_val)
    np.save(f'X_test_{name_appendix}.npy', X_test)
    np.save(f'y_test_{name_appendix}.npy', y_test)

    print('Total tracks:', total_mmsis_final)

    print('Shapes of tensors:')
    print('\nTrain:')
    print(X_train.shape)
    print(y_train.shape)
    print('Number of ships:', train_counts)

    print('\nVal:')
    print(X_val.shape[0])
    print('Number of ships:', val_counts)

    print('\nTest:')
    print(X_test.shape[0])
    print('Number of ships:', test_counts)

    print('Total sequences:', X_train.shape[0] + X_val.shape[0] + X_test.shape[0])
