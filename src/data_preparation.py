import os
import sys
import pickle
import pandas as pd # type: ignore
import random
import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # type: ignore

from scipy.spatial import cKDTree # type: ignore
from itertools import product

import warnings
from pandas.errors import PerformanceWarning # type: ignore

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Regiono koordinates
from src.duomenu_surinkimas import (
    LAT_MIN_YIPENG, LAT_MAX_YIPENG,
    LON_MIN_YIPENG, LON_MAX_YIPENG
)
from src.meteorological import construct_coordinate_grid

#Konstantos slenkancio lango  

STEP_BACK_DEFAULT = 25
STEP_FORW_DEFAULT = 25

#Ribojimas laikas (valandomis) kuri gali AIS signalo nebuti
INTERPOLATION_LIMIT = 2

# Average number of minutes to which down-sample
DOWN_SAMPLE_MINUTES = 1

# Maximum gap in the series to discard in minutes
GAP_LIMIT = 120

#Sinuso, kosinuso skai2iavimai bus atlikti:
ANGLE_COLS = [
    'COG',
    'currentDirection',
    'swellDirection',
    'waveDirection',
    'windDirection'
]

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

    time = pd.Timestamp( row['hour'] )

    weather_row = weather_df.loc[(time, lat, lon), :]

    data = np.concatenate( 
        [row.values, weather_row.values[0]]
    )

    columns = list(row.index) + list(weather_df.columns)
    return pd.Series(data=data, index=columns)


def feature_engineering(df: pd.DataFrame, return_timestamps: bool = False) -> tuple:

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
    df['COG-dif'] = np.cbrt( df['COG'].diff(1) )

    df['COG-dif'] = df['COG-dif']\
        .rolling(5)\
        .mean()

    df['dif'] = df['dif'] / 180
    df['delta_dif'] = np.cbrt( df['dif'].diff(1) / 2 )

    df['delta_dif'] = df['delta_dif']\
        .rolling(5)\
        .mean()

    df.drop(columns=['dif'], inplace=True)
    columns_to_drop = ['# Timestamp', 'MMSI']

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
        for lag in range( 1, n_lags+1 ):
            X_arr[:, lag-1, ix] = df[col].shift(-1 * lag)

    for ix, col in enumerate( outputs ):
        for lag in range( 1, n_lags+1 ):
            y_arr[:, lag-1, ix] = df[col].shift(lag)

    return X_arr[n_lags:-n_lags, :, :], y_arr[n_lags:-n_lags, :, :]


def partition_groups(mmsi_day):

    groups_shuffled = mmsi_day.copy()
    random.shuffle( mmsi_day )

    n = len(mmsi_day)
    n_train = int(0.7 * n)
    n_val = int( 0.85 * n )

    return (
        groups_shuffled[:n_train],
        groups_shuffled[n_train:n_val],
        groups_shuffled[n_val:]
    )


def belongs_in(mmsi_day, group_list):

    mmsi, day = mmsi_day

    for (mmsi_group, day_group) in group_list:
        if mmsi_group == mmsi and day_group == day:
            return True

    return False


def read_dataset(data_src: str):

    files = os.listdir(data_src)

    dfs = []
    for f in files:
        date_info = [ int(i) for i in  f.split('-')[1:4] ]
        if date(*date_info) > date(2026, 2, 1):
            continue

        df_tmp = pd.read_csv(
            f'{data_src}/{f}',
            parse_dates=['# Timestamp'],
            dayfirst=True
        ).drop(columns=['Unnamed: 0'])

        dfs.append( df_tmp.copy() )
        del df_tmp

    return pd.concat(dfs)


if __name__ == "__main__":

    raw_df = read_dataset('data/ais')

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

    print( 'AIS signals', raw_df.shape[0] )

    # Fix SOG
    raw_df.loc[raw_df['SOG'] > 25, 'SOG'] = np.nan

    # We will groupby day also, so that we do not have huge gaps to interpolate
    raw_df['day'] = raw_df['# Timestamp'].dt.normalize()

    grouped_df = raw_df\
        .set_index('# Timestamp')\
        .groupby(['MMSI', 'day'])

    print('GOT HERE')

    dfs = []
    all_time_diffs = []
    above_interpolation_limit = 0
    mmsi_timestamps = {}

    for (mmsi, day), group_df in grouped_df:

        group_df = group_df\
            [['Latitude', 'Longitude', 'SOG', 'COG', 'Heading']]\
            .drop_duplicates()

        # We define a track to be at least 50 measurements,
        #   and does not have a gap bigger than 2 hours
        time_diffs = group_df.reset_index()\
            ['# Timestamp']\
            .diff(1)

        time_diffs = (time_diffs / timedelta(minutes=1)).dropna()
        all_time_diffs.extend( list(time_diffs) )

        # Skip, if the time difference is too big
        max_time_diff = np.max(time_diffs)
        if max_time_diff > GAP_LIMIT:
            continue

        # interpolate
        down_sample_seconds = DOWN_SAMPLE_MINUTES * 60
        group_df = group_df[['Latitude', 'Longitude', 'SOG', 'COG', 'Heading']]\
            .resample(f'{down_sample_seconds}s')\
            .first()

        group_df['MMSI'] = mmsi
        group_df['day'] = day

        # At least 50
        if group_df.dropna().shape[0] < 50:
            continue

        group_df = group_df.dropna().reset_index()

        # Source: https://stackoverflow.com/questions/28773342/truncate-timestamp-column-to-hour-precision-in-pandas-dataframe
        group_df['hour'] = group_df['# Timestamp'].dt.round('h')

        # merge with meteorological data
        group_df = group_df.apply(
            lambda row: merge_with_meteo(row, meteo_df, lat_lon_grid, lat_lon_tree),
            axis='columns'
        )
        group_df = group_df.drop(columns=['hour'])

        # create features such as dif and delta dif
        group_df, timestamps = feature_engineering(group_df, return_timestamps=True)

        # Pasalinti NA ir atnaujinti timestamps pagal tai
        na_mask = group_df.isna().any(axis=1)

        group_df = group_df.loc[~na_mask, :].copy()
        timestamps = timestamps[~na_mask].copy()

        mmsi_timestamps[ (mmsi, day) ] = timestamps

        group_df['MMSI'] = int(mmsi)
        group_df['day'] = day
        dfs.append( group_df )

    df = pd.concat( dfs, ignore_index=True )
    df.drop(columns=['Latitude', 'Longitude'], inplace=True)
    df.to_csv('data/df-prepared.csv', index=False)

    print('Done with initial data preparation')
    np.save('output/gaps.npy', np.array(all_time_diffs))

    min_max_scale_cols = [
        # 'Latitude',
        # 'Longitude',
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

    # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
    with open('output/min-max-scaler.pkl', 'wb+') as f:
        pickle.dump( scaler, f)

    print(df.columns)

    # Prepare time series and concatenate
    # initialize arrays as null
    X_train, y_train, X_val, y_val, X_test, y_test = [None] * 6

    total_mmsis_final = 0

    train_groups, val_groups, test_groups = partition_groups( 
        df[['MMSI', 'day']].drop_duplicates().to_numpy().tolist()
    )
    train_counts, val_counts, test_counts = 0, 0, 0

    all_mmsis = []
    all_days = []

    val_mmsis = []
    val_days = []

    for (mmsi, day), mmsi_df in df.groupby(['MMSI', 'day']):

        # Pasalinti NA ir trackinti timestamps
        na_mask = mmsi_df.isna().any(axis=1)
        mmsi_df = mmsi_df.loc[~na_mask, :].copy()

        mmsi_timestamps[(mmsi, day)] = mmsi_timestamps\
            [(mmsi, day)]\
            [~na_mask].copy()

        mmsi_df.drop( columns=['MMSI', 'day'], inplace=True )

        mmsi_timestamps[(mmsi, day)] = mmsi_timestamps\
            [(mmsi, day)]\
            [~na_mask].copy()

        X_arr, y_arr = prepare_time_series(mmsi_df)

        if not X_arr.shape[0] >= 5:
            continue

        total_mmsis_final += 1

        # split the data
        if belongs_in( (mmsi, day), train_groups):

            train_counts += 1

            if X_train is None:
                X_train, y_train = X_arr, y_arr
            else:
                X_train = np.concatenate([X_train, X_arr], axis=0)
                y_train = np.concatenate([y_train, y_arr], axis=0)

        elif belongs_in((mmsi, day), val_groups):

            val_counts += 1

            if X_val is None:
                X_val, y_val = X_arr, y_arr
            else:
                X_val = np.concatenate([X_val, X_arr], axis=0)
                y_val = np.concatenate([y_val, y_arr], axis=0)

            val_mmsis.extend( [mmsi]*X_arr.shape[0] )
            val_days.extend( [day]*X_arr.shape[0] )

        else:

            test_counts += 1

            if X_test is None:
                X_test, y_test = X_arr, y_arr
            else:
                X_test = np.concatenate([X_test, X_arr], axis=0)
                y_test = np.concatenate([y_test, y_arr], axis=0)

            all_mmsis.extend( [mmsi]*X_arr.shape[0] )
            all_days.extend( [day]*X_arr.shape[0] )

    with open('timestamps-dict.npy', 'wb+') as f:
        pickle.dump(mmsi_timestamps, f)

    np.array(all_mmsis).dump('all-mmsis-test.npy')
    np.array(all_days).dump('all-days-test.npy')

    np.array(val_mmsis).dump('val-mmsis-test.npy')
    np.array(val_days).dump('val-days-test.npy')

    np.save('X_train_final.npy', X_train)
    np.save('y_train_final.npy', y_train)
    np.save('X_val_final.npy', X_val)
    np.save('y_val_final.npy', y_val)
    np.save('X_test_final.npy', X_test)
    np.save('y_test_final.npy', y_test)

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
