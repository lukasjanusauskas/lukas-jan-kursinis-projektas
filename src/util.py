import pandas as pd
import numpy as np
from datetime import datetime, timedelta

LAT_MIN, LAT_MAX = 54.5, 56.5
LON_MIN, LON_MAX = 15.0, 18.0

def get_dates(fname: str) -> list:
    """
    Read dates from a file and return them as a list of datetime objects.
    
    :param fname: The filename containing the dates
    :return: A list of datetime objects
    """
    
    with open(fname, "r") as f:
        dates = [datetime.strptime(line.strip(), "%Y-%m-%d") 
                 for line in f.readlines()] 

    return dates

def get_mmsi_sample(
    fpath_groups: str,
    n_sample: int,
    seed: int = 42
):
    """
    Gets a sample of MMSIs: from each day this script takes n_sample MMSIs.

    :param fpath_groups: Path to the file, where we have the MMSIs and the days.
    """

    # Read and clean the CSV
    groups = pd.read_csv(fpath_groups)

    groups = groups.drop(columns=['Unnamed: 0', 'nth_voyage'])\
        .drop_duplicates()\
        .groupby('day')

    # Pre-define generator for reproducibility
    rng = np.random.default_rng(seed=seed)

    # Take the sample
    mmsi_sample_output = []

    for day, mmsis_in_day in groups:
        mmsis = mmsis_in_day['MMSI'].values
        mmsi_sample = rng.choice(mmsis, size=n_sample)

        mmsi_sample_w_day = [(mmsi, day) for mmsi in mmsi_sample]
        mmsi_sample_output.extend(mmsi_sample_w_day)

    return mmsi_sample_output


def calculate_holdout_splits(
    size_df: int, train_frac: float,
    n_steps_back: int = 8,
    step_size_back: int = 120,
    n_step_forw: int = 8,
    step_size_forw: int = 120
) -> tuple:
    """ 
    A helper function to calculate the start and end indices of train and test sets. 
    
    :param size_df: Number of observations.
    :param n_steps_back, n_step_forw: Number of time steps in the input and target respectively
    :param step_size_back, step_size_forw: The size of time steps in seconds in input and target respectively
    """

    na_back = n_steps_back * step_size_back
    na_forw = n_step_forw * step_size_forw
    non_na_size = size_df - na_back - na_forw

    train_size = int( non_na_size * train_frac )

    return na_back, na_back + train_size, non_na_size + na_back


def check_is_non_na(df: pd.DataFrame) -> bool:
    return np.sum( df.isna().values ) == 0


def get_date_sample(n_choice, seed: int = 42):
    start_date = datetime(2024, 5, 1)
    n_date = 365

    dates = [
      start_date + timedelta(days=i)
      for i in range(n_date)
    ]

    np.random.seed(seed)
    choice = np.random.choice(dates, n_choice, replace=False)

    return choice 
