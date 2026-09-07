"""
Šiame faile surenku duomenis iš Danish Maritime Authority ir iš karto atrenku
Dalis kodo paimta iš: https://discuss.python.org/t/trying-to-scrape-and-download-zipfiles/35399/2
"""

import requests
import gc
import zipfile
from io import BytesIO
import os
from datetime import date
from tqdm import tqdm

import pandas as pd

# Apibrėžiame du regionus
LAT_MIN_YIPENG, LAT_MAX_YIPENG = 54.5, 56.5
LON_MIN_YIPENG, LON_MAX_YIPENG = 13.0, 18.0

COLUMNS = [
    '# Timestamp',
    'MMSI',
    'Latitude',
    'Longitude',
    'Navigational status',
    'ROT',
    'SOG',
    'COG',
    'Heading',
    'Cargo type',
    'Width',
    'Length',
]

dates = pd.date_range(
    start=date(2024,11,15),
    end=date(2024,12,15)
)
dates = [
    data_date.to_pydatetime()
    for data_date in dates
]

if __name__ == '__main__':
    for date_ in tqdm(dates):

        url_dir = date_.strftime('%Y')
        file_name = f"aisdk-{date_.strftime('%Y-%m-%d')}"

        response = requests.get(f'http://aisdata.ais.dk/{url_dir}/{file_name}.zip')
        bytes_content = BytesIO( response.content )

        if response.status_code != 200:
            print(response.url)

        with zipfile.ZipFile(bytes_content) as zip_file:

            zip_file.extractall()
            del bytes_content
            gc.collect()

        print(file_name)

        df = pd.read_csv(
            f'{file_name}.csv',
            chunksize=2**20
        )

        os.remove(f'{file_name}.csv')

        for ix, df_chunk in enumerate(df):
            df_tmp = df_chunk[
                (df_chunk['Latitude'].between(LAT_MIN_YIPENG, LAT_MAX_YIPENG)) &
                (df_chunk['Longitude'].between(LON_MIN_YIPENG, LON_MAX_YIPENG)) &
                ((df_chunk['Ship type'] == 'Cargo') | (df_chunk['Ship type'] == 'Tanker') ) &
                ((df_chunk['Navigational status'] == 'Under way using engine') |
                (df_chunk['Navigational status'] == 'Under way sailing'))
            ][COLUMNS]

            df_tmp['Cargo type'].fillna( 'No additional information', inplace=True )

            df_tmp.to_csv(f'data/ais/{file_name}-{ix}.csv')

        del df
        gc.collect()
