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

import pandas as pd

# Apibrėžiame du regionus
LAT_MIN_YIPENG, LAT_MAX_YIPENG = 54.5, 56.5
LON_MIN_YIPENG, LON_MAX_YIPENG = 15.0, 18.0
LAT_MIN_ROSTOCK, LAT_MAX_ROSTOCK = 54.08, 54.35
LON_MIN_ROSTOCK, LON_MAX_ROSTOCK = 11.86, 12.22

COLUMNS = [
    '# Timestamp',
    'MMSI',
    'Latitude',
    'Longitude',
    'Navigational status',
    'ROT',
    'SOG',
    'COG',
    'Heading'
]

dates = pd.date_range(
    start=date(2026,1,1),
    end=date(2026,3,1)
)
file_names = [
    f"aisdk-{data_date.to_pydatetime().strftime('%Y-%m-%d')}"
    for data_date in dates
]

print(file_names)

for file_name in file_names:
    response = requests.get(f'http://aisdata.ais.dk/{file_name}.zip')
    bytes_content = BytesIO( response.content )
    zip_file = zipfile.ZipFile(bytes_content)

    zip_file.extractall()

    del bytes_content
    gc.collect()

    df = pd.read_csv(
        f'{file_name}.csv',
        chunksize=2**20
    )

    os.remove(f'{file_name}.csv')

    for ix, df_chunk in enumerate(df):
        df_chunk[
            (df_chunk['Latitude'].between(LAT_MIN_YIPENG, LAT_MAX_YIPENG)) &
            (df_chunk['Longitude'].between(LON_MIN_YIPENG, LON_MAX_YIPENG)) &
            (df_chunk['Ship type'] == 'Cargo') &
            ((df_chunk['Navigational status'] == 'Under way using engine') |
            (df_chunk['Navigational status'] == 'Under way sailing'))
        ]\
        [COLUMNS]\
        .to_csv(f'data/yipeng/{file_name}-{ix}.csv')

        df_chunk[
            (df_chunk['Latitude'].between(LAT_MIN_ROSTOCK, LAT_MAX_ROSTOCK)) &
            (df_chunk['Longitude'].between(LON_MIN_ROSTOCK, LON_MAX_ROSTOCK)) &
            (df_chunk['Ship type'] == 'Cargo') &
            ((df_chunk['Navigational status'] == 'Under way using engine') |
            (df_chunk['Navigational status'] == 'Under way sailing'))
        ]\
        [COLUMNS]\
        .to_csv(f'data/rostock/{file_name}-{ix}.csv')

    del df
    gc.collect()
