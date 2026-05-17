from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.util import (
    LAT_MIN, LAT_MAX,
    LON_MIN, LON_MAX
)


if __name__ == "__main__":

    # If you use a different setup - change master and or config options
    spark = SparkSession.builder\
        .config("spark.driver.memory", "10G")\
        .config("spark.executor.memory", "10G")\
        .appName('data-filter-script')\
        .master('local[*]')\
        .getOrCreate()

    df = spark.read\
        .option('header', True)\
        .option('inferSchema', True)\
        .csv('data_src/aisdk-*.csv')

    df = df.filter(
        F.col('Latitude').between(LAT_MIN, LAT_MAX) &
        F.col('Longitude').between(LON_MIN, LON_MAX) &
        (F.col('Ship type') == 'Cargo') &
        ((F.col('Navigational status') == 'Under way using engine') |
        (F.col('Navigational status') == 'Under way sailing'))
    )

    columns_to_keep = [
        '# Timestamp',
        'MMSI',
        'Latitude',
        'Longitude',
        'SOG',
        'COG',
        'Heading',
    ]

    df.select(columns_to_keep)\
        .toPandas()\
        .to_csv('data_src/ais_dataset.csv')