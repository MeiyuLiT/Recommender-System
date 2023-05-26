#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode cluster convert_string.py size
'''

from pyspark.sql import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window as W

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def conversion(spark, size):
    train = spark.read.parquet('/scratch/zg745/1004_project/data/train_{}_processed.parquet'.format(size))
    val = spark.read.parquet('/scratch/zg745/1004_project/data/val_{}_processed.parquet'.format(size))
    test = spark.read.parquet('/scratch/zg745/1004_project/data/test_processed.parquet')

    # process the data, ready for lightFM
    window_spec = W.orderBy("recording_msid")
    train = train.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()
    val = val.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()
    test = test.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()

    train.write.parquet("/scratch/zg745/1004_project/data/train_{}_processed_int.parquet".format(size), mode="overwrite")
    val.write.parquet("/scratch/zg745/1004_project/data/val_{}_processed_int.parquet".format(size), mode="overwrite")
    test.write.parquet("/scratch/zg745/1004_project/data/test_processed_int.parquet", mode="overwrite")


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('convert_string_int').getOrCreate()
    size = sys.argv[1]
    conversion(spark, size)
