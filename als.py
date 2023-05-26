#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode cluster als.py all
'''

from pyspark.sql import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window as W

import sys
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')

def als_model(spark, size, rank, reg, alpha):
    # load data
    train = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/train_{}_processed.parquet'.format(size))
    val = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/val_{}_processed.parquet'.format(size))    
    test = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/test_processed.parquet') 

    selected_users = train.select("user_id").distinct().sample(0.5, seed=42)   
    train = train.join(selected_users, on="user_id", how="inner")
    val = val.join(selected_users, on="user_id", how="inner")

    window_spec = W.orderBy("recording_msid")
    train = train.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()
    val = val.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()
    test = test.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).cache()

    print('building model: ')
    als = ALS(userCol='user_id', itemCol='recording_msid_numeric', ratingCol='frequency', 
              rank=rank, regParam=reg, alpha=alpha, maxIter=5, coldStartStrategy='drop', implicitPrefs=True, seed=42)
    start_time = time.time()
    model = als.fit(train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Fitting: ", elapsed_time)

    
    print(size, rank, reg, alpha)
    print('evaluate on val: ')
    user_subset = val.select("user_id").distinct()
    prediction = model.recommendForUserSubset(user_subset, 100)
    prediction = prediction.alias("prediction")
    ground_truth = val.groupBy("user_id").agg(F.collect_set("recording_msid_numeric").alias("ground_truth"))
    combined = prediction.join(ground_truth, on="user_id", how="inner").select("recommendations.recording_msid_numeric", "ground_truth")
    combined_rdd = combined.rdd.map(lambda row: (row['recording_msid_numeric'], row['ground_truth']))
    metrics = RankingMetrics(combined_rdd)
    print(metrics.meanAveragePrecisionAt(100), metrics.precisionAt(100), metrics.ndcgAt(100))
    val_results = metrics.meanAveragePrecisionAt(100)

    print("evaluate on test: ")
    user_subset = test.select("user_id").distinct()
    prediction = model.recommendForUserSubset(user_subset, 100)
    prediction = prediction.alias("prediction")
    ground_truth = test.groupBy("user_id").agg(F.collect_set("recording_msid_numeric").alias("ground_truth"))
    combined = prediction.join(ground_truth, on="user_id", how="inner").select("recommendations.recording_msid_numeric", "ground_truth")
    combined_rdd = combined.rdd.map(lambda row: (row['recording_msid_numeric'], row['ground_truth']))
    metrics = RankingMetrics(combined_rdd)
    print(metrics.meanAveragePrecisionAt(100), metrics.precisionAt(100), metrics.ndcgAt(100))
    
    return val_results

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als_grid_search').getOrCreate()
    size = sys.argv[1]
    rank_list = [20, 50, 100]
    reg_list = [0.1, 1, 10]
    alpha_list = [1, 5, 10]
    best_rank = None
    best_reg = None
    best_alpha = None
    max_map = float("-inf")

    for rank in rank_list:
        for reg in reg_list:
            for alpha in alpha_list:
                print("Training model with rank={}, reg={}, alpha={}".format(rank, reg, alpha))
                map_at_k = als_model(spark, size, rank, reg, alpha)
                if map_at_k > max_map:
                    best_rank = rank
                    best_reg = reg
                    best_alpha = alpha
                    max_map = map_at_k
                    print('NEW best MAP FOUND!!!')
