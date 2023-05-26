#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode cluster baseline.py all
'''

import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics

def baseline_model(spark, size, damping_factor):
    # load data
    train = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/train_{}_processed.parquet'.format(size))
    val = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/val_{}_processed.parquet'.format(size))    
    item_freq = train.groupby('recording_msid').agg((F.sum('frequency')/(F.count('frequency') + damping_factor)).alias('avg_frequency'))
    
    print('Generating prediction...')
    top100_pred = item_freq.select('recording_msid').head(100)
    top100_pred = [row['recording_msid'] for row in top100_pred]
    user_ids = val.select("user_id").distinct().rdd.flatMap(lambda x: x).collect()
    predictions = []
    for user_id in user_ids:
        predictions.append((user_id, top100_pred))
    pred_df = spark.createDataFrame(predictions, ["user_id", "predictions"])
    
    print('Generating ground truth...')
    ground_truth = val.orderBy('frequency', aescending=0).groupBy("user_id").agg(F.collect_set("recording_msid").alias("ground_truth"))
    combined = ground_truth.join(pred_df, on='user_id', how='inner')
    combined_rdd = combined.select("predictions", "ground_truth").rdd.map(lambda row: (row['predictions'], row['ground_truth']))

    metrics = RankingMetrics(combined_rdd)
    m1 = metrics.meanAveragePrecisionAt(100)
    m2 = metrics.precisionAt(100)
    m3 = metrics.ndcgAt(100)
    return m1, m2, m3

def evaluate(spark, size, damping_factor):
    train = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/train_{}_processed.parquet'.format(size))
    test = spark.read.parquet('hdfs:/user/zg745_nyu_edu/final_project/test_processed.parquet'.format(size))
    item_freq = train.groupby('recording_msid').agg((F.sum('frequency')/(F.count('frequency') + damping_factor)).alias('avg_frequency'))
    top100_pred = item_freq.select('recording_msid').head(100)
    top100_pred = [row['recording_msid'] for row in top100_pred]
    user_ids = test.select("user_id").distinct().rdd.flatMap(lambda x: x).collect()
    predictions = []
    for user_id in user_ids:
        predictions.append((user_id, top100_pred))
    pred_df = spark.createDataFrame(predictions, ["user_id", "predictions"])
    ground_truth = test.orderBy('frequency', aescending=0).groupBy("user_id").agg(F.collect_set("recording_msid").alias("ground_truth"))
    combined = ground_truth.join(pred_df, on='user_id', how='inner')
    combined_rdd = combined.select("predictions", "ground_truth").rdd.map(lambda row: (row['predictions'], row['ground_truth']))
    
    metrics = RankingMetrics(combined_rdd)
    m1 = metrics.meanAveragePrecisionAt(100)
    m2 = metrics.precisionAt(100)
    m3 = metrics.ndcgAt(100)
    print('meanAveragePrecisionAt(100): ', m1)
    print('precisionAt(100): ', m2)
    print('ndcgAt(100): ', m3)
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_popularity').getOrCreate()
    size = sys.argv[1]
    beta_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    m1, m2, m3 = [], [], []

    for damping_factor in beta_list:
        a,b,c = baseline_model(spark, size, damping_factor)
        m1.append(a)
        m2.append(b)
        m3.append(c)
    best_beta = beta_list[np.argmax(np.array(m1))] 
    print('meanAveragePrecisionAt(100): ', m1)
    print('precisionAt(100): ', m2)
    print('ndcgAt(100): ', m3)
    print('best beta: ', beta_list[np.argmax(np.array(m1))])

    print('Testing: ')
    evaluate(spark, size, best_beta)

