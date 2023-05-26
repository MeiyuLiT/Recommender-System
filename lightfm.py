from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset

import pandas as pd
import numpy as np

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.window import Window as W
import itertools as it

def lightfm(spark):
    train_interaction_small_processed_path = "hdfs:/user/ml8457_nyu_edu/train_small_processed"
    train_interaction = spark.read.parquet(train_interaction_small_processed_path)
    train_interaction.createOrReplaceTempView("train_interaction")

    val_interaction_small_processed_path = "hdfs:/user/ml8457_nyu_edu/val_small_processed"
    val_interaction = spark.read.parquet(val_interaction_small_processed_path)
    val_interaction.createOrReplaceTempView("val_interaction")

    test_interaction = spark.read.parquet('hdfs:/user/ml8457_nyu_edu/test_processed.parquet')
    test_interaction.createOrReplaceTempView("test_interaction")

    percent = 0.01 #choose 1% user_id
    user_id_df = train_interaction.select("user_id").distinct()
    user_id_df = user_id_df.limit(int(percent * user_id_df.count()))\
        .select("user_id").rdd.flatMap(lambda x: x).collect()
    train_interaction = train_interaction.filter(F.col("user_id").isin(user_id_df))
    val_interaction = val_interaction.filter(F.col("user_id").isin(user_id_df))
    print("downsample pass")

    num_partitions = 10
    par_train_interaction = train_interaction.repartition(num_partitions, "user_id")
    par_val_interaction = val_interaction.repartition(num_partitions, "user_id")
    par_test_interaction = test_interaction.repartition(num_partitions, "user_id")

    window_spec = W.orderBy("recording_msid")
    train = par_train_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()
    val = par_val_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()
    test = par_test_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()
    print("window pass")

#LightFM starts ------
    train_df = train.select('user_id', 'recording_msid_numeric').toPandas()
    val_df = val.select('user_id', 'recording_msid_numeric').toPandas()
    test_df = test.select('user_id', 'recording_msid_numeric').toPandas()

    dataset = Dataset()
    dataset.fit(train_df['user_id'], train_df['recording_msid_numeric'])
    interactions, weights = dataset.build_interactions((train_df['user_id'][i], train_df['recording_msid_numeric'][i]) for i in range(len(train_df)))
    print("train sparse matrix pass")

    model = LightFM(loss='warp')
    model.fit(interactions)
    print("model fit pass")

    dataset = Dataset()
    dataset.fit(val_df['user_id'], val_df['recording_msid_numeric'])
    interactions_val, weights_val = dataset.build_interactions((val_df['user_id'][i], val_df['recording_msid_numeric'][i]) for i in range(len(val_df)))
    print('val sparse matrix pass')

    dataset = Dataset()
    dataset.fit(test_df['user_id'], test_df['recording_msid_numeric'])
    interactions_test, weights_test = dataset.build_interactions((test_df['user_id'][i], test_df['recording_msid_numeric'][i]) for i in range(len(test_df)))
    print('test sparse matrix pass')

    k = 100  # top k items to recommend
    mapatk = precision_at_k(model, interactions_val, k=k).mean()
    print("Val mean AP is ", mapatk)

    mapatk = precision_at_k(model, interactions_test, k=k).mean()
    print("Test mean AP is ", mapatk)
