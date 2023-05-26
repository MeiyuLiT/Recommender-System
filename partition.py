#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ spark-submit --deploy-mode cluster partition.py size
'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(spark, size):
    # load dataset
    if size == 'small':
        interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    elif size == 'all':
        interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    else:
        raise ValueError('Invalid dataset size') 
    interactions.createOrReplaceTempView("interactions")
    print('Dataset {}: '.format(size))
    shape = spark.sql('select count(distinct user_id) as num_users, count(distinct recording_msid) as num_tracks, count(*) as num_interactions from interactions')
    shape.show()
    
    # filter out songs with too few interactions and users with too few interactions 
    query1 = '''
    select recording_msid from interactions
    group by recording_msid
    having count(distinct user_id)>5
    '''
    query2 = '''
    select user_id from interactions
    group by user_id
    having count(distinct recording_msid)>5
    '''
    msid_filtered = spark.sql(query1)
    msid_filtered.createOrReplaceTempView("msid_filtered")
    users_filtered = spark.sql(query2)
    users_filtered.createOrReplaceTempView("users_filtered")

    query = '''
    select i.*
    from interactions i
    join msid_filtered m on i.recording_msid = m.recording_msid
    join users_filtered u on i.user_id = u.user_id
    '''
    interactions_filter = spark.sql(query)
    print('Dataset {} after preprocessing: '.format(size))
    num_users = interactions_filter.agg(F.countDistinct("user_id").alias("num_users")).collect()[0]["num_users"]
    num_tracks = interactions_filter.agg(F.countDistinct("recording_msid").alias("num_tracks")).collect()[0]["num_tracks"]
    num_interactions = interactions_filter.count()
    print(f"Number of distinct users: {num_users}")
    print(f"Number of distinct tracks: {num_tracks}")
    print(f"Number of interactions: {num_interactions}")
    
    # group by user_id, msid
    grouped = interactions_filter.groupBy("user_id", "recording_msid").agg(F.count("*").alias("frequency"))
    selected_users = grouped.select("user_id").distinct().sample(0.5, seed=42)   # use grouped.select("user_id").distinct() for all_users
    grouped = grouped.join(selected_users, on="user_id", how="inner")
    fractions = {user_id: 0.8 for user_id in selected_users.rdd.map(lambda x: x[0]).collect()}
    train = grouped.sampleBy("user_id", fractions, seed=42)
    val = grouped.subtract(train)
    
    train.write.parquet("hdfs:/user/zg745_nyu_edu/final_project/train_{}_processed.parquet".format(size), mode="overwrite")
    val.write.parquet("hdfs:/user/zg745_nyu_edu/final_project/val_{}_processed.parquet".format(size), mode="overwrite")

    print("Dataset {} train count: ".format(size), train.count())
    print("Dataset {} validation count: ".format(size), val.count())

    train_count = train.groupBy("user_id").agg(F.countDistinct("recording_msid").alias("train_count"))
    val_count = val.groupBy("user_id").agg(F.countDistinct("recording_msid").alias("val_count"))
    result = train_count.join(val_count, on="user_id", how="inner")
    result = result.withColumn("total_count", F.col("train_count") + F.col("val_count"))
    result.sort(F.desc("total_count")).show()

def test_process(spark):
    interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    interactions.createOrReplaceTempView("interactions")
    print('Test Dataset: ')
    shape = spark.sql('select count(distinct user_id) as num_users, count(distinct recording_msid) as num_tracks, count(*) as num_interactions from interactions')
    shape.show()

    # group by user_id, msid
    grouped = interactions.groupBy("user_id", "recording_msid").agg(F.count("*").alias("frequency"))
    selected_users = grouped.select("user_id").distinct()     
    grouped = grouped.join(selected_users, on="user_id", how="inner")
    grouped.write.parquet("hdfs:/user/zg745_nyu_edu/final_project/test_processed.parquet", mode="overwrite")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('load_partition').getOrCreate()
    size = sys.argv[1]
    main(spark, size)
    # test_process(spark)
