import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.window import Window as W
import itertools as it
from pyspark.ml.recommendation import ALS
from lightfm import LightFM
#from scipy.sparse import csr_matrix
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset

def pre_process(spark):
    # interactions_small = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    # interactions_small.createOrReplaceTempView("interactions")

    # grouped = interactions_small.groupBy("user_id", "recording_msid").agg(F.count("*").alias("frequency"))

    # selected_users = grouped.select("user_id").distinct()
    # grouped = grouped.join(selected_users, on = "user_id", how = "inner")
    # fractions = {user_id: 0.8 for user_id in selected_users.rdd.map(lambda x: x[0]).collect()}
    # train = grouped.sampleBy("user_id", fractions, seed = 42)
    # val = grouped.subtract(train)
   
    # train.write.parquet("train_all_processed", mode = "overwrite")
    # val.write.parquet("val_all_processed", mode = "overwrite")
    #train_count = train.groupBy("user_id").agg(F.countDistinct("recording_msid").alias("train_count"))
    #val_count = val.groupBy("user_id").agg(F.countDistinct("recording_msid").alias("val_count"))
    #result = train_count.join(val_count, on = "user_id", how = "inner")
    #result = result.withColumn("total_count", F.col("train_count") + F.col("val_count"))
    #result.sort(F.desc("total_count")).show()
    
    #train.show()
    size = "all"
    fraction = 1.0
    if size == 'small':
        interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    elif size == 'all':
        interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    else:
        raise ValueError('Invalid dataset size') 
    interactions.createOrReplaceTempView("interactions")
    # print('Dataset {}: '.format(size))
    # shape = spark.sql('select count(distinct user_id) as num_users, count(distinct recording_msid) as num_tracks, count(*) as num_interactions from interactions')
    # shape.show()

     # filter out songs with too few interactions and users with too few interactions 
    # query1 = '''
    # select recording_msid from interactions
    # group by recording_msid
    # having count(distinct user_id)>5
    # '''
    # query2 = '''
    # select user_id from interactions
    # group by user_id
    # having count(distinct recording_msid)>5
    # '''
    # msid_filtered = spark.sql(query1)
    # msid_filtered.createOrReplaceTempView("msid_filtered")
    # users_filtered = spark.sql(query2)
    # users_filtered.createOrReplaceTempView("users_filtered")

    msid_filtered = interactions.groupBy("recording_msid")\
        .agg(F.countDistinct("user_id").alias("distinct_users"))\
        .filter("distinct_users > 5").select("recording_msid")
    
    users_filtered = interactions.groupBy("user_id")\
        .agg(F.countDistinct("recording_msid").alias("distinct_msid"))\
        .filter("distinct_msid > 5").select("user_id")


    # query = '''
    # select i.*
    # from interactions i
    # join msid_filtered m on i.recording_msid = m.recording_msid
    # join users_filtered u on i.user_id = u.user_id
    # '''
    # interactions_filter = spark.sql(query)
    # interactions_filter.createOrReplaceTempView("interactions_filter")

    interactions_filter = interactions.join(msid_filtered, "recording_msid").join(users_filtered, "user_id")


    # print('Dataset {} after preprocessing: '.format(size))
    # shape = spark.sql('select count(distinct user_id) as num_users, count(distinct recording_msid) as num_tracks, count(*) as num_interactions from interactions_filter')
    # shape.show()

    # group by user_id, msid
    grouped = interactions_filter.groupBy("user_id", "recording_msid").agg(F.count("*").alias("frequency"))
    selected_users = grouped.select("user_id").distinct().sample(fraction, seed=42)      
    #use grouped.select("user_id").distinct() for all_users
    grouped = grouped.join(selected_users, on="user_id", how="inner")
    fractions = {user_id: 0.8 for user_id in selected_users.rdd.map(lambda x: x[0]).collect()}
    train = grouped.sampleBy("user_id", fractions, seed=42)
    val = grouped.subtract(train)

    train.write.parquet("hdfs:/user/ml8457_nyu_edu/final_project/train_{}_{}_processed.parquet".format(size, fraction), mode="overwrite")
    val.write.parquet("hdfs:/user/ml8457_nyu_edu/final_project/val_{}_{}_processed.parquet".format(size, fraction), mode="overwrite")

def main(spark, is_baseline = False):
    #read data
    train_interaction_small_processed_path = "hdfs:/user/ml8457_nyu_edu/train_all_processed"
    train_interaction = spark.read.parquet(train_interaction_small_processed_path)
    train_interaction.createOrReplaceTempView("train_interaction")

    val_interaction_small_processed_path = "hdfs:/user/ml8457_nyu_edu/val_all_processed"
    val_interaction = spark.read.parquet(val_interaction_small_processed_path)
    val_interaction.createOrReplaceTempView("val_interaction")


    test_interaction = spark.read.parquet('hdfs:/user/ml8457_nyu_edu/test_processed.parquet')
    test_interaction.createOrReplaceTempView("test_interaction")

    
    if is_baseline:
        damping_factors = list(range(0, 101, 10)) 
        meanAP_res = []
        for damping_factor in damping_factors:
            train_interaction_pop = train_interaction.groupBy("recording_msid")\
                .agg((F.sum("frequency") / (F.count('*') + damping_factor)).alias("avg_frequency"), (F.count('*')).alias("num_counts"))\
                .select('recording_msid', 'avg_frequency', 'num_counts')\
                .sort(F.desc('avg_frequency'))
            #Top 100 most popular msid
            prediction_100 = train_interaction.select("recording_msid").limit(100).rdd.map(lambda x: x[0]).collect()

            predictionAndLabels = val_interaction.groupBy("user_id")\
                .agg((F.collect_list("recording_msid")).alias("msid_list"))\
                .withColumn("prediction", F.array([F.lit(x) for x in prediction_100]))\
                #.limit(1)

            predictionAndLabels = predictionAndLabels.select("prediction", "msid_list").rdd.map(tuple).collect()

            predictionAndLabels = spark.sparkContext.parallelize(predictionAndLabels)
            meanAP_res.append(evaluate(predictionAndLabels))

        dict_meanAP = dict(zip(damping_factors, meanAP_res))
        print(dict_meanAP)
        print("the highest meanAP is ",  max(dict_meanAP.items(), key=lambda x: x[1]))


    else:
        percent = 1 #choose 100% user_id
        user_id_df = train_interaction.select("user_id").distinct()
        user_id_df = user_id_df.limit(int(percent * user_id_df.count()))\
            .select("user_id").rdd.flatMap(lambda x: x).collect()
        train_interaction = train_interaction.filter(F.col("user_id").isin(user_id_df))
        val_interaction = val_interaction.filter(F.col("user_id").isin(user_id_df))
        test_interaction = test_interaction.filter(F.col("user_id").isin(user_id_df))
        
        print("filtered: ")
        train_interaction.show()

        num_partitions = 10
        par_train_interaction = train_interaction.repartition(num_partitions, "user_id")
        par_val_interaction = val_interaction.repartition(num_partitions, "user_id")
        par_test_interaction = test_interaction.repartition(num_partitions, "user_id")

        window_spec = W.orderBy("recording_msid")
        train = par_train_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()
        val = par_val_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()
        test = par_test_interaction.withColumn("recording_msid_numeric", F.dense_rank().over(window_spec)).select("user_id", "recording_msid_numeric", "frequency").cache()

        user_subset = val.select("user_id").distinct()
        ground_truth = val.groupBy("user_id").agg(F.collect_set("recording_msid_numeric").alias("ground_truth"))

        test_truth = test.groupBy("user_id").agg(F.collect_set("recording_msid_numeric").alias("test_truth"))

        print("window pass, percentage of users = ", percent)

        best_meanAP = float('inf')
        best_model = None

        for rank in [20, 50]:
            for reg in [0.1, 1]:
                for alpha in [0.01, 0.1]:
                    num_partitions = 8
                    train = train.repartition(num_partitions, "user_id")
                    val = val.repartition(num_partitions, "user_id")
                    test = test.repartition(num_partitions, "user_id")
                    als = ALS(userCol='user_id', 
                              itemCol='recording_msid_numeric',
                              ratingCol='frequency', 
                              rank=rank, regParam=reg, alpha=alpha, 
                              maxIter=2, coldStartStrategy='drop', implicitPrefs=True)
                    model = als.fit(train)
                    print("als pass, rank = ", rank, "reg = ", reg, "alpha = ", alpha)
                    prediction = model.recommendForUserSubset(user_subset, 100)
                    print("prediction pass")
                    
                    prediction = prediction.alias("prediction")
                    combined = prediction.join(ground_truth, on="user_id", how="inner")\
                        .select("recommendations.recording_msid_numeric", "ground_truth")
                    print("combined join pass")
                    combined_rdd = combined.rdd.map(lambda row: (row['recording_msid_numeric'], row['ground_truth']))
                    
                    metrics = RankingMetrics(combined_rdd)
                    meanAP = metrics.meanAveragePrecisionAt(100)
                    if meanAP > best_meanAP:
                        best_meanAP = meanAP
                        best_model = model
                        print("new best model with meanAP at ", best_meanAP, 
                              " is rank = ", rank, "reg = ", reg, "alpha = ", alpha)
                        
                        combined = prediction.join(test_truth, on="user_id", how="inner")\
                            .select("recommendations.recording_msid_numeric", "test_truth")
                        combined_rdd = combined.rdd.map(lambda row: (row['recording_msid_numeric'], row['test_truth']))
                        metrics = RankingMetrics(combined_rdd)
                        meanAP_test = metrics.meanAveragePrecisionAt(100)
                        print("test accuracy MeanAP is", meanAP_test)
                    


        # Print ranking metrics
        # print("Mean Average Precision at k = 100:", metrics.meanAveragePrecisionAt(100))
        # print("Precision at k = 100:", metrics.precisionAt(100))
        # print("NDCG at k = 100:", metrics.ndcgAt(100))
        # print("Mean Average Precision:", metrics.meanAveragePrecision)

def evaluate(predictionAndLabels):
    metrics = RankingMetrics(predictionAndLabels)
    meanAP = metrics.meanAveragePrecisionAt(100) #change it to 100!!!
    return meanAP

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





if __name__ == "__main__":
    spark = SparkSession.builder.appName('ALS_all').getOrCreate()
    #pre_process(spark)
    lightfm(spark)





def pre_process_test(spark):
    interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
    interactions.createOrReplaceTempView("interactions")
    test = interactions.groupBy("user_id", "recording_msid").agg(F.count("*").alias("frequency"))
    test.show()
    test.write.parquet("test_processed", mode = "overwrite")
