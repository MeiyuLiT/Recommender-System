#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Usage:
    $ pip install lightfm
    $ python lightfm_ext.py
'''

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset

import sys
import pandas as pd
import numpy as np
import time

def lightfm_model(size):
    train = pd.read_parquet('/scratch/zg745/1004_project/data/train_{}_processed_int.parquet'.format(size))
    val = pd.read_parquet('/scratch/zg745/1004_project/data/val_{}_processed_int.parquet'.format(size))
    test = pd.read_parquet('/scratch/zg745/1004_project/data/test_processed_int.parquet')

    train_df = train[['user_id', 'recording_msid_numeric', 'frequency']]
    val_df = val[['user_id', 'recording_msid_numeric', 'frequency']]
    test_df = test[['user_id', 'recording_msid_numeric', 'frequency']]

    unique_users = train_df['user_id'].drop_duplicates()
    selected_users = unique_users.sample(frac=0.5)  # select 50% of users

    train_df = train_df[train_df['user_id'].isin(selected_users)]
    val_df = val_df[val_df['user_id'].isin(selected_users)]

    dataset = Dataset()
    dataset.fit(pd.concat([train_df['user_id'], val_df['user_id'], test_df['user_id']]),
                pd.concat([train_df['recording_msid_numeric'], val_df['recording_msid_numeric'], test_df['recording_msid_numeric']]))
    interactions, freq = dataset.build_interactions((row['user_id'], row['recording_msid_numeric'], row['frequency']) for index, row in train_df.iterrows())
    print("train sparse matrix pass")

    start_time = time.time()

    model = LightFM(loss='warp')
    model.fit(interactions, sample_weight=freq)
    print("model fit pass")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Fitting: ", elapsed_time)

    interactions_val, freq_val = dataset.build_interactions((row['user_id'], row['recording_msid_numeric'], row['frequency']) for index, row in val_df.iterrows())
    print('val sparse matrix pass')

    interactions_test, freq_test = dataset.build_interactions((row['user_id'], row['recording_msid_numeric'], row['frequency']) for index, row in test_df.iterrows())
    print('test sparse matrix pass')

    k = 100  # top k items to recommend
    mapatk = precision_at_k(model, interactions_val, k=k).mean()
    print("Val mean AP is ", mapatk)

    mapatk = precision_at_k(model, interactions_test, k=k).mean()
    print("Test mean AP is ", mapatk)

if __name__ == "__main__":
    size = sys.argv[1]
    lightfm_model(size)
