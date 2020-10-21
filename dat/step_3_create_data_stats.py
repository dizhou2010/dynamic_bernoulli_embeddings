# -*- coding: utf-8 -*-

import glob
import os
import numpy as np
import pickle
import pandas as pd


# Change this to the name of the folder where your dataset is
dataset_name = 'd_emb_seg'

# Change this to a list of the time slices 
time_slices = pd.date_range('2014-12-01','2020-08-31', freq='MS').strftime("%Y%m").tolist()

#Change this to the number of characters in the file names that should be matched to the timeslice prefix.
# i.e. if you use time_slices = [91, 92, 98, ...] 
#         use prefix_length = 2
# if you use time_slices = [1998, 1999, 2000, 2001]
#         use prefix_length = 4
prefix_length = 6

# Change this to a list of query words you would like the algorithm to print descriptive statistics of (i.e. a trajectory of the learned dynamic embeddings)
query_words = ['美国', '中国', '自由', '民主', '特朗普']


# No need to modify any code below
#######################################################
dat_stats={}
dat_stats['name'] = dataset_name
dat_stats['T_bins'] = time_slices
dat_stats['prefix'] = prefix_length
dat_stats['query_words'] = query_words
T = len(dat_stats['T_bins'])

def count_words(split):
    dat_stats[split] = np.zeros(T)
    files = glob.glob(dataset_name + '/'+ split + '/*.npy')
    for t, i in enumerate(dat_stats['T_bins']):
        dat_files = [f for f in files if int(os.path.basename(f)[:dat_stats['prefix']]) == i]
        for fname in dat_files:
            dat = np.load(fname)
            dat_stats[split][t] += len(dat)

count_words('train')
count_words('test')
count_words('valid')

pickle.dump(dat_stats, open(dataset_name + '/dat_stats.pkl', "ab+" ) )
