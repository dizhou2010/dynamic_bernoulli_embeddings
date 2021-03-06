# -*- coding: utf-8 -*-

import glob
import numpy as np
import pandas as pd
import pickle
import re

# Change this to the name of the folder where your dataset is
dataset_name = 'd_emb_seg'
# Change this to the number of words you want in the vocabulary
V = 5000

files = glob.glob(dataset_name +'/raw/*.txt')
dictionary = {}
count = {}
N = len(files)

for f_number, fn in enumerate(files):
    #print(str(f_number)+" out of "+str(N))
    with open(fn, 'r') as myfile:
        words = myfile.read().replace('\n', ' ').split(' ')
    data = np.zeros(len(words))
    for idx, word in enumerate(words):
        if word not in dictionary:
            count[len(dictionary)] = 1
            dictionary[word] = len(dictionary)
        data[idx] = dictionary[word]
        count[data[idx]] += 1
    np.save(fn.replace('.txt2','.npy'), data.astype('int32'))


#pickle.dump( dictionary, open( 'raw/dict.pkl', "a+" ) )
#pickle.dump( count, open( 'raw/counts.pkl', "a+" ) )

df = pd.DataFrame.from_dict(dictionary, orient='index')
cf = pd.DataFrame.from_dict(count, orient='index')
df.columns = ['idx']
cf.columns = ['cnt']
uni = df.join(cf, on = 'idx')

unig = uni.sort_values(by='cnt', ascending = False).reset_index().reset_index()

unig.columns = ['new_idx', 'word', 'old_idx', 'cnt']

old_idx = unig.old_idx.values

files = glob.glob(dataset_name +'/raw/*.npy')
for f_number, fname in enumerate(files):
    print(str(f_number)+" out of "+ str(N))
    dat = np.load(fname) #dat is an (old) index array
    new_dat = np.zeros_like(dat) + 2*V
    
    # replace old index with new index
    for ni, oi in enumerate(old_idx[:V]):
        new_dat[dat == oi] = ni
    new_dat = new_dat[new_dat < V].astype('int32') # remove words have a new index larger than capped number
    new_fname = fname.replace('raw/','train/')
    np.save(new_fname, new_dat)

unig.head(V).to_csv(dataset_name + '/unigram.txt',header=False, index = False, sep = '\t', columns = ['word', 'new_idx', 'cnt'])

