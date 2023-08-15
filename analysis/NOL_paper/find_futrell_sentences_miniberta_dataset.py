import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import jsonlines
from collections import Counter
from nltk.util import ngrams
import nltk
import json
import pickle
# get data id as avariable from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='10M')
parser.add_argument('--chunk_id', type=int, default=0)
import re
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
def search_string_in_example(example, search_string):
    # make sure that example is not empty
    if isinstance(example['text'], (str, bytes)):
        if isinstance(example, bytes):
            example['text'] = example['text'].decode('utf-8')
        return dict(occur=[bool(re.search(search_string, example['text']))])
    else:
        return dict(occur=[False])

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

if __name__ == "__main__":
    args = parser.parse_args()
    data=args.data
    chunk_id=int(args.chunk_id)
    mini_dataset = load_dataset('/om2/user/ehoseini/MyData/miniBERTa_v2', f'miniBERTa-{data}')
    indices = np.arange(len(mini_dataset['train']))
    indices_split = np.array_split(indices, 50)
    mini_dataset_sample = mini_dataset['train'].select(indices_split[chunk_id])

    file_path = f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/proc_data_train_for_{data}.raw'
    Futrell_stim=pd.read_csv('/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/Futrell2018-stimulus_set.csv')
    Futrell_sentence=[]
    for idx, x in Futrell_stim.groupby('sentence_id'):
        # sort the panda by word_within_sentence_id
        x=x.sort_values(by=['word_within_sentence_id'])
        # get the sentence
        sentence=' '.join(x['word'].tolist())
        Futrell_sentence.append(sentence)

    string_occurance = []
    for search_string in tqdm(Futrell_sentence, total=len(Futrell_sentence)):
        occurance = mini_dataset_sample.map(lambda example: search_string_in_example(example, search_string),
                                            remove_columns='text', batched=False, num_proc=10, keep_in_memory=False,
                                            load_from_cache_file=False)
        string_occurance.append(np.sum(occurance['occur']))

    print(f'total occurance: {np.sum(string_occurance)}')

    with open(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/Futrell_occurance_chunk_{chunk_id}.pkl',
              'wb') as handle:
        pickle.dump(string_occurance, handle, protocol=pickle.HIGHEST_PROTOCOL)



