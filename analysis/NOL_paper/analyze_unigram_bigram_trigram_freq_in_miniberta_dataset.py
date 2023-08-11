import os
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
parser.add_argument('--ngram', type=int, default=2)


if __name__ == "__main__":
    args = parser.parse_args()
    data=args.data
    ngram=int(args.ngram)
    #data = '10M'
    # check if pickle file exists
    if os.path.exists(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_{ngram}.pkl'):
        print('pickle file exists')
        exit()
    else:
        print('pickle file does not exist')
    mini_dataset = load_dataset('/om2/user/ehoseini/MyData/miniBERTa_v2', f'miniBERTa-{data}')
    counts_dictionary={}
        # print what ngram is recorded
    print('ngram: ', ngram)

    counts=Counter()
    for text in tqdm(mini_dataset['train'],total=len(mini_dataset['train'])):
        counts.update(Counter(ngrams(nltk.word_tokenize(text['text']), n=ngram)))
    counts_dictionary[ngram]=counts

    # print the legnth each count
    print(len(counts_dictionary[ngram]))

    # save the dictionary of counts as a pickle file

    with open(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl', 'wb') as handle:
        pickle.dump(counts_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

