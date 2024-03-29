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
# get ngram and data_id as variables from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ngram', type=int, default=2)
parser.add_argument('--data_id', type=int, default=0)



if __name__ == "__main__":
    args = parser.parse_args()
    ngram=int(args.ngram)
    data_id=int(args.data_id)
    # check if text file exists
    file_path=f'/om2/user/ehoseini/MyData/openwebtext-tokenized/train_count_ngram_tokenized_{ngram}_data_id_{data_id}.txt'
    if os.path.exists(file_path):
        print('text file exists')
        exit()

    openwebtext_dataset = load_from_disk('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext-crunched/train_context_len_1024/')
    # randomize the dataset
    openwebtext_dataset = openwebtext_dataset['train'].shuffle(seed=42)
    openwebtext_dataset=openwebtext_dataset.flatten_indices()
    # make a tokenizer for gpt2
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # calucluate how many line would correspond to 0.1%,1%,10%,100% of the dataset
    # 0.1% of the dataset
    line_01=int(len(openwebtext_dataset)*(0.1/100))
    # # 1% of the dataset
    line_1=int(len(openwebtext_dataset)*(1/100))
    # # 10% of the dataset
    line_10=int(len(openwebtext_dataset)*(10/100))
    # # 100% of the dataset
    line_100=int(len(openwebtext_dataset)*(100/100))
    # create a list of indices for 0 to line_01, from line_01 to line_1, from line_1 to line_10, from line_10 to line_100, using np.arange
    # 0 to line_01
    indices_01=np.arange(line_01)
    # # line_01 to line_1
    indices_1=np.arange(line_01,line_1)
    # # line_1 to line_10
    indices_10=np.arange(line_1,line_10)
    # # line_10 to line_100
    indices_100=np.arange(line_10,line_100)
    # devide up list 100 to 10 chuncks
    indices_100_split=np.array_split(indices_100,10)
    # make it a list of lists
    indieces_list=[indices_01,indices_1,indices_10]
    # add indices_100 to indieces_list
    indieces_list_1=[indieces_list,indices_100_split]
    # flatten the indieces_list as a new list
    indieces_list_flat = [item for sublist in indieces_list_1 for item in sublist]
    #
    #
    # # split the data to 4 parts from 0 to line_01, from line_01 to line_1, from line_1 to line_10, from line_10 to line_100
    # # 0 to line_01
    # # save the openwebtext_dataset in the following path f'/om2/user/ehoseini/MyData/miniBERTa_v2/openwebtext-tokenized/train_data_id_{data_id}_01' as a text file
    for data_id in range(len(indieces_list_flat)):
        openwebtext_dataset_sample=openwebtext_dataset.select(indieces_list_flat[data_id])
        text_list=[]
        for x in tqdm(openwebtext_dataset_sample,total=len(openwebtext_dataset_sample)):
             text_list.append(tokenizer.decode(x['input_ids']))
        with open(f'/om2/user/ehoseini/MyData/openwebtext/train_tokenized_data_id_{data_id}.txt', 'w') as f:
            for item in tqdm(text_list,total=len(text_list)):
                f.write("%s\n" % item)

    # do map tokenize_decode on the openwebtext_dataset_sample using a lampda function
    text_list=[]
    for x in tqdm(openwebtext_dataset_sample,total=len(openwebtext_dataset_sample)):
        text_list.append(tokenizer.decode(x['input_ids']))

    #save the text_list in the following path f'/om2/user/ehoseini/MyData/miniBERTa_v2/openwebtext-tokenized/train_data_id_{data_id}_01.txt' as a text file
    with open(f'/om2/user/ehoseini/MyData/openwebtext/train_tokenized_data_id_{data_id}.txt', 'w') as f:
        for item in tqdm(text_list,total=len(text_list)):
            f.write("%s\n" % item)

    # read the train_data_id_data_id.text and put each line in a list
    with open(f'/om2/user/ehoseini/MyData/openwebtext/train_data_id_{data_id}.txt', 'r') as f:
        text_list = f.readlines()
    #openwebtext_dataset=openwebtext_dataset.select(indieces_list[data_id])
    # print length of the dataset
    #print(f'dataset length {len(openwebtext_dataset)}\n')
    counts = Counter()
    for text in tqdm(text_list,total=len(text_list)):
        #text=tokenizer.decode(token_ids=x['input_ids'])
        counts.update(Counter(ngrams(nltk.word_tokenize(text), n=ngram)))
    # print length of counts

    print(f'counts length for ngram {ngram}, and data_id {data_id}, tokenized : {len(counts)}\n')

    # create a csv with 3 columns (ngram,dataset_id,len(count)
    # save the csv file in the following path f'/om2/user/ehoseini/MyData/miniBERTa_v2/openwebtext-tokenized/train_count_ngram_{ngram}_data_id_{data_id}.txt'
    with open(f'/om2/user/ehoseini/MyData/openwebtext-tokenized/train_count_ngram_tokenized_{ngram}_data_id_{data_id}.txt', 'w') as f:
        f.write("%s,%s,%s\n"%(ngram,data_id,len(counts)))




