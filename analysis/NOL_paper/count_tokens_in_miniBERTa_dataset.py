import os
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import jsonlines

if __name__ == "__main__":
    data = '1B'
    mini_dataset = load_dataset('/Users/eghbalhosseini/MyData/miniBERTa_v2', f'miniBERTa-{data}')

    mb_len = len(mini_dataset['train'])
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset_50m = mini_dataset['train']

    dataset_50m = dataset_50m.map(lambda x: tokenizer(x['text']), batched=True)
    # count the number of words and tokens
    num_words_50m = 0
    num_tokens_50m = 0
    for idx, x in tqdm(enumerate(dataset_50m),total=len(dataset_50m)):
        num_tokens_50m += len(x['input_ids'])
        num_words_50m += len(x['text'].split(' '))


    # print number of words and tokens
    print(f'Number of words in {data} dataset: {num_words_50m}')
    print(f'Number of tokens in {data} dataset: {num_tokens_50m}')

    #Number of words in 1M dataset: 1083077
    #Number of tokens in 1M dataset: 1507936

    #Number of words in 10M dataset: 10290746
    #Number of tokens in 10M dataset: 14113827

    #Number of words in 100M dataset: 101205621
    #Number of tokens in 100M dataset: 139503529

    #Number of words in 1B dataset: 101,205,621
    #Number of tokens in 1B dataset: 139503529

    #Number of words in 1B dataset: 977702371
    #Number of tokens in 1B dataset: 1349855240

