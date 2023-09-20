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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='10M')
parser.add_argument('--ngram', type=int, default=2)
parser.add_argument('--chunk_id', type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    data=args.data
    ngram=int(args.ngram)
    #data = '10M'
    # check if pickle file exists
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    mini_dataset = load_from_disk('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/miniBERTa_v2/miniBERTa-10M-crunched/train_context_len_1024/')
    # create a np.arange of indices from 0 to len(mini_dataset['train'])
    text_list_minberta = []
    for x in tqdm(mini_dataset, total=len(mini_dataset)):
        text_list_minberta.append(tokenizer.decode(x['input_ids']))

    # do steps of 500K
    step=1024*500
    # go from 0 to length of text_list_minberta with step of 500K
    indices=np.arange(0,len(text_list_minberta)*1024,step)
    # divide indicies bu 1024
    indices=np.divide(indices,1024).astype(int)

    # split it to 50 chuncks
    counts_per_bucket=[]
        # print what ngram is recorded
    for chunk_id in range(len(indices)-1):
        indices_split=np.arange(indices[chunk_id],indices[chunk_id+1])
        mini_dataset_sample = [text_list_minberta[x] for x in indices_split]
        counts=Counter()
        for text in tqdm(mini_dataset_sample,total=len(mini_dataset_sample)):
            counts.update(Counter(ngrams(nltk.word_tokenize(text), n=ngram)))
        counts_per_bucket.append(counts)
        #counts_dictionary[ngram]=counts
    cumulative_count=[]
    cumulative_count.append(len(counts_per_bucket[0]))
    for k in range(len(counts_per_bucket)):
        counts_total=Counter()
        [counts_total.update(counts_per_bucket[i]) for i in range(k-1)]
        counts_k=counts_per_bucket[k]
        print(len(counts_k-counts_total))
        cumulative_count.append(len(counts_k-counts_total))

    # print the legnth each count
    openwebtext_dataset = load_from_disk(
        '/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext-crunched/train_context_len_1024/')
    # randomize the dataset
    openwebtext_dataset = openwebtext_dataset['train'].shuffle(seed=42)
    # make a tokenizer for gpt2
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    #openwebtext_dataset = load_from_disk('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext/train')
    line_01 = int(len(openwebtext_dataset) * (0.25 / 100))
    step = 1024 * 500

    # go from 0 to length of text_list_minberta with step of 500K
    indices = np.arange(0, line_01 , 500)
    #indices = np.divide(indices, 1024).astype(int)


    openwebtext_dataset_sample = openwebtext_dataset.select(np.arange(line_01))
    #openwebtext_dataset_sample = openwebtext_dataset.select(indices_01)
    text_list = []
    for x in tqdm(openwebtext_dataset_sample, total=len(openwebtext_dataset_sample)):
        text_list.append(tokenizer.decode(x['input_ids']))

    # print what ngram is recorded
    # print what ngram is recorded
    counts_per_bucket_ow = []
    # print what ngram is recorded
    for chunk_id in range(len(indices) - 1):
        indices_split = np.arange(indices[chunk_id], indices[chunk_id + 1])
        ow_dataset_sample = [text_list[x] for x in indices_split]
        counts = Counter()
        for text in tqdm(ow_dataset_sample, total=len(ow_dataset_sample)):
            counts.update(Counter(ngrams(nltk.word_tokenize(text), n=ngram)))
        counts_per_bucket_ow.append(counts)
    cumulative_count_ow = []
    cumulative_count_ow.append(len(counts_per_bucket_ow[0]))
    for k in range(len(counts_per_bucket_ow)):
        counts_total = Counter()
        [counts_total.update(counts_per_bucket_ow[i]) for i in range(k - 1)]
        counts_k = counts_per_bucket_ow[k]
        print(len(counts_k - counts_total))
        cumulative_count_ow.append(len(counts_k - counts_total))


    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .45, .35))
    cumulative_count_cum=np.cumsum(cumulative_count)
    cumulative_count_cum_ow=np.cumsum(cumulative_count_ow)
    # repeat the last element of cumulative_count_cum 10 times
    cumulative_count_cum=np.concatenate([cumulative_count_cum,np.repeat(cumulative_count_cum[-1],cumulative_count_cum_ow.shape[0]-cumulative_count_cum.shape[0])])
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(cumulative_count_cum.shape[0] + 1), cumulative_count_cum.shape[0] + 1))

    # plot first column of cound data as a line with marker
    x_coords = np.arange(len(cumulative_count_cum)) * 500000
    for idx, scr in enumerate(cumulative_count_cum):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, cumulative_count_cum, color='k', linewidth=2, zorder=1)

    cmap_all=cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(cumulative_count_cum_ow.shape[0] + 1), cumulative_count_cum_ow.shape[0] + 1))
    x_coords = np.arange(len(cumulative_count_cum_ow)) * 500000
    for idx, scr in enumerate(cumulative_count_cum_ow):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, cumulative_count_cum_ow, color='k', linewidth=2, zorder=1)


    #ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # s
    ax.set_xticks([0, 5000000, 10000000, 15000000, 20000000])
    ax.set_xticklabels(['0', '5M', '10M', '15M', '20M'], rotation=0)
    ax.set_yticks([100000, 200000, 300000, 400000, 500000])
    ax.set_yticklabels(['100K', '200K', '300K', '400K', '500K'], rotation=0)
    fig.show()
    # save figure '
    analysis_dir = '/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    fig.savefig(os.path.join(analysis_dir, f'buckets.png'), dpi=250,
                format='png',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'buckets.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)



