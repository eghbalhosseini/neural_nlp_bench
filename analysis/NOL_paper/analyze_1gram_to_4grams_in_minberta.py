import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import jsonlines
from collections import Counter
from nltk.util import ngrams
import nltk
import json
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
def read_pickle_generator(filename, chunk_size):
    with open(filename, 'rb') as file:
        while True:
            try:
                chunk = pickle.load(file)
                yield chunk
            except EOFError:
                break

if __name__=="__main__":
    #first get the data for each section
    data='1M'
    counts_1m=[]
    # for ngram in [1,2,3,4]:
    #     ngram_dat=joblib.load(os.path.join(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl'))
    #     counts_1m.append(len(ngram_dat[ngram]))
    # del ngram_dat
    counts_1m=[71945, 486891, 914257, 1089235]
    # data='10M'
    # counts_10m=[]
    # for ngram in tqdm([1,2,3,4]):
    #     # use with open to avoid memory error
    #     chunk_size=0
    #     for chunk in read_pickle_generator(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl', chunk_size=5000):
    #         chunk_size+=len(chunk[ngram])
    #     #with open(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl', 'rb') as handle:
    #     #    ngram_dat = pickle.load(handle)
    #     counts_10m.append(chunk_size)
    #     del chunk
    counts_10m=[294761, 2835765, 6971527, 9556465]
    data='100M'
    # counts_100m=[]
    # for ngram in tqdm([1,2,3,4]):
    #     # use with open to avoid memory error
    #     chunk_size = 0
    #     for chunk in read_pickle_generator(
    #             f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl',chunk_size=5000):
    #         chunk_size += len(chunk[ngram])
    #     # with open(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl', 'rb') as handle:
    #     #    ngram_dat = pickle.load(handle)
    #     counts_100m.append(chunk_size)
    #     del chunk
    counts_100m=[1374223, 15751418, 49478551, 80132643]
    data='1B'
    # counts_1b=[]
    # for ngram in tqdm([1,2,3,4]):
    #     chunk_size = 0
    #     for chunk_id in tqdm(range(50)):
    #         # use with open to avoid memory error
    #         for chunk in read_pickle_generator(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}_chunk_{chunk_id}.pkl',chunk_size=5000):
    #             chunk_size += len(chunk[ngram])
    #         # with open(f'/om2/user/ehoseini/MyData/miniBERTa_v2/miniBERTa-{data}/counts_dictionary_ngram_{ngram}.pkl', 'rb') as handle:
    #         #    ngram_dat = pickle.load(handle)
    #     counts_1b.append(chunk_size)
    #     del chunk
    # # combine counts
    counts_1b=[21991006, 220941597, 583759656, 849081083]


    counts_data=[counts_1m,counts_10m,counts_100m,counts_1b]
    counts_data=np.stack(counts_data,axis=0)

    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(counts_data.shape[0]+1), counts_data.shape[0]+1))
    # skip the first color
    all_col=all_col[1:,:]
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .11, .35))
    x_coords = [ 0.1, 1, 10, 100]
    count_=counts_data[:,0]
    # plot first column of cound data as a line with marker
    for idx, scr in enumerate(count_):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, count_, color='k', linewidth=2, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks)
    # extend the x lim so that the marker is not on the edge
    ax.set_xlim([0.05, 200])
    ax.set_xticklabels([ '1M', '10M', '100M', '1B'], rotation=0)
    ax.set_ylim([1e4, 9e9])
    ax = plt.axes((.24, .4, .11, .35))
    x_coords = [0.1, 1, 10, 100]
    count_ = counts_data[:, 1]
    # plot first column of cound data as a line with marker
    for idx, scr in enumerate(count_):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, count_, color='k', linewidth=2, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks)
    # extend the x lim so that the marker is not on the edge
    ax.set_xlim([0.05, 200])
    ax.set_xticklabels(['1M', '10M', '100M', '1B'], rotation=0)
    ax.set_ylim([1e4, 9e9])
    # turn off xticks and xticklabels and yticks and yticklabels
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


    ax = plt.axes((.38, .4, .11, .35))
    x_coords = [0.1, 1, 10, 100]
    count_ = counts_data[:, 2]
    # plot first column of cound data as a line with marker
    for idx, scr in enumerate(count_):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, count_, color='k', linewidth=2, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks)
    # extend the x lim so that the marker is not on the edge
    ax.set_xlim([0.05, 200])
    ax.set_ylim([1e4,9e9])
    ax.set_xticklabels(['1M', '10M', '100M', '1B'], rotation=0)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.axes((.52, .4, .11, .35))
    x_coords = [0.1, 1, 10, 100]
    count_ = counts_data[:, 3]
    # plot first column of cound data as a line with marker
    for idx, scr in enumerate(count_):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, count_, color='k', linewidth=2, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks)
    # extend the x lim so that the marker is not on the edge
    ax.set_xlim([0.05, 200])
    ax.set_xticklabels(['1M', '10M', '100M', '1B'], rotation=0)
    ax.set_ylim([1e4, 9e9])
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'miniberta_1gram_2gram_3gram_4gram_counts.png'),dpi=250, format='png',
                metadata=None,bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'miniberta_1gram_2gram_3gram_4gram_counts.eps'),format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)


    token_per_checkpoint = [131072000,262144000,1867776000,60948480000]
    unique_token_per_chekcpoint = [x[0] / x[1] * 100 for x in zip(counts_data[:, 0], token_per_checkpoint)]

    ax = plt.axes((.1, .4, .11, .35))
    x_coords = [0.1, 1, 10, 100]
    count_ = unique_token_per_chekcpoint
    # plot first column of cound data as a line with marker
    for idx, scr in enumerate(count_):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=2)
    ax.plot(x_coords, count_, color='k', linewidth=2, zorder=1)
    ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks)
    # extend the x lim so that the marker is not on the edge
    ax.set_xlim([0.05, 200])
    ax.set_ylim([-0.09, 3])
    ax.set_xticklabels(['0.1', '1.0', '10', '100'], rotation=0)
    # ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], rotation=0)
    ax.set_ylabel('Percentage of unique tokens in total tokens')
    ax.set_xlabel('training step')
    # ax.set_xticks([])
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'miniberta_percentage_unique_in_total.png'), dpi=250, format='png',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'miniberta_percentage_unique_in_total.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
