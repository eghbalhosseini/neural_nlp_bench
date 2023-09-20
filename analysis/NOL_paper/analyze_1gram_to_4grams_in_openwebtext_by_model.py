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
from glob import glob
if __name__=="__main__":
    #first get the data for each section
    count_all=[]
    for ngram in [1,2,3,4]:
        files=glob(f'/rdma/vast-rdma/vast/evlab/ehoseini/MyData/openwebtext-tokenized/train_count_ngram_{ngram}_data_id_*.txt')
        # sort files based on data_id
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # assert there are 13 files
        assert len(files)==13
        # read all the text files and save it in a list
        all_text=[]
        for file in tqdm(files):
            with open(file,'r') as f:
                read_data=f.read()
                # there are 3 numbers with seperated by comma and \n, extracth them
                read_data=read_data.split(',')
                # convert them to int
                read_data=[int(x) for x in read_data]
                all_text.append(read_data)

        # make a numpy array
        all_text=np.array(all_text).astype('float')
        # split the matrix into 2 based on values in the second column, in the first split values of that are less than 4 and the second split values of that are greater than 4
        split_1=np.where(all_text[:,1]<3)[0]
        split_2=np.where(all_text[:,1]>=3)[0]
        all_text_1=all_text[split_1,:]
        all_text_2=all_text[split_2,:]
        counts_=all_text_1[:, -1]
        # sum up the last colum of all_text_2
        sum_all_text_2=np.sum(all_text_2[:,2]).astype('float')
        count_1=np.concatenate((counts_,np.asarray([sum_all_text_2])))
        count_all.append(count_1)

    # counts length for ngram 2, and data_id 0 : 2069542

    # go throught



    counts_dat=count_all
    counts_dat=np.stack(counts_dat,axis=0).transpose()
    counts_data=np.cumsum(counts_dat,axis=0)
    counts_data=counts_data.transpose()
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(counts_data.shape[0]+1), counts_data.shape[0]+1))
    # skip the first color
    all_col=all_col[0:,:]
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .12, .35))
    x_coords = [ 0.1, 1, 10,100]
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
    ax.set_xticklabels([ '1Gr', '2Gr', '3Gr', '4Gr'], rotation=0)
    ax.set_ylim([1e4, 9e9])
    ax = plt.axes((.24, .4, .12, .35))
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


    ax = plt.axes((.38, .4, .12, .35))
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
    ax.set_ylim([1e4, 9e9])

    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.axes((.52, .4, .12, .35))
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

    ax.set_ylim([1e4, 9e9])
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #fig.show()

    fig.savefig(os.path.join(analysis_dir, f'openwebtext_1gram_2gram_3gram_4gram_counts_by_model.png'),dpi=250, format='png',
                metadata=None,bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'openwebtext_1gram_2gram_3gram_4gram_counts_by_model.eps'),format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)




