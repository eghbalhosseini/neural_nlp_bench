import pickle as pkl
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#%%
import getpass
import sys
import pickle
import xarray as xr
from glob import glob
user=getpass.getuser()
print(user)
import re
from tqdm import tqdm

if user=='eghbalhosseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmark='Pereira2018-v2-encoding'
    #model='mistral-caprica-gpt2-small-x81-ckpnt-200000'
    model='gpt2'
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
    # order files
    # find
    # file_order = ['-ckpnt-200000,','permuted,','untrained,', 'untrained_hf', 'untrained-1,', 'untrained-2,', 'untrained-3,', 'untrained-4,',
    #               'untrained-5,', 'untrained-6,', 'untrained-7,', 'untrained-std-1,', 'untrained-mu-1,','untrained-mu-2,',
    #               'untrained-ln-hf,', 'untrained-ln-uniform']
    file_order = ['gpt2,', 'permuted,', 'untrained,', 'untrained_hf', 'untrained-1,', 'untrained-2,',
                  'untrained-3,', 'untrained-4,',
                  'untrained-5,', 'untrained-6,', 'untrained-7,', 'untrained-std-1,', 'untrained-mu-1,',
                  'untrained-mu-2,',
                  'untrained-ln-hf,', 'untrained-ln-uniform']
    # find the string in each element of filer_order in files
    reorder = []
    for y in file_order:
        reorder.append([y in x for x in files].index(True))


    files_srt=[files[x] for x in reorder]
    # remove chekcpoint zero if exist
    scores_mean = []
    scors_std = []
    for ix, file in tqdm(enumerate(files_srt)):
        x = pd.read_pickle(file)['data'].values
        scores_mean.append(x[:, 0])
        scors_std.append(x[:, 1])
    l_names = pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('plasma')
    labels = file_order  # ['layerNorm_1','Self Attention Weight','Self Attention Projection','layerNorm_2','Feedforward_1','Feedforward_2','ALL']
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for idx, scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        #    if 'untrained' in files_srt[idx]:
        #       label= f'{chkpoints_srt[idx]}-untrained'
        #    else:
        #        label = f'{chkpoints_srt[idx]}'
        ax.plot(r3, scr, color=all_col[idx, :], linewidth=2, label=f'HuggingFace initialization for {labels[idx]}')
        ax.errorbar(r3, scr, yerr=scors_std[idx], linewidth=2, color=all_col[idx, :], marker='.', markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(l_names) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))

    ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')

    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'variation_on_untrained_{model}_{benchmark}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'variation_on_untrained_{model}_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .45, .45))
    scr_layer = [x[-1] for x in scores_mean]
    scr_layer_std = [x[-1] for x in scors_std]
    x_coords = np.arange(len(scr_layer))
    for idx, scr in enumerate(scr_layer):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1,
                label=f'{file_order[idx]}', zorder=2)
        ax.errorbar(x_coords[idx], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
    # add precomputed
    # ax.errorbar(idx+.5,model_bench['score'],yerr=model_bench['error'],linestyle='--',fmt='.',markersize=20,linewidth=2,color=(0,0,0,1),label='trained(Schrimpf)',zorder=1)
    # ax.errorbar(-0.5, model_unt_bench['score'], yerr=model_unt_bench['error'], linestyle='--', fmt='.', markersize=20,
    #            linewidth=2, color=(.5, .5, .5, 1), label='untrained(Schrimpf)', zorder=1)

    # ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.set_xticks(np.arange(len(scr_layer)))

    ax.set_xticklabels(file_order, rotation=90, fontsize=8)
    ax.set_ylabel('Pearson Corr')

    fig.show()
