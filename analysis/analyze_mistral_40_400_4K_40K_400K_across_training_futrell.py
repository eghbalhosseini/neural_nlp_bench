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
from scipy.stats import ttest_ind_from_stats, ttest_ind

if user=='eghbalhosseini':
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":

    benchmark='Futrell2018-encoding'
    #benchmark = 'Futrell2018-sentences_encoding'
    model='mistral-caprica-gpt2-small-x81'
    chkpnts=[0,40,400,4000,40000,400000]

    files_ckpnt=[]
    for ckpnt in chkpnts:
        if ckpnt==0:
            ckpnt=str(ckpnt)+'-untrained'
        file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},*.pkl'))
        print(file_c)
        if len(file_c)>0:
            files_ckpnt.append(file_c[0])

    chkpoints_srt=['untrained-manual (n=0)','0.01% (n=40)','0.1% (n=400)' ,'1% (n=4K)','10% (n=40K)','100% (n=400K)']

    shcrimpf = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model=gpt2,*.pkl'))
    schirmpf_data = pd.read_pickle(shcrimpf[0])['data']

    # order files
    scores_mean=[]
    scors_std=[]
    score_data=[]
    for ix, file in tqdm(enumerate(files_ckpnt)):
        x=pd.read_pickle(file)['data']
        scores_mean.append(x.values[:,0])
        scors_std.append(x.values[:,1])
        score_data.append(x)
    # read precomputed scores
    l_names = pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .45, .35))

    x_coords = [0.001, 0.01, 0.1, 1, 10, 100]
    for idx, scr in enumerate(scores_mean):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1,
                label=f'{chkpoints_srt[idx]}', zorder=2)
        ax.errorbar(x_coords[idx], scr, yerr=scors_std[idx], color='k', zorder=1)
    ax.set_xscale('log')
    ax.plot(x_coords, scores_mean, color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate(
        [np.arange(1, 11) * 1e-3, np.arange(1, 11) * 1e-2, np.arange(1, 11) * 1e-1, np.arange(1, 11) * 1e0,
         np.arange(1, 11) * 1e1])

    ax.plot(8e2, np.asarray(model_bench['score'])[layer_id], color=(.3, .3, .3, 1), linewidth=2, marker='o',
            markersize=10,
            label=f'Schrimpf(2021)', zorder=2)
    ax.errorbar(8e2, np.asarray(model_bench['score'])[layer_id], yerr=np.asarray(model_bench['error'])[layer_id],
                color='k', zorder=1)

    ax.plot(0.0008, untrained_hf[layer_id][0], color=all_col[0, :], linewidth=2, marker='o', markeredgecolor='w',
            markersize=10, label=f'HF_untrained', zorder=2)
    ax.errorbar(0.0008, untrained_hf[layer_id][0], yerr=untrained_hf[layer_id][1], color='k', zorder=1)


    for idx, scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        ax.plot(idx, scr, color=all_col[idx, :], linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(idx, scr, yerr=scors_std[idx], linewidth=2,
                    color=all_col[idx, :], marker='.', markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    #ax.legend(bbox_to_anchor=(1., .8), frameon=True,fontsize=8)
    ax.set_xlim((-1,len(scores_mean)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylim([.2, 0.9])
    plt.grid(True, which="both", ls="-", color='0.9')
    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} \n model:{model}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_through_training_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_through_training_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)


