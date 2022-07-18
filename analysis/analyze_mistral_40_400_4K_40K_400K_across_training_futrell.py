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
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":

    benchmark='Futrell2018-encoding'
    model='mistral_caprica-gpt2-small-x81'
    chkpnts=[0,40,400,4000,40000,400000]
    files_ckpnt=[]
    for ckpnt in chkpnts:
        if ckpnt==0:
            ckpnt=str(ckpnt)+'-untrained'
        file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model}_ckpt_{ckpnt},*.pkl'))
        print(file_c)
        if len(file_c)>0:
            files_ckpnt.append(file_c[0])

    chkpoints_srt=['untrained-manual (n=0)','0.01% (n=40)','0.1% (n=400)' ,'1% (n=4K)','10% (n=40K)','100% (n=400K)']
    # order files
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_ckpnt)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])

    # read precomputed scores
    l_names = pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .45, .35))
    for idx, scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        ax.plot(idx, scr, color=all_col[idx, :], linewidth=2, marker='.', markersize=20,
                label=f'ck:{chkpoints_srt[idx]},', zorder=2)
        ax.errorbar(idx, scr, yerr=scors_std[idx], color='k', zorder=1)    # add precomputed
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1., .8), frameon=True,fontsize=8)
    ax.set_xlim((-1,len(scores_mean)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylim([-.15, 1.0])
    plt.grid(True, which="both", ls="-", color='0.9')
    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} \n model:{model}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_through_training_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_through_training_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)