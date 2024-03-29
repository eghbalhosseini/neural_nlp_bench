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
    benchmark='Pereira2018-encoding'
    model='mistral-caprica-gpt2-small-x81-ckpnt-400000-untrained-mu'
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
    # order files




    files_srt=files
    # remove chekcpoint zero if exist
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])
    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('plasma')
    labels=['mu=0.5','mu=1']
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for idx,scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        label=labels[idx]
        ax.plot(r3, scr, color=all_col[idx,:],linewidth=2,label=f'{label}')
        ax.errorbar(r3, scr, yerr=scors_std[idx], linewidth=2, color=all_col[idx, :],marker='.', markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.4, 2), frameon=True,fontsize=8)
    ax.set_xlim((0-.5,len(l_names)-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))

    ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'model:{model}, benchmark {benchmark}')
    #fig.show()

    fig.savefig(os.path.join(analysis_dir,f'effect_of_mu_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'effect_of_mu_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    ''' comparing untrained vs manaul permutation'''
    unt_files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model}_ckpt_0-untrained*.pkl'))
    init_files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model}_ckpt_0,*.pkl'))

    scores_unt = pd.read_pickle(unt_files[0])['data'].values
    scores_init = pd.read_pickle(init_files[0])['data'].values

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))

    r3 = np.arange(len(scores_unt))
    ax.plot(r3, scores_unt[:,0], color='k', linewidth=2, label=f'untrained-manual permutation')
    ax.errorbar(r3, scores_unt[:,0], yerr=scores_unt[:,1], linewidth=2, color='k', marker='.', markersize=10)

    ax.plot(r3, scores_init[:, 0], color=(.5,.5,.5,1), linewidth=2, label=f'untrained-HG')
    ax.errorbar(r3, scores_init[:, 0], yerr=scores_init[:, 1], linewidth=2, color=(.5,.5,.5,1), marker='.', markersize=10)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.4, 1), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(scores_unt) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_unt)))

    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('Layer')
    ax.set_title(f'model:{model}, benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'untrained_score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'untrained_score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)