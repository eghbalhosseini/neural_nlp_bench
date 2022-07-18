import pickle as pkl
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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
#%%
if __name__ == "__main__":
    benchmark='Pereira2018-encoding'
    #benchmark='Fedorenko2016v3-encoding'
    model='gpt2-neox-pos_learned-1B'
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
    # order files
    chkpoints=[re.findall(r'ckpnt-\d+',x)[0] for x in files]
    chkpoints=[int(x.replace('ckpnt-','')) for x in chkpoints]
    reorder=np.argsort(chkpoints)
    chkpoints_srt=[chkpoints[x] for x in reorder]
    files_srt=[files[x] for x in reorder]
#%%
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])
    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for idx,scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx,:],linewidth=2,label=f'ck:{chkpoints_srt[idx]}')
        ax.errorbar(r3, scr, yerr=scors_std[idx], linewidth=2, color=all_col[idx, :],marker='.', markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.8, 2), frameon=True,fontsize=8,ncol=3)
    ax.set_xlim((0-.5,len(l_names)-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))

    ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax = plt.axes((.1, .6, .45, .35))
    vmax = np.ceil(10 * (np.max(np.stack(scores_mean)) + .1)) / 10
    ax.imshow(np.stack(scores_mean), vmax=vmax, aspect='auto')
    ax.set_yticks(np.arange(0,len(scores_mean),5))
    ax.set_yticklabels(chkpoints_srt[::5], fontsize=6)

    ax.set_title(f'model:{model}, benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)