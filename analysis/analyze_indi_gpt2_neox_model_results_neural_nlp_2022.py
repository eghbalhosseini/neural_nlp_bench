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
    benchmark='Futrell2018-encoding'
    model='gpt2-neox-pos_learned-1B'
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
    # order files
    chkpoints=[re.findall(r'ckpnt-\d+',x)[0] for x in files]
    chkpoints=[int(x.replace('ckpnt-','')) for x in chkpoints]
    reorder=np.argsort(chkpoints)
    chkpoints_srt=[chkpoints[x] for x in reorder]
    files_srt = [files[x] for x in reorder]
    # subsample here
    reorder_new=np.concatenate((np.asarray([0,]),np.squeeze(np.argwhere(np.mod(np.asarray(chkpoints_srt)[:,],10000)==0))))
    files_select=[files_srt[x] for x in reorder_new]
    chkpoints_select=[chkpoints_srt[x] for x in reorder_new]
#%%
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_select)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])
    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    locas=[]
    for idx, scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        if 'untrained' in files_select[idx]:
            label = f'{chkpoints_select[idx]}-untrained'
            loca=0
        else:
            label = f'{chkpoints_select[idx]}'
            loca=chkpoints_select[idx];
        locas.append(loca)
        ax.plot(loca, scr, color=all_col[idx, :], linewidth=2, label=f'ck:{label}')
        ax.errorbar(loca, scr, yerr=scors_std[idx], linewidth=2, color=all_col[idx, :], marker='.',
                    markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.plot(locas, scores_mean, color='k', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    # ax.set_xlim((0-.5,len(scores_mean)-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))

    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('training step')
    ax.set_title(f'model:{model}, benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)