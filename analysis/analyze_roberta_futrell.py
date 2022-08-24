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
    benchmark = 'Futrell2018-encoding'
    model_1B = 'nyu-mll_roberta-base-1B-1'
    precomputed_model = 'roberta-base'
    model_100M = 'nyu-mll_roberta-base-100M-1'
    model_10M = 'nyu-mll_roberta-base-10M-1'

    file_1B_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-untrained*.pkl'))
    file_1B = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model_1B}*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_100M}*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_10M}*.pkl'))
    files_srt = [file_1B_untrained[0], [], file_10M[0], file_100M[0], file_1B[0]]
    # files_srt = [ file_10M[0], file_100M[0], file_1B[0]]
    chkpoints_srt = ['untrained', '1M', '10M', '100M', '1B']
    # order files
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_srt)):
        if len(file) > 0:
            x = pd.read_pickle(file)['data'].values
            scores_mean.append(x[:, 0])
            scors_std.append(x[:, 1])
        else:
            scores_mean.append(np.nan)
            scors_std.append(np.nan)

    # read precomputed scores
    precomputed=pd.read_csv('/om/user/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench=precomputed[precomputed['benchmark']==benchmark]
    model_bench=precomputed_bench[precomputed_bench['model']==precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model+'-untrained']

    #file_precompute = glob(os.path.join(result_caching, 'neural_nlp.score',
    #                            f'benchmark={benchmark},model={precomputed_model},*.pkl'))
    #model_bench=pd.read_pickle(file_precompute[0])['data'].values

    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    #ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .35, .35))
    x_coords=[1e5,1e6,10e6,100e6,1000e6]
    for idx,scr in enumerate(scores_mean):
        if scr is not np.nan:
            ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10,
                    label=f'{chkpoints_srt[idx]}')
            ax.errorbar(x_coords[idx], scr,yerr=scors_std[idx],color='k',zorder=1)
    ax.set_xscale('log')
    ax.plot(x_coords,scores_mean,color='k',linewidth=2,zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate([np.arange(1,11)*1e5,np.arange(1,11)*1e6,np.arange(1,11)*1e7,np.arange(1,11)*1e8])
    ax.plot(8000e6, np.asarray(model_bench.score.values[0]), color=(.3,.3,.3,1), linewidth=2, marker='o', markersize=10,
            label=f'Schrimpf(2021)', zorder=2)
    ax.errorbar(8000e6, np.asarray(model_bench.score.values), yerr=np.asarray(model_bench.error.values), color='k', zorder=1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(np.concatenate([major_ticks, [8000e6]]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticklabels(['untrained','1M', '10M', '100M', '1B', 'Schrimpf\n(2021)'], rotation=0)
    ax.set_ylim([-0.05, 1])
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True, fontsize=8)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox_{benchmark}.png'), dpi=300, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)