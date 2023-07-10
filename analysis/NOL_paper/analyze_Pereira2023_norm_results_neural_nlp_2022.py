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
from pathlib import Path
ROOTDIR = (Path('/om/weka/evlab/ehoseini/MyData/fmri_DNN/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'outputs')).resolve()
PLOTDIR = (Path(OUTDIR / 'plots')).resolve()
from glob import glob
if user=='eghbalhosseini':
    #analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    True
elif user=='ehoseini':
    analysis_dir='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmarks=['Pereira2018-norm-sentence-encoding']#,
                #'Pereira2023aud-sent-passage-RidgeEncoding', 'Pereira2023aud-sent-sentence-RidgeEncoding']
    #benchmark = 'LangLocECoGv2-encoding'
    models=['distilgpt2',
      'distilgpt2-untrained',]
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]

    precomputed = pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == 'Pereira2018-encoding']
    model_bench = [precomputed_bench[precomputed_bench['model'] == x] for x in models]
    models_scores=[]
    for model in models:
        benchmark_score=[]
        for benchmark in benchmarks:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
            assert len(files)>0
            scores_mean=[]
            scors_std=[]
            x=pd.read_pickle(files[0])['data']
            scores_mean=x.values[:,0]
            scores_std=x.values[:,1]
            benchmark_score.append([scores_mean,scores_std])
        models_scores.append(benchmark_score)


    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.018 * len(model_bench[0])
    ax = plt.axes((.1, .4, fig_length, .35))
    for k in [0,1]:
        x = np.arange(model_bench[0]['score'].shape[0])
        y = model_bench[k]['score'].values
        y_err = model_bench[k]['error'].values
        layer_name = model_bench[k]['layer'].values
    # rects1 = ax.bar(x-width, y, width, color=colors[0],linewidth=.5,label='passage')
        rects1 = ax.plot(x, y, color=colors[k], linewidth=2, marker='s', markersize=5, markerfacecolor='r',
                     markeredgecolor='r', label=f'{models[k]}-regular')
    # ax.errorbar(x-width, y, yerr=y_err, linestyle='', color='k')
        ax.fill_between(x, y - y_err, y + y_err, facecolor=colors[k], alpha=0.2)


    for k in [0,1]:
        y = models_scores[k][0][0]
        y_err = models_scores[k][0][1]
    # rects1 = ax.bar(x-width, y, width, color=colors[0],linewidth=.5,label='passage')
        rects1 = ax.plot(x, y, color=colors[k], linewidth=2, marker='o', markersize=5, markerfacecolor='k',
                     markeredgecolor='k', label=f'{models[k]}-norm')
    # ax.errorbar(x-width, y, yerr=y_err, linestyle='', color='k')
        ax.fill_between(x, y - y_err, y + y_err, facecolor=colors[k], alpha=0.2)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=.5)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, .8),
              ncol=1, fancybox=False, shadow=False)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_name, rotation=90)
    ax.set_ylim((-.1, 1.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'Pereira benchmark')
    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'score_Pereira2018_regular_vs_norm_{models[0]}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

