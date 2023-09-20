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
    benchmarks=['Futrell2018-encoding']#,
                #'Pereira2023aud-sent-passage-RidgeEncoding', 'Pereira2023aud-sent-sentence-RidgeEncoding']
    #benchmark = 'LangLocECoGv2-encoding'
    benchmarks = ['Pereira2018-v2-encoding']
    models=['gpt2-'+x for x in ['untrained', 'untrained_hf','untrained-1','untrained-2', 'untrained-3',
                               'untrained-4', 'untrained-5',
                               'untrained-6', 'untrained-std-1', 'untrained-std-2',
                               'untrained-mu-1', 'untrained-mu-2',
                               'untrained-ln-uniform']]

    models=['gpt2-'+x for x in ['untrained', 'untrained_hf', 'untrained-std-1', 'untrained-std-2',
                                'untrained-ln-uniform']]
    models.append('gpt2')

    file_order = [
                  'N(0,0.0.2)',
                  'HuggingFace(HF)',
                  'LayerNorm1=HF',
                  'Attention=HF',
                  'Projection=HF',
                  'LayerNorm2=HF',
                  'Feedforward1=HF',
                  'Feedforward2=HF',
                  'N(0,0.05)',
                  'N(0,0.1)',
                  'N(0.5,0.02)',
                  'N(1.0,0.02)',
                  'LayerNorm=uniform[0,1]',
                'Trained',]

    file_order = [
        'N(0,0.0.2)',
        'HuggingFace(HF)',
        'N(0,0.05)',
        'N(0,0.1)',
        'LayerNorm=uniform[0,1]',
        'Trained', ]
    # create a list of colors for the number of models
    colors = cm.rainbow(np.linspace(0, 1, len(models)))


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
    ax = plt.axes((.1, .4, .35, .35))
    for k in range(len(models_scores)):
        y = models_scores[k][0][0]
        y_err = models_scores[k][0][1]
        x=np.arange(len(y))
    # rects1 = ax.bar(x-width, y, width, color=colors[0],linewidth=.5,label='passage')
        rects1 = ax.plot(x, y, color=colors[k], linewidth=2, marker='o', markersize=5, markerfacecolor='k',
                     markeredgecolor='k', label=f'{models[k]}')
    # ax.errorbar(x-width, y, yerr=y_err, linestyle='', color='k')
        #ax.fill_between(x, y - y_err, y + y_err, facecolor=colors[k], alpha=0.2)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=.5)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.legend( bbox_to_anchor=(1.1, 1),
              ncol=1, fancybox=False, shadow=False)
    ax.set_xticks(x)
    ax.set_xlabel('model layer')
    ax.set_ylim((-.1, 1.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'{benchmarks[0]}')
    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'score_{benchmarks[0]}_untrained_versions_gpt2.png'), dpi=250, format='png',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    # save as eps
    fig.savefig(os.path.join(PLOTDIR, f'score_{benchmarks[0]}_untrained_versions_gpt2.eps'), dpi=250, format='eps',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)



    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .45, .45))
    scr_layer=[x[0][0][-1] for x in models_scores]
    scr_layer_std=[x[0][1][-1] for x in models_scores]
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(scr_layer)+1), len(scr_layer)+1))

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

    #ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.set_xticks(np.arange(len(scr_layer)))

    ax.set_xticklabels(file_order, rotation=90, fontsize=8)
    ax.set_ylabel('Pearson Corr')
    ax.set_ylim((-.1, 1.1))
    fig.show()

    fig.savefig(os.path.join(PLOTDIR, f'score_{benchmarks[0]}_untrained_versions_gpt2_last_layer.png'), dpi=250, format='png',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    # save as eps
    fig.savefig(os.path.join(PLOTDIR, f'score_{benchmarks[0]}_untrained_versions_gpt2_last_layer.eps'), dpi=250, format='eps',
                metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
