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
ROOTDIR = (Path('/om/weka/evlab/ehoseini/MyData/ecog_SN/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'outputs')).resolve()
PLOTDIR = (Path(OUTDIR / 'plots')).resolve()

if user=='eghbalhosseini':
    #analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    True
elif user=='ehoseini':
    analysis_dir='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmarks=['Pereira2018-max-encoding', 'Pereira2018-min-encoding' ,'Pereira2018-rand-encoding']
    #benchmark = 'LangLocECoGv2-encoding'
    models=['roberta-base',
      'xlnet-large-cased',
      'bert-large-uncased-whole-word-masking',
      'xlm-mlm-en-2048',
      'gpt2-xl',
      'albert-xxlarge-v2',
      'ctrl']
    for benchmark in benchmarks:
        for model in models:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
        assert len(files)>0
    # order files

        scores_mean=[]
        scors_std=[]
        x=pd.read_pickle(files[0])['data']
        scores_mean=x.values[:,0]
        scores_std=x.values[:,1]
        l_names=x.layer.values
        cmap_all = cm.get_cmap('plasma')
        all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))

        width = 0.7  # the width of the bars
        fig = plt.figure(figsize=(11, 8),dpi=250,frameon=False)
        fig_length=0.0135*len(l_names)
        ax = plt.axes((.2, .4, fig_length, .35))
        x = np.arange(scores_mean.shape[0])
        y = scores_mean
        y_err = scores_std
        layer_name = l_names
        rects1 = ax.bar(x, y, width, color=np.divide((188, 80, 144), 255))
        ax.errorbar(x, y, yerr=y_err, linestyle='', color='k')
        ax.axhline(y=0, color='k', linestyle='-')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Pearson correlation')
        ax.set_title(f'{model} \n performance on {benchmark}')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_name,rotation=90)
        ax.set_ylim((-.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(os.path.join(analysis_dir,f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    '''ANN result across models'''
    pereira_benchmark = [ 'Pereira2018-min-V2-encoding','Pereira2018-rand-V2-encoding','Pereira2018-max-V2-encoding' ]

    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    layer_ids = (2, 24, 12, 12, 44, 5, 47)
    benchmark_scores=[]
    benchmark_scores_raw=[]
    for benchmark in pereira_benchmark:
        models_scores = []
        model_score_raw=[]
        for model,layer in model_layers:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
            assert len(files)>0
            # order files
            scores_mean=[]
            scors_std=[]
            x=pd.read_pickle(files[0])['data']
            scores_mean=x.values[:,0]
            scores_std=x.values[:,1]


            l_names=x.layer.values
            ann_layer=int(np.argwhere(l_names ==layer))
            models_scores.append([scores_mean[ann_layer],scores_std[ann_layer]])
            model_score_raw.append(x.raw)
        benchmark_scores.append(models_scores)
        benchmark_scores_raw.append(model_score_raw)

    width = 0.15  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)

    fig_length = 0.02 * len(benchmark_scores)*len(model_layers)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(len(model_layers))
    colors=[np.divide((188, 80, 144), 255),np.divide((55, 76, 128), 256),np.divide((255, 128, 0), 255),np.divide((55, 76, 128), 256)]
    model_name = [f'{x[0]}' for x in model_layers]
    # plot pereira_min
    y = np.asarray(benchmark_scores[0])[:,0]
    y_err = np.asarray(benchmark_scores[0])[:,1]
    rects1 = ax.bar(x -2*width, y, width, color=colors[0], label='Pereira')
    ax.errorbar(x  -2*width, y, yerr=y_err, linestyle='', color='k')

    y = np.asarray(benchmark_scores[1])[:, 0]
    y_err = np.asarray(benchmark_scores[1])[:, 1]
    rects1 = ax.bar(x - 1 * width, y, width, color=colors[1], label='Pereira')
    ax.errorbar(x - 1 * width, y, yerr=y_err, linestyle='', color='k')

    y = np.asarray(benchmark_scores[2])[:, 0]
    y_err = np.asarray(benchmark_scores[2])[:, 1]
    rects1 = ax.bar(x -0 * width, y, width, color=colors[2], label='Pereira')
    ax.errorbar(x - 0 * width, y, yerr=y_err, linestyle='', color='k')

#    y = np.asarray(benchmark_scores[3])[:, 0]
#    y_err = np.asarray(benchmark_scores[3])[:, 1]
##    rects1 = ax.bar(x +1 * width, y, width, color=colors[3], label='Pereira')
#    ax.errorbar(x + 1 * width, y, yerr=y_err, linestyle='', color='k')

    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.1, 1.1))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_{benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_{benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    bench_243_raw=[]
    for bench_raw in benchmark_scores_raw:
        raw_243sent=[x.raw.raw.sel(experiment='243sentences').mean(dim='split').groupby('subject').median(dim='neuroid').median('subject') for x in bench_raw]
        # select modellayer for model_layers in raw_243sent
        raw_243sent=[x.sel(layer=model_layers[i][1]).values for i,x in enumerate(raw_243sent)]
        bench_243_raw.append(raw_243sent)

    # plot benchmark 243sentences
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.02 * len(benchmark_scores) * len(model_layers)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(len(model_layers))
    colors = [np.divide((188, 80, 144), 255), np.divide((55, 76, 128), 256), np.divide((255, 128, 0), 255),
              np.divide((55, 76, 128), 256)]
    model_name = [f'{x[0]}' for x in model_layers]
    # plot pereira_min
    y = np.asarray(bench_243_raw[0])
    rects1 = ax.bar(x - 2 * width, y, width, color=colors[0], label='Pereira')
    y = np.asarray(bench_243_raw[1])
    rects1 = ax.bar(x - 1 * width, y, width, color=colors[1], label='Pereira')
    y = np.asarray(bench_243_raw[2])
    rects1 = ax.bar(x - 0 * width, y, width, color=colors[2], label='Pereira')

    fig.show()