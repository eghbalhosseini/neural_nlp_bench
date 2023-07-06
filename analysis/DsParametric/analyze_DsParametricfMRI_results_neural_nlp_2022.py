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
    benchmarks=['DsParametricfMRI_v1-min-RidgeEncoding', 'DsParametricfMRI_v1-rand-RidgeEncoding','DsParametricfMRI_v1-max-RidgeEncoding']
    #benchmark = 'LangLocECoGv2-encoding'
    models=['roberta-base',
      'xlnet-large-cased',
      'bert-large-uncased-whole-word-masking',
      'xlm-mlm-en-2048',
      'gpt2-xl',
      'albert-xxlarge-v2',
      'ctrl','distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'  ]
    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]
    for model in models:
        x_list=[]
        for benchmark in benchmarks:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
            assert len(files)>0
            x = pd.read_pickle(files[0])['data']
            x_list.append(x)
    # order files

        scores_means=[]
        scors_stds=[]

        scores_means=[x.values[:,0] for x in x_list]
        scores_stds=[x.values[:,1] for x in x_list]
        l_names=x_list[0].layer.values
        #cmap_all = cm.get_cmap('plasma')
        #all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
        width = 0.25  # the width of the bars
        fig = plt.figure(figsize=(11, 8),dpi=250,frameon=False)
        fig_length=0.018*len(l_names)
        ax = plt.axes((.1, .4, fig_length, .35))
        x = np.arange(scores_means[0].shape[0])
        y = scores_means[0]
        y_err = scores_stds[0]
        layer_name = l_names
        rects1 = ax.bar(x-.25, y, width, color=colors[0],linewidth=.5,edgecolor='k',label='min')
        #ax.errorbar(x-.2, y, yerr=y_err, linestyle='', color='k')
        # plot the second item
        y = scores_means[1]
        y_err = scores_stds[1]
        rects2 = ax.bar(x, y, width, color=colors[1],linewidth=.5,edgecolor='k',label='rand')
        #ax.errorbar(x, y, yerr=y_err, linestyle='', color='k')
        # plot the third item
        y = scores_means[2]
        y_err = scores_stds[2]
        #
        rects3 = ax.bar(x+.25, y, width, color=colors[2],linewidth=.5,edgecolor='k',label='max')
        #ax.errorbar(x+.2, y, yerr=y_err, linestyle='', color='k')

        # show the legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                    ncol=1, fancybox=False, shadow=False)

        ax.axhline(y=0, color='k', linestyle='-',linewidth=.5)
            # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Pearson correlation')

        ax.set_xticks(x)
        ax.set_xticklabels(layer_name,rotation=90)
        #ax.set_ylim((-.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'{model}')
        fig.show()
        fig.savefig(os.path.join(PLOTDIR,f'score_{model}_DsParametricfMRI_{benchmark}.png'), dpi=250, format='png', metadata=None,
            bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

        fig.savefig(os.path.join(PLOTDIR, f'score_{model}_DsParametricfMRI_{benchmark}.eps'), format='eps',metadata=None,
                    bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    '''ANN result across models'''
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    layer_ids = (2, 24, 12, 12, 44, 5, 47)
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    models_scores=[]
    for model,layer in model_layers:
        benchmark_score=[]
        for benchmark in benchmarks:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
            assert len(files)>0
            scores_mean=[]
            scors_std=[]
            x=pd.read_pickle(files[0])['data']
            scores_mean=x.values[:,0]
            scores_std=x.values[:,1]
            l_names=x.layer.values
            ann_layer=int(np.argwhere(l_names ==layer))
            benchmark_score.append([scores_mean[ann_layer],scores_std[ann_layer]])
        models_scores.append(benchmark_score)

    models_scores=np.stack(models_scores)
    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    #fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(models_scores.shape[0])

    model_name = model_sh

    rects2 = ax.bar(x -0.25 , models_scores[:,0,0], width, label='min',color=colors[0],linewidth=.5,edgecolor='k')
    ax.errorbar(x -0.25, models_scores[:,0,0], yerr=models_scores[:,0,1], linestyle='', color='k')
    # plot the second item
    rects2 = ax.bar(x , models_scores[:,1,0], width, label='rand',color=colors[1],linewidth=.5,edgecolor='k')
    ax.errorbar(x , models_scores[:,1,0], yerr=models_scores[:,1,1], linestyle='', color='k')
    # plot the third item
    rects3 = ax.bar(x +0.25, models_scores[:,2,0], width, label='max',color=colors[2],linewidth=.5,edgecolor='k')
    ax.errorbar(x +0.25, models_scores[:,2,0], yerr=models_scores[:,2,1], linestyle='', color='k')

    #ax.errorbar(x  , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation (un-normalized)')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on DsParametric')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=45)
    ax.set_ylim((-.175, 0.175))
    ax.set_xlim((-.5, 6.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{benchmark}_err.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{benchmark}_err.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

