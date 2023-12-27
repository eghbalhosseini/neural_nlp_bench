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
    #benchmark='LangLocECoG-sentence-encoding'
    benchmark = 'ANNSet1ECoG-Sentence-encoding'
    #benchmark = 'LangLocECoGv2-encoding'
    models=['roberta-base',
      'xlnet-large-cased',
      'bert-large-uncased-whole-word-masking',
      'xlm-mlm-en-2048',
      'gpt2-xl',
      'albert-xxlarge-v2',
      'ctrl']
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
        ax.set_ylim((-.1, .4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(os.path.join(analysis_dir,f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.eps'), format='eps',metadata=None,
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
    models_scores=[]
    models_scores_best = []
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
        # find best layer score and add it to models_scores_best
        best_layer=np.argmax(scores_mean)
        models_scores_best.append([scores_mean[best_layer],scores_std[best_layer]])

    models_scores=np.stack(models_scores)
    models_scores_best=np.stack(models_scores_best)
    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    #fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]}' for x in model_layers]

    rects2 = ax.bar(x + width , models_scores[:,0], width, label='ANNSet1 Layer',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    rects1 = ax.bar(x, models_scores_best[:,0], width, label='Best Layer',color=np.divide((80, 144, 188), 255))
    ax.errorbar(x, models_scores_best[:, 0], yerr=models_scores_best[:, 1], linestyle='', color='k')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.15, .3))
    ax.set_xlim((-.5, 6.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_{benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_{benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    """compare models from the same class on the data """
    models = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large' ,'gpt2-xl' ]
    models_scores = []
    model_layers=[]
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
        l_max=np.argmax(x.values[:, 0])
        model_layers.append(l_max)
        ann_layer=int(l_max)
        models_scores.append([scores_mean[ann_layer],scores_std[ann_layer]])


    models_scores=np.stack(models_scores)
    width = 0.35  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in zip(models,model_layers)]
    rects2 = ax.bar(x + width / 2, models_scores[:,0], width, label='ANNSet1_fMRI',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.1, .3))
    #ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'GPT2_models_scores_{benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'GPT2_models_scores_{benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    """look at all models"""
    all_models=["roberta-large", "distilroberta-base",
    "xlnet-base-cased",
    "bert-base-uncased",
    "bert-base-multilingual-cased",
    "bert-large-uncased",
    "xlm-mlm-enfr-1024", "xlm-mlm-100-1280",
     "albert-base-v2", "albert-large-v1", "albert-large-v2", "albert-xlarge-v1"]
    all_models_scores = []
    all_per_models_scores = []
    all_model_layers = []

    ALL_model_scores=[]
    for model in all_models:
        files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model},*.pkl'))
        assert len(files) > 0
        # order files

        scors_std = []
        x = pd.read_pickle(files[0])['data']

        scores_mean = x.values[:, 0]
        scores_std = x.values[:, 1]
        l_names = x.layer.values
        l_max = np.argmax(x.values[:, 0])
        all_model_layers.append(l_max)
        ann_layer = int(l_max)
        all_models_scores.append([scores_mean[ann_layer], scores_std[ann_layer]])

        cmap_all = cm.get_cmap('plasma')
        all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))

        width = 0.7  # the width of the bars
        fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
        fig_length = 0.0135 * len(l_names)
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
        ax.set_xticklabels(layer_name, rotation=90)
        ax.set_ylim((-.1, .6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.eps'), format='eps', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    models_scores=np.stack(all_models_scores)

    width = 0.35  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in zip(all_models,all_model_layers)]


    rects2 = ax.bar(x + width / 2, models_scores[:,0], width, label='',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.1, 0.2))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'ALL_models_scores_{benchmark}.png'), dpi=250,
                format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'ALL_models_scores_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
