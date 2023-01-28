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
ROOTDIR = (Path('/om/user/ehoseini/MyData/fmri_DNN/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'outputs')).resolve()
PLOTDIR = (Path(OUTDIR / 'plots')).resolve()

if user=='eghbalhosseini':
    #analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    #result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
    True
elif user=='ehoseini':
    #analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    #result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmark='ANNSet1fMRI-wordForm-encoding'
    models=['roberta-base',
      'xlnet-large-cased',
      'bert-large-uncased-whole-word-masking',
      'xlm-mlm-en-2048',
      'gpt2-xl',
      'albert-xxlarge-v2',
      'ctrl']
    for model in models:
        files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
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
        fig.savefig(os.path.join(PLOTDIR,f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

        fig.savefig(os.path.join(PLOTDIR, f'score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    '''ANN result across models'''
    pereira_benchmark = 'Pereira2018-encoding'
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    layer_ids = (2, 24, 12, 12, 44, 5, 47)
    models_scores=[]
    per_models_scores = []
    for model,layer in model_layers:
        files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
        assert len(files)>0
        # order files
        per_files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={pereira_benchmark},model={model}*.pkl'))
        assert len(per_files)>0
        scores_mean=[]
        scors_std=[]
        x=pd.read_pickle(files[0])['data']
        x_per = pd.read_pickle(per_files[0])['data']
        scores_mean=x.values[:,0]
        scores_std=x.values[:,1]

        per_scores_mean = x_per.values[:,0]
        per_scores_std = x_per.values[:, 1]

        l_names=x.layer.values
        ann_layer=int(np.argwhere(l_names ==layer))
        models_scores.append([scores_mean[ann_layer],scores_std[ann_layer]])
        per_models_scores.append([per_scores_mean[ann_layer], per_scores_std[ann_layer]])

    models_scores=np.stack(models_scores)
    per_models_scores = np.stack(per_models_scores)
    width = 0.35  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in model_layers]

    rects1 = ax.bar(x - width / 2, per_models_scores[:,0], width, color=np.divide((55, 76, 128), 256), label='Pereira')
    ax.errorbar(x - width / 2, per_models_scores[:, 0], yerr=per_models_scores[:, 1], linestyle='', color='k')

    rects2 = ax.bar(x + width / 2, models_scores[:,0], width, label='ANNSet1_fMRI',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
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

    """compare models from the same class on the data """
    models = ['openaigpt', 'gpt2', 'gpt2-medium', 'gpt2-large' ,'gpt2-xl','distilgpt2' ]
    models=['roberta-base', 'roberta-large', 'distilroberta-base',
      'xlnet-large-cased', 'xlnet-base-cased',
      'bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased', 'bert-large-uncased-whole-word-masking',
      'xlm-mlm-en-2048', 'xlm-mlm-enfr-1024', 'xlm-mlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280',
      'openaigpt', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2',
      'albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2',
    'albert-xxlarge-v1', 'albert-xxlarge-v2']

    for model in models:
        files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model},subsample=None.pkl'))
        assert len(files) > 0
        # order files

        scores_mean = []
        scors_std = []
        x = pd.read_pickle(files[0])['data']
        scores_mean = x.values[:, 0]
        scores_std = x.values[:, 1]
        l_names = x.layer.values
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
        ax.set_ylim((-.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(os.path.join(PLOTDIR, f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

        fig.savefig(os.path.join(PLOTDIR, f'score_{model}_{benchmark}.eps'), format='eps', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)



        """look at all models"""
        models=['sentence-length','word-position','random-embedding','skip-thoughts','skip-thoughts-untrained','lm_1b','lm_1b-untrained',
 'word2vec','word2vec-untrained','glove','glove-untrained','transformer','transformer-untrained',
 'ETM','ETM-untrained','bert-base-uncased','bert-base-multilingual-cased',
 'bert-large-uncased','bert-large-uncased-whole-word-masking','openaigpt','gpt2','gpt2-medium',
'gpt2-large','gpt2-xl','distilgpt2','transfo-xl-wt103','xlnet-base-cased','xlnet-large-cased','xlm-mlm-en-2048','xlm-mlm-enfr-1024',
 'xlm-mlm-xnli15-1024','xlm-clm-enfr-1024','xlm-mlm-100-1280','roberta-base','roberta-large','distilroberta-base','distilbert-base-uncased','ctrl',
 'albert-base-v1','albert-base-v2','albert-large-v1','albert-large-v2','albert-xlarge-v1','albert-xlarge-v2','albert-xxlarge-v1','albert-xxlarge-v2',
 't5-small','t5-base','t5-large','t5-3b','t5-11b',
 'xlm-roberta-base','xlm-roberta-large','bert-base-uncased-untrained','bert-base-multilingual-cased-untrained',
 'bert-large-uncased-untrained','bert-large-uncased-whole-word-masking-untrained','openaigpt-untrained','gpt2-untrained','gpt2-medium-untrained',
 'gpt2-large-untrained','gpt2-xl-untrained','distilgpt2-untrained',
 'transfo-xl-wt103-untrained','xlnet-base-cased-untrained','xlnet-large-cased-untrained','xlm-mlm-en-2048-untrained','xlm-mlm-enfr-1024-untrained',
 'xlm-mlm-xnli15-1024-untrained','xlm-clm-enfr-1024-untrained','xlm-mlm-100-1280-untrained','roberta-base-untrained',
 'roberta-large-untrained','distilroberta-base-untrained','distilbert-base-uncased-untrained',
 'ctrl-untrained','albert-base-v1-untrained','albert-base-v2-untrained',
 'albert-large-v1-untrained','albert-large-v2-untrained','albert-xlarge-v1-untrained',
 'albert-xlarge-v2-untrained', 'albert-xxlarge-v1-untrained','albert-xxlarge-v2-untrained',
 't5-small-untrained','t5-base-untrained','t5-large-untrained','t5-3b-untrained','t5-11b-untrained','xlm-roberta-base-untrained','xlm-roberta-large-untrained']
        ALL_model_scores=[]
        for model in models:
            files = glob(os.path.join(result_caching, 'neural_nlp.score',
                                      f'benchmark={benchmark},model={model},subsample=None.pkl'))
            if len(files) > 0:
            # order files
                x = pd.read_pickle(files[0])['data']
                scores_mean = x.values[:, 0]
                scores_std = x.values[:, 1]
                best_layer=np.argmax(scores_mean)
                ALL_model_scores.append([model,scores_mean[best_layer],scores_std[best_layer]])



        model_classes = ['glove', 'ETM', 'word2vec', 'lm_1b', 'skip-thoughts', 'bert-', 'roberta', 'xlm', 'xlm-roberta-'
            , 'transfo-xl', 'xlnet', 'ctrl', 't5', 'albert-', 'gpt']
        color_groups = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                        'RdPu', 'BuPu',
                        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        color_ids = []
        model_set = scores['model'].unique()
        model_perf = []
        model_perf_untrained = []
        model_names = []
        model_layer_ids = []
        for model_id in model_set:
            model_score = scores['score'][scores['model'] == model_id]
            model_error = scores['error'][scores['model'] == model_id]
            model_layer_ = scores['layer'][scores['model'] == model_id]
            if model_id.find('untrained') == -1:
                model_perf.append([model_score.max(), model_error[model_score.idxmax()]])
                model_names.append(f"{model_id} ({model_layer_[model_score.idxmax()]})")
                model_layer_ids.append(model_layer_)
                # get untrained score
                model_score_untr = list(scores['score'][scores['model'] == f"{model_id}-untrained"])
                model_error_untr = list(scores['error'][scores['model'] == f"{model_id}-untrained"])
                model_perf_untrained.append(
                    [model_score_untr[np.argmax(model_score)], model_error_untr[np.argmax(model_score)]])
                if model_id == 'xlm-roberta-base' or model_id == 'xlm-roberta-large':
                    color_loc = 8
                else:
                    color_loc = int(np.argwhere([model_id.find(x) != -1 for x in model_classes])[-1].squeeze())
                color_ids.append(color_loc)

        num_cols = [len(np.where(np.asarray(color_ids) == x)[0]) for idx, x in enumerate(np.unique(color_ids))]
        h0s = [cm.get_cmap(color_groups[x], num_cols[idx] + 2) for idx, x in enumerate(np.unique(color_ids))]
        all_colors = [np.flipud(x(np.arange(num_cols[idx]) / (num_cols[idx] + 1))) for idx, x in enumerate(h0s)]
        # all_colors=[x(np.arange(num_cols[idx])/(num_cols[idx]+1)) for idx, x in enumerate(h0s)],
        all_colors = [item for sublist in all_colors for item in sublist]

        y_pos = np.arange(len(model_names))
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_axes((.5, .1, .4, .85))
        ax.barh(y_pos, np.asarray(model_perf)[:, 0], height=.8, xerr=np.asarray(model_perf)[:, 1], align='center',
                color=np.asarray(all_colors), edgecolor=(0, 0, 0), linewidth=1, error_kw={'linewidth': 1})
        # ax.barh(y_pos, np.asarray(model_perf_untrained)[:,0],height=0.8, align='center',color=np.asarray(all_colors),edgecolor=(.2,.2,.2),linewidth=1)
        ax.barh(y_pos, np.asarray(model_perf_untrained)[:, 0], height=0.8, align='center', color=(.8, .8, .8),
                edgecolor=(.2, .2, .2), linewidth=1, alpha=.7)
        # ax.scatter(np.asarray(model_perf_untrained)[:,0],y_pos,s=20,zorder=10)
        ax.set_yticks(y_pos)
        ax.set_ylim(y_pos.min() - 1, y_pos.max() + 1)
        ax.set_yticklabels(model_names, fontsize=8, fontweight='normal')
        ax.tick_params(axis='x', which='both', labelsize=8)
        ax.set_xlabel('score', fontsize=8)
        plt.xticks(fontsize=8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.savefig(os.path.join(ANALYZE_DIR, 'schrimpf_2021_preira_model_scores.pdf'), transparent=True)