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
    benchmark='ANNSet1ECoG-encoding'
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
        ax.set_ylim((-.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        fig.savefig(os.path.join(analysis_dir,f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    '''ANN result across models'''
    langloc_benchmark='LangLocECoGv2-encoding'
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
        files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
        assert len(files)>0
        # order files
        per_files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={langloc_benchmark},model={model},*.pkl'))
        assert len(per_files)>0
        scores_mean=[]
        scors_std=[]
        x=pd.read_pickle(files[0])['data']
        x_per = pd.read_pickle(per_files[0])['data']
        scores_mean=x.values[:,0]
        max_ann=np.argmax(scores_mean)
        scores_std=x.values[:,1]

        per_scores_mean = x_per.values[:,0]
        per_scores_std = x_per.values[:, 1]
        max_per = np.argmax(per_scores_mean)
        l_names=x.layer.values
        ann_layer=int(np.argwhere(l_names ==layer))
        #models_scores.append([scores_mean[ann_layer],scores_std[ann_layer]])
        models_scores.append([scores_mean[max_ann], scores_std[max_ann]])
        #per_models_scores.append([per_scores_mean[ann_layer], per_scores_std[ann_layer]])
        per_models_scores.append([per_scores_mean[max_per], per_scores_std[max_per]])

    models_scores=np.stack(models_scores)
    per_models_scores = np.stack(per_models_scores)
    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    #fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in model_layers]

    rects1 = ax.bar(x , per_models_scores[:,0], width, color=np.divide((55, 76, 128), 256), label='Pereira')
    ax.errorbar(x , per_models_scores[:, 0], yerr=per_models_scores[:, 1], linestyle='', color='k')

    rects2 = ax.bar(x + width , models_scores[:,0], width, label='ANNSet1_fMRI',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.15, 1.15))
    ax.set_xlim((-.5, 6.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_max_layer_{benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_max_layer_{benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    """compare models from the same class on the data """
    models = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large' ,'gpt2-xl' ]
    models_scores = []
    per_models_scores = []
    model_layers=[]
    for model in models:
        files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
        assert len(files)>0
        # order files
        per_files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={langloc_benchmark},model={model},*.pkl'))
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
        l_max=np.argmax(x_per.values[:, 0])
        model_layers.append(l_max)
        ann_layer=int(l_max)
        models_scores.append([scores_mean[ann_layer],scores_std[ann_layer]])
        per_models_scores.append([per_scores_mean[ann_layer], per_scores_std[ann_layer]])

    models_scores=np.stack(models_scores)
    per_models_scores = np.stack(per_models_scores)
    width = 0.35  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.2, .4, fig_length, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in zip(models,model_layers)]

    #rects1 = ax.bar(x - width / 2, per_models_scores[:,0], width, color=np.divide((55, 76, 128), 256), label='Pereira')
    #ax.errorbar(x - width / 2, per_models_scores[:, 0], yerr=per_models_scores[:, 1], linestyle='', color='k')

    rects2 = ax.bar(x + width / 2, models_scores[:,0], width, label='ANNSet1_fMRI',color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width / 2, models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    ax.set_ylim((-.1, .5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'GPT2_models_scores_{benchmark}_vs_{langloc_benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'GPT2_models_scores_{benchmark}_vs_{langloc_benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    """look at all models"""
    all_models=['roberta-base','bert-large-uncased-whole-word-masking',
    'xlnet-large-cased',
    'xlm-mlm-en-2048',
    'albert-xxlarge-v2',
    'gpt2-xl','ctrl']
    all_models_scores = []
    all_per_models_scores = []
    all_model_layers = []

    ALL_model_scores=[]
    for model in all_models:
        files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model},*.pkl'))
        assert len(files) > 0
        # order files
        per_files = glob(
            os.path.join(result_caching, 'neural_nlp.score', f'benchmark={langloc_benchmark},model={model},*.pkl'))
        assert len(per_files) > 0
        scores_mean = []
        scors_std = []
        x = pd.read_pickle(files[0])['data']
        x_per = pd.read_pickle(per_files[0])['data']
        scores_mean = x.values[:, 0]
        scores_std = x.values[:, 1]
        per_scores_mean = x_per.values[:, 0]
        per_scores_std = x_per.values[:, 1]
        l_names = x.layer.values
        l_max = np.argmax(x_per.values[:, 0])
        all_model_layers.append(l_max)
        ann_layer = int(l_max)
        all_models_scores.append([scores_mean[ann_layer], scores_std[ann_layer]])
        all_per_models_scores.append([per_scores_mean[ann_layer], per_scores_std[ann_layer]])
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
        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

        fig.savefig(os.path.join(analysis_dir, f'score_{model}_{benchmark}.eps'), format='eps', metadata=None,
                    bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    models_scores=np.stack(all_models_scores)
    per_models_scores = np.stack(all_per_models_scores)
    width = 0.35  # the width of the bars
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    fig_length = 0.04 * len(models_scores)
    ax = plt.axes((.05, .4, fig_length, .35))
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in zip(all_models,all_model_layers)]

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
    ax.set_xlim((-.5, len(models_scores)-.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'ALL_models_scores_{benchmark}_vs_{langloc_benchmark}.png'), dpi=250,
                format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'ALL_models_scores_{benchmark}_vs_{langloc_benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%% plot the response for layers of gpt2-xl and gpt2-large
    model = 'xlnet-large-cased'
    files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model},*.pkl'))
    per_files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={langloc_benchmark},model={model},*.pkl'))
    scores_mean = []
    scors_std = []
    x = pd.read_pickle(files[0])['data']
    x_per = pd.read_pickle(per_files[0])['data']
    scores_mean = x.values[:, 0]
    scores_std = x.values[:, 1]
    per_scores_mean = x_per.values[:, 0]
    per_scores_std = x_per.values[:, 1]
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)+2), len(scores_mean)+2))

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .2, .5, .2))
    scores_mean_np = np.stack(scores_mean).transpose()
    scores_std_np = np.stack(scores_std).transpose()
    r3 = np.arange(len(scores_mean_np))
    for i in r3:
        ax.plot(r3[i], scores_mean_np[i], color=all_col[i,:], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,  zorder=5)
        ax.plot(r3[i]+.2, per_scores_mean[i], color=all_col[i, :], linewidth=2, marker='s', markersize=8,
                markeredgecolor='w', markeredgewidth=1, zorder=5)
        ax.errorbar(r3[i], scores_mean_np[i], yerr=scores_std_np[i], color=all_col[i,:], zorder=3)
        ax.errorbar(r3[i]+.2, per_scores_mean[i], yerr=per_scores_std[i], color=all_col[i, :], zorder=3)
    ax.plot(r3 , scores_mean_np, color=(.5, .5, .5), linewidth=1, zorder=4)
    ax.plot(r3+.2, per_scores_mean, color=(.5, .5, .5), linewidth=1, zorder=4)

    ax.axhline(y=0, color='k', linestyle='-', zorder=2)
    # ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(l_names) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_mean_np)))
    #plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylim((-0.1, 1.1))
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} - layerwise')
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'layerwise_{model}_scores_{benchmark}_vs_{langloc_benchmark}.png'), dpi=250,
                format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'layerwise_{model}_scores_{benchmark}_vs_{langloc_benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    model = 'distilgpt2'
    files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={benchmark},model={model},*.pkl'))
    per_files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={langloc_benchmark},model={model},*.pkl'))
    scores_mean = []
    scors_std = []
    x = pd.read_pickle(files[0])['data']
    x_per = pd.read_pickle(per_files[0])['data']
    scores_mean = x.values[:, 0]
    scores_std = x.values[:, 1]
    per_scores_mean = x_per.values[:, 0]
    per_scores_std = x_per.values[:, 1]
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)+2), len(scores_mean)+2))

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .2, .9, .35))
    scores_mean_np = np.stack(scores_mean).transpose()
    scores_std_np = np.stack(scores_std).transpose()
    r3 = np.arange(len(scores_mean_np))
    for i in r3:
        ax.plot(r3[i], scores_mean_np[i], color=all_col[i,:], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,  zorder=5)
        ax.plot(r3[i]+.2, per_scores_mean[i], color=all_col[i, :], linewidth=2, marker='s', markersize=8,
                markeredgecolor='w', markeredgewidth=1, zorder=5)
        ax.errorbar(r3[i], scores_mean_np[i], yerr=scores_std_np[i], color=all_col[i,:], zorder=3)
        ax.errorbar(r3[i]+.2, per_scores_mean[i], yerr=per_scores_std[i], color=all_col[i, :], zorder=3)
    ax.plot(r3 , scores_mean_np, color=(.5, .5, .5), linewidth=1, zorder=4)
    ax.plot(r3+.2, per_scores_mean, color=(.5, .5, .5), linewidth=1, zorder=4)
    fig.show()

# model_classes = [ 'bert-', 'roberta', 'xlm','xlnet', 'ctrl', 't5', 'albert-', 'gpt']
#     color_groups = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'PuRd',
#                     'RdPu', 'BuPu',
#                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
#
#     color_ids = []
#     model_set = [x[0] for x in ALL_model_scores]
#     model_perf = []
#     model_perf_untrained = []
#     model_names = []
#     model_layer_ids = []
#     for idx, model_id in enumerate(model_set):
#
#         model_score = ALL_model_scores[idx][1]
#         model_error = ALL_model_scores[idx][2]
#         model_layer_ = ALL_model_scores[idx][3]
#         if model_id.find('untrained') == -1:
#             model_perf.append((model_score,model_error))
#             model_names.append(f"{model_id} ({model_layer_})")
#             model_layer_ids.append(model_layer_)
#             # get untrained score
#         if model_id == 'xlm-roberta-base' or model_id == 'xlm-roberta-large':
#             color_loc = 8
#         else:
#             color_loc = int(np.argwhere([model_id.find(x) != -1 for x in model_classes])[-1].squeeze())
#             assert color_loc is not None
#             color_ids.append(color_loc)
#
#     model_perf = np.array(model_perf)
#     num_cols = [len(np.where(np.asarray(color_ids) == x)[0]) for idx, x in enumerate(np.unique(color_ids))]
#     h0s = [cm.get_cmap(color_groups[x], num_cols[idx] + 2) for idx, x in enumerate(np.unique(color_ids))]
#     all_colors = [np.flipud(x(np.arange(num_cols[idx]) / (num_cols[idx] + 1))) for idx, x in enumerate(h0s)]
#     # all_colors=[x(np.arange(num_cols[idx])/(num_cols[idx]+1)) for idx, x in enumerate(h0s)],
#     all_colors = [item for sublist in all_colors for item in sublist]
#
#     y_pos = np.arange(len(model_names))
#     fig = plt.figure(figsize=(8, 10))
#     ax = fig.add_axes((.5, .1, .4, .85))
#     ax.barh(y_pos, np.asarray(model_perf)[:, 0], height=.8, xerr=np.asarray(model_perf)[:, 1], align='center',
#             color=np.asarray(all_colors), edgecolor=(0, 0, 0), linewidth=1, error_kw={'linewidth': 1})
#     # ax.barh(y_pos, np.asarray(model_perf_untrained)[:,0],height=0.8, align='center',color=np.asarray(all_colors),edgecolor=(.2,.2,.2),linewidth=1)
#     #ax.barh(y_pos, np.asarray(model_perf_untrained)[:, 0], height=0.8, align='center', color=(.8, .8, .8),
#     #        edgecolor=(.2, .2, .2), linewidth=1, alpha=.7)
#     # ax.scatter(np.asarray(model_perf_untrained)[:,0],y_pos,s=20,zorder=10)
#     # add a vertical line at zero
#     ax.axvline(0, color='k', linewidth=1)
#     ax.axvline(1, color='k',linestyle='--', linewidth=1)
#     ax.set_yticks(y_pos)
#     ax.set_ylim(y_pos.min() - 1, y_pos.max() + 1)
#     ax.set_yticklabels(model_names, fontsize=8, fontweight='normal')
#     ax.tick_params(axis='x', which='both', labelsize=8)
#     ax.set_xlabel('score', fontsize=8)
#     plt.xticks(fontsize=8)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_title('Model performance on %s'%benchmark,fontsize=10)
#
#     fig.savefig(os.path.join(analysis_dir, f'scores_{benchmark}.pdf'), transparent=True)