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
    #benchmark='Pereira2018-encoding'
    #ylims = (-.2, 1.1)
    benchmark = 'Blank2014fROI-encoding'
    ylims = (-.2, .5)
    #benchmark = 'Futrell2018-encoding'
    #ylims = (.2, .9)

    model = 'mistral-caprica-gpt2-small-x81'
    precomputed_model = 'gpt2'


    chkpnts=[0,40,400,4000,40000,400000]
    files_ckpnt = []
    for ckpnt in chkpnts:
        if ckpnt == 0:
            ckpnt = str(ckpnt) + '-untrained'
        file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                   f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},subsample=None.pkl'))
        print(file_c)
        if len(file_c) > 0:
            files_ckpnt.append(file_c[0])
        else:
            files_ckpnt.append([])
    chkpoints_srt = ['untrained (n=0)', '0.01% (n=40)', '0.1% (n=400)', '1% (n=4K)', '10% (n=40K)', '100% (n=400K)']

    precomputed = pd.read_csv('/om/user/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']

    # order files
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_ckpnt)):
        if len(file)>0:
            x=pd.read_pickle(file)['data'].values
            scores_mean.append(x[:,0])
            scors_std.append(x[:,1])
        else:
            scores_mean.append(np.nan)
            scors_std.append(np.nan)
    # get perplexity results:
    score_max=[max(x) for x in scores_mean]
    score_loc=[np.argmax(x) for x in scores_mean]
    score_std=[scors_std[x][score_loc[x]] for x in range(len(score_loc))]
    #
    preplex_benchmark='wikitext-103-raw-v1-test'
    training_perplex=[50324.4453,4392.1587,885.9544,61.31,42.1,32.75]

    #training_perplex = [4392.1587, 885.9544, 42.1, 32.75]

    validation_perpelxity=np.asarray(training_perplex)
    validation_score=np.asarray(score_max)

    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5,.5,.5))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2,
                    color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.set_xscale('log')
    ax.invert_xaxis()
    #minor_ticks = np.concatenate(
    #    [np.arange(2, 11) * 1e1, np.arange(1, 11) * 1e2,np.arange(1, 5) * 1e3])
    #ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    #minor_ticks = np.concatenate(
    #    [np.arange(2, 11,2) * 1e1, np.arange(1, 11,2) * 1e2,np.arange(1, 5,2) * 1e3])

    #ax.set_xticks(np.unique(minor_ticks))
    #ax.set_xticklabels(np.unique(minor_ticks).astype(int))

    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)

    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)


    #%% do the same analysis but take the 1b model and find how it behaves

    model_with_max = scores_mean[np.argmax(np.max(np.stack(scores_mean),axis=1))]
    score_max_1b=np.argmax(model_with_max)
    scores_max= [scores_mean[x][score_max_1b] for x in range(len(scores_mean))]
    score_std = [scors_std[x][score_max_1b] for x in range(len(scores_mean))]
    validation_score = np.asarray(scores_max)

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=3, color=(0,0,0))
    for idx in range(len(validation_score)):
        #ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr for best layer in model')
    ax.set_xlabel('perplexity')
    ax.set_xscale('log')
    ax.invert_xaxis()
    minor_ticks = np.concatenate(
        [np.arange(2, 11) * 1e1, np.arange(1, 11) * 1e2,np.arange(1, 11) * 1e3, np.arange(1, 6) * 1e4])
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    #minor_ticks = np.concatenate(
    #    [np.arange(2, 11, 4) * 1e1, np.arange(1, 11, 4) * 1e2, np.arange(1, 6, 4) * 1e3])
    #minor_ticks=[  20.,   40.,  100.,  200.,400, 1000.,2000, 4000.]
    #ax.set_xticks(np.unique(minor_ticks))
    #ax.set_xticklabels(np.unique(minor_ticks).astype(int))

    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)
    ax.set_ylim(ylims)

    ax.legend(bbox_to_anchor=(1.6, .8), frameon=True, fontsize=8)
    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}_for_best_layer_in_model_w_untrained.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}_for_best_layer_in_model_w_untrainend.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
#%% analysis based on Schrimpf
    layer_id = np.argmax(model_bench['score'])
    scr_layer = [x[layer_id] for x in scores_mean]
    scr_layer_std = [x[layer_id] for x in scors_std]
    validation_score=scr_layer
    score_std=scr_layer_std
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5, .5, .5))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2,
                    color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.set_xscale('log')
    ax.invert_xaxis()
    # minor_ticks = np.concatenate(
    #    [np.arange(2, 11) * 1e1, np.arange(1, 11) * 1e2,np.arange(1, 5) * 1e3])
    # ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    # minor_ticks = np.concatenate(
    #    [np.arange(2, 11,2) * 1e1, np.arange(1, 11,2) * 1e2,np.arange(1, 5,2) * 1e3])

    # ax.set_xticks(np.unique(minor_ticks))
    # ax.set_xticklabels(np.unique(minor_ticks).astype(int))

    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)

    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}_schrimpf_layer.png'),
        dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}_against_perplexity_{preplex_benchmark}_schrimpf_layer.eps'),
        format='eps', metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
