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
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmark='Pereira2018-encoding'
    ylims = (-.12, 1.1)
    #benchmark = 'Blank2014fROI-encoding'
    #ylims = (-.2, .5)
    benchmark = 'Futrell2018-encoding'
    ylims = (.2, .9)

    model = 'mistral-caprica-gpt2-small-x81'
    precomputed_model = 'gpt2'


    chkpnts=[0,20,200,2000,20000,200000]
    files_ckpnt = []
    for ckpnt in chkpnts:
        if ckpnt == 0:
            ckpnt = str(ckpnt) + '-untrained'
            file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                   f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},subsample=None.pkl'))
            #file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
            #                           f'benchmark={benchmark},model=gpt2-untrained,subsample=None.pkl'))
        else:
            file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},subsample=None.pkl'))
        print(file_c)
        if len(file_c) > 0:
            files_ckpnt.append(file_c[0])
        else:
            files_ckpnt.append([])
    chkpoints_srt=['n=0','n=20','n=200' ,'n=2K','n=20K','n=200K']

    precomputed = pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']
    # load checkpoint 0
    files_ckpnt_0=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}-ckpnt-0,subsample=None.pkl'))
    score_data_chkpnt_0=pd.read_pickle(files_ckpnt_0[0])['data'].values

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
    layer_id = np.argmax(model_bench['score'])
    layer_name = model_bench['layer'].iloc[layer_id]
    score_loc = [x[layer_id] for x in scores_mean]
    scr_layer_std = [x[layer_id] for x in scors_std]

    #score_max=[max(x) for x in scores_mean]
    #score_loc=[np.argmax(x) for x in scores_mean]
    #scr_layer_std=[scors_std[x][score_loc[x]] for x in range(len(score_loc))]
    #
    preplex_benchmark='wikitext-103-raw-v1-test'
    training_perplex=[53099.7070,   #0
                      21383.7090,   #20
                      1439.5457,    #200
                      75.1746,      #2000
                      43.0601,      #20000
                      35.9569       #200000
                     ]
    #training_perplex=[50324.4453,4392.1587,885.9544,61.31,42.1,32.75]
    gpt2_perp=29.17378
    gpt2_unt_hg=56145.7422
    ckpnt_0_per=54000.9102
    score_bench=precomputed_bench[precomputed_bench['layer']==layer_name]['score'].values[0]
    score_bench_unt=precomputed_bench[precomputed_bench['layer']==layer_name]['score'].values[1]
    score_bench_unt_std=precomputed_bench[precomputed_bench['layer']==layer_name]['error'].values[1]

    score_chkpnt_0=score_data_chkpnt_0[layer_id][0]
    score_chkpnt_0_std=score_data_chkpnt_0[layer_id][0]
    #training_perplex = [4392.1587, 885.9544, 42.1, 32.75]

    validation_perpelxity=np.asarray(training_perplex)
    validation_score=np.asarray(score_loc)

    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5,.5,.5))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=scr_layer_std[idx], linewidth=2,
                    color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_ylim(ylims)

    ax.plot(ckpnt_0_per, score_chkpnt_0, color=all_col[0, :], linewidth=2, marker='o', markeredgecolor='w',
            markersize=10, label=f'HF_untrained', zorder=2)
    ax.errorbar(ckpnt_0_per, score_chkpnt_0, yerr=score_chkpnt_0_std, color='k', zorder=1)
    #minor_ticks = np.concatenate(
    #    [np.arange(2, 11) * 1e1, np.arange(1, 11) * 1e2,np.arange(1, 5) * 1e3])
    #ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    #minor_ticks = np.concatenate(
    #    [np.arange(2, 11,2) * 1e1, np.arange(1, 11,2) * 1e2,np.arange(1, 5,2) * 1e3])

    #ax.set_xticks(np.unique(minor_ticks))
    #ax.set_xticklabels(np.unique(minor_ticks).astype(int))
    ax.set_ylim(ylims)

    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)

    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)


    #%% do the same analysis but take the 1b model and find how it behaves

    model_with_max = scores_mean[np.argmax(np.max(np.stack(scores_mean),axis=1))]
    score_max_1b=np.argmax(model_with_max)
    scores_max= [scores_mean[x][score_max_1b] for x in range(len(scores_mean))]
    score_std = [scors_std[x][score_max_1b] for x in range(len(scores_mean))]
    validation_score = np.asarray(scores_max)

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .45))
    ax.plot(validation_perpelxity, validation_score, zorder=3, color=(0,0,0))
    for idx in range(len(validation_score)):
        #ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=10, markeredgecolor='k',
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

    #ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)
    ax.set_ylim(ylims)

    ax.legend(bbox_to_anchor=(1.6, .8), frameon=True, fontsize=8)
    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}_for_best_layer_in_model_w_untrained.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}_for_best_layer_in_model_w_untrainend.eps'), format='eps',metadata=None,
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
    ax = plt.axes((.1, .2, .45, .45))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5, .5, .5))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2,
                    color='k', marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_ylim(ylims)
    # minor_ticks = np.concatenate(
    #    [np.arange(2, 11) * 1e1, np.arange(1, 11) * 1e2,np.arange(1, 5) * 1e3])
    # ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    # minor_ticks = np.concatenate(
    #    [np.arange(2, 11,2) * 1e1, np.arange(1, 11,2) * 1e2,np.arange(1, 5,2) * 1e3])

    # ax.set_xticks(np.unique(minor_ticks))
    # ax.set_xticklabels(np.unique(minor_ticks).astype(int))

    #ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)

    ax.set_title(f'model:{model} \n benchmark {benchmark} against \n perplexity {preplex_benchmark}')
    fig.show()
    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}_schrimpf_layer.png'),
        dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_{model}_20_{benchmark}_against_perplexity_{preplex_benchmark}_schrimpf_layer.eps'),
        format='eps', metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
