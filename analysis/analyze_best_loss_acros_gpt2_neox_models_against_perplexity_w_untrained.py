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
    #ylims=(-.1,0.4)
    benchmark = 'Futrell2018-encoding'
    ylims=(.2,0.9)
    #benchmark = 'Fedorenko2016v3-encoding'
    model_1B='gpt2-neox-pos_learned-1B'
    loss_1B_ckpnt='310000'
    model_100M = 'gpt2-neox-pos_learned-100M'
    #loss_100M_ckpnt='11600'
    loss_100M_ckpnt='14250'
    model_10M = 'gpt2-neox-pos_learned-10M'
    #loss_10m_ckpnt='1900'
    loss_10M_ckpnt = '2000'
    model_1M = 'gpt2-neox-pos_learned-1M'
    loss_1M_ckpnt = '1000'
    version = 'v3'


    file_1B = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1B}-{version}-ckpnt-{loss_1B_ckpnt},*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_100M}-{version}-ckpnt-{loss_100M_ckpnt},*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_10M}-{version}-ckpnt-{loss_10M_ckpnt},*.pkl'))
    version='v2'
    file_1M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1M}-{version}-ckpnt-{loss_1M_ckpnt},*.pkl'))

    files_srt = [ file_1M[0], file_10M[0], file_100M[0], file_1B[0]]
    chkpoints_srt = ['untrained', '1M', '10M', '100M', '1B']

    chkpoints_srt = ['untrained', '1M', '10M', '100M', '1B']
    wikitext_perplexity=[50340.6719,2351.0483,639.1099,206.9432,68.1662]
    precomputed_model = 'gpt2'
    precomputed = pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']


    file_1B_untrained_hf = glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark=Futrell2018-encoding,model=gpt2-neox-pos_learned-1B-v2-ckpnt-310000-untrained,subsample=None.pkl'))
    file_1B_untrained=file_1B_untrained_hf[0].replace('_hf','')
    with open(file_1B_untrained_hf[0], 'rb') as f:
        score_un_hf=pickle.load(f)
    with open(file_1B_untrained, 'rb') as f:
        score_un=pickle.load(f)

    # order files
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_srt)):
        #x=pd.read_pickle(file)['data'].values
        with open(file, 'rb') as f:
            x=pickle.load(f)
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])
    # get perplexity results:
    layer_id = np.argmax(model_bench['score'])
    layer_name = model_bench['layer'].iloc[layer_id]
    score_loc = [x[layer_id] for x in scores_mean]
    score_std = [x[layer_id] for x in scors_std]


    validation_perpelxity=np.asarray(wikitext_perplexity)
    validation_score=np.asarray(score_loc)

    gpt2_unt_hg=56145.7422
    score_bench=precomputed_bench[precomputed_bench['layer']==layer_name]['score'].values[0]
    score_bench_unt=score_untrained_hf[layer_id][0].values
    score_bench_unt_std=score_untrained_hf[layer_id][1].values


    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    #all_col= all_col[1:,:]
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.set_xscale('log')
    ax.plot(validation_perpelxity, validation_score, zorder=2, color=(0,0,.5))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx], validation_score[idx], color=(all_col[idx, :]), linewidth=2, marker='o',
                markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color=all_col[idx, :], marker=None, markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)
    ax.plot(gpt2_unt_hg, score_bench_unt, color=all_col[0, :], linewidth=2, marker='o', markeredgecolor='w',
            markersize=10, label=f'HF_untrained', zorder=2)
    ax.errorbar(gpt2_unt_hg, score_bench_unt, yerr=score_bench_unt_std, color='k', zorder=1)
    ax.set_ylim(ylims)


    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_title(f'model:gpt_neox \n benchmark {benchmark} against perplexity')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)


    #%% do the same analysis but take the 1b model and find how it behaves

    score_max_1b = np.argmax(scores_mean[-1])
    scores_max= [scores_mean[x][score_max_1b] for x in range(len(scores_mean))]
    score_std = [scors_std[x][score_max_1b] for x in range(len(scores_mean))]
    validation_score = np.asarray(scores_max)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    ax.set_xscale('log')
    ax.plot(validation_perpelxity, validation_score, zorder=3, color=(0,0,0))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx],validation_score[idx], color=(all_col[idx,:]), linewidth=2, marker='o', markersize=8, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        #ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=4,label=chkpoints_srt[idx])
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    #minor_ticks = np.concatenate(
    #    [np.arange(5, 11,1) * 1e1, np.arange(1, 11,1) * 1e2,np.arange(1, 11,1) * 1e3,np.arange(1, 6,1) * 1e4])
    #ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')

    #minor_ticks = np.unique(np.concatenate(
    #    [np.arange(5, 11,2) * 1e1, np.arange(1, 11,4) * 1e2,np.arange(1, 11,4) * 1e3,np.arange(1, 6,2) * 1e4]))

    #ax.set_xticks(np.unique(minor_ticks))
    #ax.set_xticklabels(np.unique(minor_ticks).astype(int))
    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)
    ax.set_ylim(ylims)
    ax.set_title(f'model:gpt_neox \n benchmark {benchmark} against perplexity \n for best layer in 1B ')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_for_best_layer_in_1b_w_untrained.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_for_best_layer_in_1b_w_untrained.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
#%% schirmpf layer

    layer_id = np.argmax(model_bench['score'])
    scr_layer = [x[layer_id] for x in scores_mean]
    scr_layer_std = [x[layer_id] for x in scors_std]
    validation_score=scr_layer
    score_std=scr_layer_std

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .2, .45, .45))
    ax.set_xscale('log')
    ax.plot(validation_perpelxity, validation_score, zorder=3, color=(0,0,0))
    for idx in range(len(validation_score)):
        ax.plot(validation_perpelxity[idx],validation_score[idx], color=(all_col[idx,:]), linewidth=2, marker='o', markersize=10, markeredgecolor='k',
                markeredgewidth=1, zorder=5)
        #ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=4,label=chkpoints_srt[idx])
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color='k', marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    #minor_ticks = np.concatenate(
    #    [np.arange(5, 11,1) * 1e1, np.arange(1, 11,1) * 1e2,np.arange(1, 11,1) * 1e3,np.arange(1, 6,1) * 1e4])
    #ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')

    #minor_ticks = np.unique(np.concatenate(
    #    [np.arange(5, 11,2) * 1e1, np.arange(1, 11,4) * 1e2,np.arange(1, 11,4) * 1e3,np.arange(1, 6,2) * 1e4]))

    #ax.set_xticks(np.unique(minor_ticks))
    #ax.set_xticklabels(np.unique(minor_ticks).astype(int))
    #ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_axisbelow(True)
    ax.set_ylim(ylims)
    ax.set_title(f'model:gpt_neox \n benchmark {benchmark} against perplexity \n for schrimpf layer')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_for_schrimpf.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_for_schrimpf.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
