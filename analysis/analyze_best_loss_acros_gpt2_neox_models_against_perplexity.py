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
    benchmark='Pereira2018-encoding'
    #benchmark = 'Blank2014fROI-encoding'
    #benchmark = 'Futrell2018-encoding'
    #benchmark = 'Fedorenko2016v3-encoding'
    model_1B='gpt2-neox-pos_learned-1B'
    loss_1B_ckpnt='310000'
    model_100M = 'gpt2-neox-pos_learned-100M'
    #loss_100M_ckpnt='11600'
    loss_100M_ckpnt='14250'
    model_10M = 'gpt2-neox-pos_learned-10M'
    #loss_10m_ckpnt='1900'
    loss_10M_ckpnt = '2000'
    file_1B=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{loss_1B_ckpnt}*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_100M}-v2-ckpnt-{loss_100M_ckpnt}*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_10M}-v2-ckpnt-{loss_10M_ckpnt}*.pkl'))
    files_srt=[file_10M[0],file_100M[0],file_1B[0]]
    chkpoints_srt=['10M','100M','1B']
    # order files
    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])
    # get perplexity results:
    score_max=[max(x) for x in scores_mean]
    score_loc=[np.argmax(x) for x in scores_mean]
    score_std=[scors_std[x][score_loc[x]] for x in range(len(score_loc))]
    training_results = pd.read_pickle(os.path.join(result_dir, 'data', 'miniberta_train_valid_set.pkl'))
    # find the location of perplexity score closest to checkpoint:
    training_chkpnt = [training_results[key]['validation_result'][:, 0].astype(int) for key in training_results.keys()]
    training_perplex = [training_results[key]['validation_result'][:, 2] for key in
                       training_results.keys()]
    # reorder them to be 10m, 100m, 1B
    training_chkpnt.reverse()
    training_perplex.reverse()
    #
    model_perplex=[]
    for idx, los_chk in enumerate([loss_10M_ckpnt,loss_100M_ckpnt,loss_1B_ckpnt]):
        chkpoints=training_chkpnt[idx]
        loc=np.argmin(np.abs(chkpoints-int(los_chk)))
        model_perplex.append(training_perplex[idx][loc])

    validation_perpelxity=np.asarray(model_perplex)
    validation_score=np.asarray(score_max)

    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5,.5,.5))
    for idx in range(len(validation_score)):
        ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
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

    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    ax.plot(validation_perpelxity, validation_score, zorder=1, color=(.5,.5,.5))
    for idx in range(len(validation_score)):
        ax.scatter(validation_perpelxity[idx], validation_score[idx],s=50, c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.errorbar(validation_perpelxity[idx], validation_score[idx], yerr=score_std[idx], linewidth=2, color=all_col[idx, :], marker='.', markersize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr')
    ax.set_xlabel('perplexity')
    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)
    ax.set_title(f'model:gpt_neox \n benchmark {benchmark} against perplexity \n for best layer in 1B ')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_for_best_layer_in_1b.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox_{benchmark}_against_perplexity_in_1b.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)