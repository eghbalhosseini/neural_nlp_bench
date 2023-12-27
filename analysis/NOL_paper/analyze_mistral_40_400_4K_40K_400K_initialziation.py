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
    benchmark='Pereira2018-v2-encoding'
    benchmark='Blank2014fROI-encoding'
    ylims=(-0.1,0.5)
    #benchmark = 'Fedorenko2016v3-encoding'
    benchmark='Futrell2018-encoding'
    model='mistral-caprica-gpt2-small-x81'
    models=['mistral-caprica-gpt2-small-x81', 'alias-gpt2-small-x21', 'expanse-gpt2-small-x777']
    chkpnts=[40,400,4000,40000,400000]
    precomputed_model = 'gpt2'
    model_files_c=[]
    for ckpnt in chkpnts:
        files_ckpnt = []
        for model in models:
            if ckpnt==0:
                ckpnt=str(ckpnt)+'-untrained'
            else:
                ckpnt = str(ckpnt) + ''
            file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},subsample=None.pkl'))
            print(file_c)
            if len(file_c)>0:
                files_ckpnt.append(file_c[0])
            else:
                files_ckpnt.append([])
        model_files_c.append(files_ckpnt)

    chkpoints_srt=['0.01% (n=40)','0.1% (n=400)' ,'1% (n=4K)','10% (n=40K)','100% (n=400K)']
    precomputed = pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']

    # order files
    model_scores=[]
    model_scores_std=[]
    for files_ckpnt in model_files_c:
        scores_mean = []
        scors_std = []
        for ix, file in tqdm(enumerate(files_ckpnt)):
            if len(file)!=0:
                x=pd.read_pickle(file)['data'].values
                scores_mean.append(x[:,0])
                scors_std.append(x[:,1])
            else:
                scores_mean.append(np.nan)
                scors_std.append(np.nan)
        model_scores.append(scores_mean)
        model_scores_std.append(scors_std)

    # read precomputed scores
    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(model_scores)+1), len(model_scores)+1))
    all_col= all_col[1:,:]
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for id_mod,scores_mean in enumerate(model_scores):

        scors_std=model_scores_std[id_mod]
        scr=np.stack(scores_mean).mean(axis=0)
        scr_std=np.stack(scores_mean).std(axis=0)
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[id_mod, :], linewidth=2, marker='.', markersize=10,
                label=f'ck:{chkpoints_srt[id_mod]}')
        ax.errorbar(r3,scr,yerr=scr_std,fmt='',color=all_col[id_mod, :],linewidth=2)
        #ax.fill_between(r3, scr-scors_std[idx],scr+scors_std[idx], facecolor=all_col[id_mod, :],alpha=0.1)
    # add precomputed
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1., .8), frameon=True,fontsize=8)
    ax.set_xlim((0-.5,len(l_names)-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylim(ylims)
    plt.grid(True, which="both", ls="-", color='0.9')
    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} \n model:{model}-permuted')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'mistral_effect_of_initialzation_{benchmark}.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'mistral_effect_of_initialzation_{benchmark}.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    # plot for best layer of Shrimpf study
    layer_id = np.argmax(model_bench['score'])

    scrs_layer=[]
    for id_mod, scores_mean in enumerate(model_scores):
        scr_layer = [x[layer_id] for x in scores_mean]
        scrs_layer.append(scr_layer)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .35, .35))
    x_coords = [ 0.01, 0.1, 1, 10,100]
    scrs_layer_np=np.stack(scrs_layer)
    for col in scrs_layer_np.transpose():
        ax.plot(x_coords, col, color=(.3,.3,.3), linewidth=1, zorder=1)
    for id_mod,scr_layer in enumerate(scrs_layer):
        markers=['o','s','d']
        for idx, scr in enumerate(scr_layer):
            ax.plot(x_coords[id_mod], scr, color=all_col[id_mod, :], linewidth=2, marker=markers[idx], markersize=10,
                label=f'{models[idx]}', zorder=2,markeredgecolor='k')
            ax.errorbar(x_coords[id_mod], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
    ax.set_xscale('log')
    ax.axhline(y=0, color='k', linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate([ np.arange(1, 11) * 1e-2,np.arange(1, 11) * 1e-1,np.arange(1, 11) * 1e0,np.arange(1, 11) * 1e1])

    ax.set_xticks(np.concatenate([major_ticks]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)
    chkpoints_label = [ '0.01%', '0.1%', '1%', '10%',
                     '100%',]
    ax.set_xticklabels(chkpoints_label, rotation=0)
    ax.set_ylim(ylims)
    ax.legend(bbox_to_anchor=(1.6, .8), frameon=True, fontsize=8)

    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'mistral_effect_of_initialzation_schrimpf_layer_through_training_{benchmark}.png'), dpi=250,
                format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'mistral_effect_of_initialzation_schrimpf_layer_through_training_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

