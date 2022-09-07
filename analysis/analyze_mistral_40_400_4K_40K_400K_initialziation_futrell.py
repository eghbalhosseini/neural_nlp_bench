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
    benchmark='Blank2014fROI-encoding'
    ylims=(.5,0.7)
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
    precomputed = pd.read_csv('/om/user/ehoseini/neural-nlp-2022/precomputed-scores.csv')
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
                scores_mean.append(np.asarray([np.nan]))
                scors_std.append(np.asarray([np.nan]))
        model_scores.append(scores_mean)
        model_scores_std.append(scors_std)

    # read precomputed scores


    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(model_scores)+1), len(model_scores)+1))
    all_col= all_col[1:,:]



    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .35, .35))
    x_coords = [ 0.01, 0.1, 1, 10,100]
    scrs_layer_np=np.squeeze(np.stack(model_scores))

    for id_mod,scr_layer in enumerate(scrs_layer_np):
        markers=['o','s','d']
        for idx, scr in enumerate(scr_layer):
            ax.plot(x_coords[id_mod], scr, color=all_col[id_mod, :], linewidth=2, marker=markers[idx], markersize=10,
                label=f'{models[idx]}', zorder=2,markeredgecolor='k')
    for x in scrs_layer_np.transpose():
        ax.plot(x_coords, x, color='k', linewidth=1, zorder=1)
            #ax.errorbar(x_coords[id_mod], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
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

