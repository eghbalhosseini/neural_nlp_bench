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
from scipy import linalg
from scipy.spatial import distance

if user=='eghbalhosseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    stimuli_name='Pereira2018-*'
    model_list=['mistral-caprica-gpt2-small-x81-ckpnt-400000-untrained-1',
                'mistral-caprica-gpt2-small-x81-ckpnt-400000-untrained-4',
                'mistral-caprica-gpt2-small-x81-ckpnt-400000']
    model_norms=[]
    model_scores=[]
    model_corrs=[]
    for model in model_list:
        benchmark = 'Pereira2018-encoding'
        print(model)
        file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                   f'benchmark={benchmark},model={model},subsample=None.pkl'))
        x = pd.read_pickle(file_c[0])['data'].values
        model_scores.append((x[:, 0],x[:, 1]))

        activtiy_folder='neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored'
        if model== 'mistral-caprica-gpt2-small-x81-ckpnt-400000':
            model='mistral_caprica-gpt2-small-x81_ckpt_400000'
            files = glob(os.path.join(result_caching, activtiy_folder,
                                      f'identifier={model},stimuli_identifier={stimuli_name}*.pkl'))
        else:
            files = glob(os.path.join(result_caching, activtiy_folder,
                                      f'identifier={model},stimuli_identifier={stimuli_name}*.pkl'))
        # order files
        print(model)
        files_srt=files
        # remove chekcpoint zero if exist
        norms=[]
        corrs=[]
        for ix, file in tqdm(enumerate(files_srt)):
            x=pd.read_pickle(file)['data']
            layer_norm=[]
            layer_corr=[]
            for grp in x.groupby('layer'):
                layer_norm.append(linalg.norm(grp[1], axis=1))
                layer_corr.append(np.expand_dims(distance.pdist(grp[1],metric='correlation'),axis=0))
            norms.append(np.stack(layer_norm))
            corrs.append(np.concatenate(layer_corr,axis=0))
        model_norms.append(np.concatenate(norms,axis=1))
        model_corrs.append(np.concatenate(corrs,axis=1))

    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(model_scores)), len(model_scores)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .1, .35, .35))
    for idx, scr_ in enumerate(model_scores):
        scr=scr_[0]
        scr_std=scr_[1]
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx, :], linewidth=2, marker='.', markersize=10,
                label=f'ck:')
        ax.fill_between(r3, scr - scr_std, scr + scr_std, facecolor=all_col[idx, :], alpha=0.1)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(scr) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson Corr')
    ylims=(-.1,1)
    ax.set_ylim(ylims)
    plt.grid(True, which="both", ls="-", color='0.9')
    #fig.show()
    ax = plt.axes((.6, .1, .35, .35))
    for idx, norms in enumerate(model_norms):
        scr = np.mean(norms,axis=1)
        scr_std = np.std(norms,axis=1)
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx, :], linewidth=2, marker='.', markersize=10,
                label=f'ck:')
        ax.fill_between(r3, scr - scr_std, scr + scr_std, facecolor=all_col[idx, :], alpha=0.1)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(scr) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 norm of activations')
    #ax.set_ylim(ylims)
    plt.grid(True, which="both", ls="-", color='0.9')
    ax = plt.axes((.6, .6, .35, .35))
    for idx, norms in enumerate(model_corrs):
        scr = np.mean(norms, axis=1)
        scr_std = np.std(norms, axis=1)
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx, :], linewidth=2, marker='.', markersize=10,
                label=f'ck:')
        ax.fill_between(r3, scr - scr_std, scr + scr_std, facecolor=all_col[idx, :], alpha=0.1)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(scr) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 norm of activations')
    # ax.set_ylim(ylims)
    plt.grid(True, which="both", ls="-", color='0.9')

    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'score_vs_activations_for_layer_norm_variation.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'score_vs_activations_for_layer_norm_variation.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)