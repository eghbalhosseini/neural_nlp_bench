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
    #ylims = (-0.1, 1)
    benchmark='Blank2014fROI-encoding'
    ylims = (-0.1, .5)


    benchmark='Futrell2018-encoding'
    ylims = (.2, .8)



    model_1B = 'gpt2-neox-pos_learned-1B'
    precomputed_model = 'gpt2'
    loss_1B_ckpnt = '310000'
    model_100M = 'gpt2-neox-pos_learned-100M'
    # loss_100M_ckpnt='11600'
    loss_100M_ckpnt = '14250'
    model_10M = 'gpt2-neox-pos_learned-10M'
    loss_10M_ckpnt = '2250'

    model_1M = 'gpt2-neox-pos_learned-1M'
    loss_1M_ckpnt = '1000'

    # loss_10M_ckpnt = '2000'
    file_1B_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{2500}-untrained*.pkl'))
    file_1B = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{loss_1B_ckpnt}*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_100M}-v2-ckpnt-{loss_100M_ckpnt}*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_10M}-v2-ckpnt-{loss_10M_ckpnt}*.pkl'))
    file_1M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1M}-v2-ckpnt-{loss_1M_ckpnt}*.pkl'))
    #files_srt_1 = [file_1B_untrained[0], file_1M[0], file_10M[0], file_100M[0], file_1B[0]]
    files_srt_1 = [ file_1M[0], file_10M[0], file_100M[0], file_1B[0]]


    model_1B = 'gpt2-neox-pos_learned-1B-v2-init2'
    loss_1B_ckpnt = '107500'
    model_100M = 'gpt2-neox-pos_learned-100M-v2-init2'
    # loss_100M_ckpnt='11600'
    loss_100M_ckpnt = '17500'
    model_10M = 'gpt2-neox-pos_learned-10M-v2-init2'
    loss_10M_ckpnt = '2000'

    model_1M = 'gpt2-neox-pos_learned-1M-v2-init2'
    loss_1M_ckpnt = '1000'

    # loss_10M_ckpnt = '2000'
    file_1B = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1B}-ckpnt-{loss_1B_ckpnt}*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_100M}-ckpnt-{loss_100M_ckpnt}*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_10M}-ckpnt-{loss_10M_ckpnt}*.pkl'))
    file_1M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_1M}-ckpnt-{loss_1M_ckpnt}*.pkl'))
    #files_srt_2 = [file_1B_untrained[0], file_1M[0], file_10M[0], file_100M[0], file_1B[0]]
    files_srt_2 = [ file_1M[0], file_10M[0], file_100M[0], file_1B[0]]


    precomputed_model = 'gpt2'
    precomputed = pd.read_csv('/om/user/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']
    model_files_c=[[files_srt_1[0],files_srt_2[0]],
                [files_srt_1[1], files_srt_2[1]],
                [files_srt_1[2], files_srt_2[2]],
                [files_srt_1[3], files_srt_2[3]],
                #[files_srt_1[4], files_srt_2[4]]]
                   ]
    # order files
    model_scores=[]
    model_scores_std=[]
    for files_ckpnt in model_files_c:
        scores_mean = []
        scors_std = []
        for ix, file in tqdm(enumerate(files_ckpnt)):
            x=pd.read_pickle(file)['data'].values
            scores_mean.append(x[:,0])
            scors_std.append(x[:,1])
        model_scores.append(scores_mean)
        model_scores_std.append(scors_std)

    # read precomputed scores
    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(model_scores)+1), len(model_scores)+1))
    all_col=all_col[1:,:]

    # plot for best layer of Shrimpf study
    layer_id = np.argmax(model_bench['score'])

    scrs_layer=[]

    for id_mod, scores_mean in enumerate(model_scores):
        scr_layer = [x[layer_id] for x in scores_mean]
        scrs_layer.append(scr_layer)

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .4, .35, .35))
    #x_coords = [1e5, 1e6, 10e6, 100e6, 1000e6]
    x_coords = [ 1e6, 10e6, 100e6, 1000e6]
    scrs_layer_np=np.stack(scrs_layer)
    for col in scrs_layer_np.transpose():
        ax.plot(x_coords, col, color=(.3,.3,.3), linewidth=1, zorder=1)
    for id_mod,scr_layer in enumerate(scrs_layer):
        markers=['o','s','d']
        for idx, scr in enumerate(scr_layer):
            ax.plot(x_coords[id_mod], scr, color=all_col[id_mod, :], linewidth=2, marker=markers[idx], markersize=10,
                label=f'init{idx}', zorder=2,markeredgecolor='k')
            #ax.errorbar(x_coords[id_mod], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
    ax.set_xscale('log')
    ax.axhline(y=0, color='k', linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate(
        [np.arange(1, 11) * 1e5, np.arange(1, 11) * 1e6, np.arange(1, 11) * 1e7, np.arange(1, 11) * 1e8])
    minor_ticks = np.concatenate(
        [ np.arange(1, 11) * 1e6, np.arange(1, 11) * 1e7, np.arange(1, 11) * 1e8])

    ax.set_xticks(np.concatenate([major_ticks]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)
    #ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B'], rotation=0)
    ax.set_xticklabels(['1M', '10M', '100M', '1B'], rotation=0)

    ax.set_ylim(ylims)
    ax.legend(bbox_to_anchor=(1.6, .8), frameon=True, fontsize=8)

    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'gpt2_neox_effect_of_initialzation_schrimpf_layer_through_training_{benchmark}.png'), dpi=250,
                format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'gpt2_neox_effect_of_initialzation_schrimpf_layer_through_training_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

