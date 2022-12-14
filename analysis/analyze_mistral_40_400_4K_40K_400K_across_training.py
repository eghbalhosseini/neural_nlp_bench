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
    ylims = (-.12, 1.1)
    #benchmark='Blank2014fROI-encoding'
    #ylims=(-.2,.5)
    #benchmark = 'Fedorenko2016v3-encoding'
    #benchmark='Futrell2018-encoding'
    model='mistral-caprica-gpt2-small-x81'
    chkpnts=[0,40,400,4000,40000,400000]
    precomputed_model = 'gpt2'
    files_ckpnt=[]
    permuted=''
    for ckpnt in chkpnts:
        if ckpnt==0:
            ckpnt=str(ckpnt)+'-untrained'
            #ckpnt = str(ckpnt)
        else:
            ckpnt = str(ckpnt) + permuted
        file_c = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model}-ckpnt-{ckpnt},subsample=None.pkl'))
        print(file_c)
        if len(file_c)>0:
            files_ckpnt.append(file_c[0])

    file_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model={model}-ckpnt-{0},*.pkl'))
    untrained_hf = pd.read_pickle(file_untrained[0])['data'].values

    chkpoints_srt=['untrained (n=0)','0.01% (n=40)','0.1% (n=400)' ,'1% (n=4K)','10% (n=40K)','100% (n=400K)']
    precomputed = pd.read_csv('/om/user/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench = precomputed[precomputed['benchmark'] == benchmark]
    model_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model]
    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model + '-untrained']

    # order files
    scores_mean=[]
    scors_std=[]
    score_data=[]
    for ix, file in tqdm(enumerate(files_ckpnt)):
        x=pd.read_pickle(file)['data']
        scores_mean.append(x.values[:,0])
        scors_std.append(x.values[:,1])
        score_data.append(x)

    # read precomputed scores

    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('inferno')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for idx,scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx,:],linewidth=2,marker='.',markersize=10,label=f'ck:{chkpoints_srt[idx]}')
        ax.fill_between(r3, scr-scors_std[idx],scr+scors_std[idx], facecolor=all_col[idx, :],alpha=0.1)
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

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_through_training_{model}{permuted}_{benchmark}_hf_init.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_through_training_{model}{permuted}_{benchmark}_hf_init.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
#%%
    # %%
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .2, .9, .35))
    scores_mean_np = np.stack(scores_mean).transpose()
    scores_std_np = np.stack(scors_std).transpose()
    for idx, scr in enumerate(scores_mean_np):
        r3 = np.arange(len(scr))
        r3 = .95 * r3 / len(r3)
        r3 = r3 - np.mean(r3)
        std_v = scores_std_np[idx, :]
        for idy, sc in enumerate(scr):
            ax.plot(r3[idy]+idx, sc, color=all_col[idy, :], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,  zorder=5)
            ax.errorbar(r3[idy]+idx, sc, yerr=std_v[idy], color='k', zorder=3)
        ax.plot(r3 + idx, scr, color=(.5, .5, .5), linewidth=1, zorder=4)

    ax.axhline(y=0, color='k', linestyle='-',zorder=2)
    #ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(l_names) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_mean_np)))
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylim(ylims)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} - layerwise')
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_through_training_{model}{permuted}_{benchmark}_layerwise.png'), dpi=250,
                format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_through_training_{model}{permuted}_{benchmark}_layerwise.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    #%%
    # plot for best layer of Shrimpf study
    layer_id = np.argmax(model_bench['score'])
    layer_name=model_bench['layer'].iloc[layer_id]
    scr_layer = [x[layer_id] for x in scores_mean]
    scr_layer_std = [x[layer_id] for x in scors_std]


    if benchmark=='Blank2014fROI-encoding':
        voxel_scores=[[x for idx, x in y.raw.raw.groupby('layer') if idx ==layer_name] for y in score_data]
        for idx, x in enumerate(voxel_scores):
            [h,pval]=ttest_ind(x[0].groupby('subject_UID').median(),voxel_scores[-1][0].groupby('subject_UID').median(),nan_policy='omit',axis=1)
            print(f'{idx}, {h}, {pval} \n')
    else:
        voxel_scores=[[x for idx, x in y.raw.raw.groupby('layer') if idx ==layer_name] for y in score_data]
        for idx, x in enumerate(voxel_scores):
            [h,pval]=ttest_ind(x[0].groupby('subject').median(),voxel_scores[-1][0].groupby('subject').median(),nan_policy='omit',axis=1,alternative='less')
            print(f'{idx}, {h}, {pval} \n')
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .45, .35))

    x_coords = [0.001, 0.01, 0.1, 1, 10,100]
    for idx, scr in enumerate(scr_layer):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='o', markersize=10,markeredgecolor='k',markeredgewidth=1,
                label=f'{chkpoints_srt[idx]}', zorder=2)
        ax.errorbar(x_coords[idx], scr, yerr=scr_layer_std[idx], color='k', zorder=1)
    # add precomputed
    # ax.errorbar(idx+.5,model_bench['score'],yerr=model_bench['error'],linestyle='--',fmt='.',markersize=20,linewidth=2,color=(0,0,0,1),label='trained(Schrimpf)',zorder=1)
    # ax.errorbar(-0.5, model_unt_bench['score'], yerr=model_unt_bench['error'], linestyle='--', fmt='.', markersize=20,
    #            linewidth=2, color=(.5, .5, .5, 1), label='untrained(Schrimpf)', zorder=1)
    ax.set_xscale('log')
    ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')

    # ax.set_xlim((-1,len(scores_mean)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate([ np.arange(1, 11) * 1e-3, np.arange(1, 11) * 1e-2,np.arange(1, 11) * 1e-1,np.arange(1, 11) * 1e0,np.arange(1, 11) * 1e1])

    ax.plot(8e2, np.asarray(model_bench['score'])[layer_id], color=(.3, .3, .3, 1), linewidth=2, marker='o',
            markersize=10,
            label=f'Schrimpf(2021)', zorder=2)
    ax.errorbar(8e2, np.asarray(model_bench['score'])[layer_id], yerr=np.asarray(model_bench['error'])[layer_id],
                color='k', zorder=1)

    ax.plot(0.0008, untrained_hf[layer_id][0], color=all_col[0, :], linewidth=2, marker='o', markeredgecolor='w',
            markersize=10, label=f'HF_untrained', zorder=2)
    ax.errorbar(0.0008, untrained_hf[layer_id][0], yerr=untrained_hf[layer_id][1], color='k', zorder=1)

    ax.set_xticks(np.concatenate([major_ticks, [8e2]]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)

    chkpoints_label = ['untrained', '0.01%', '0.1%', '1%', '10%',
                     '100%','Schrimpf\n(2021)']
    ax.set_xticklabels(chkpoints_label, rotation=0)
    ax.set_ylim(ylims)

    # ax.set_xticks((-.5,0,1,2,3,3.5))
    # ax.set_xticks((-.5, 0, 1, 2, 3, 3.5))
    # ax.set_xticklabels(['untrained(Shrimpf)','untrained','10M','100M','1B','trained(Schrimpf)'],rotation=90)
    # ax.set_xticks(np.asarray(x_coords))
    ax.legend(bbox_to_anchor=(1.6, .8), frameon=True, fontsize=8)
    # ax.set_xlim((min(x_coords),max(x_coords)))
    # ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_for_schrimpf_layer_through_training_{model}{permuted}_{benchmark}_hf_init.png'), dpi=250,
                format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_for_schrimpf_layer_through_training_{model}{permuted}_{benchmark}_hf_init.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

#%%
    score_change=np.diff(scores_mean_np)
    score_change=np.cumsum(score_change,axis=1)
    # %%
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .85, .35))

    for idx, scr in enumerate(score_change):
        r3 = np.arange(len(scr))
        r3 = .8 * r3 / len(r3)
        r3 = r3 - np.mean(r3)
        for idy, sc in enumerate(scr):
            ax.plot(r3[idy] + idx, sc, color=all_col[idy, :], linewidth=2, marker='o', markersize=8,
                    markeredgecolor='k', zorder=5)
        ax.plot(r3 + idx, scr, color=(.5, .5, .5), linewidth=1, zorder=4)

    ax.axhline(y=0, color='k', linestyle='-', zorder=1)
    ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(l_names) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_mean_np)))
    ax.set_ylim(ylims)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} - layerwise')
    fig.show()