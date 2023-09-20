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
plt.rcdefaults()
from scipy.stats import ttest_ind_from_stats, ttest_ind, ttest_1samp
## Set up LaTeX fonts
import matplotlib
from scipy.stats import ttest_ind_from_stats
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

# TODO : put the data for ttest on OSF

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
    #benchmark='Blank2014fROI-encoding'
    #ylims=(-.1,.5)
    #benchmark = 'Fedorenko2016v3-encoding'
    model_1B='gpt2-neox-pos_learned-1B'
    precomputed_model='gpt2'
    loss_1B_ckpnt='310000'
    model_100M = 'gpt2-neox-pos_learned-100M'
    #loss_100M_ckpnt='11600'
    loss_100M_ckpnt='14250'
    model_10M = 'gpt2-neox-pos_learned-10M'
    #loss_10M_ckpnt='2250'
    loss_10M_ckpnt = '2000'

    model_1M = 'gpt2-neox-pos_learned-1M'
    loss_1M_ckpnt = '1000'
    #permuted='-permuted'
    permuted=''
    version='v2'
    #loss_10M_ckpnt = '2000'
    file_1B_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-{version}-ckpnt-{310000}-untrained*.pkl'))
    file_1B=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model_1B}-{version}-ckpnt-{loss_1B_ckpnt}{permuted},*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_100M}-{version}-ckpnt-{loss_100M_ckpnt}{permuted},*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_10M}-{version}-ckpnt-{loss_10M_ckpnt}{permuted},*.pkl'))
    file_1M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_1M}-{version}-ckpnt-{loss_1M_ckpnt}{permuted},*.pkl'))
    files_srt = [file_1B_untrained[0], file_1M[0], file_10M[0], file_100M[0], file_1B[0]]
    chkpoints_srt = ['untrained', '1M', '10M', '100M', '1B']
    # order files

    file_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-v3-ckpnt-{310000}-untrained,*.pkl'))

    file_untrained_hf = glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model={model_1B}-v3-ckpnt-{310000}-untrained_hf,*.pkl'))

    shcrimpf= glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model=gpt2,*.pkl'))
    shcrimpf_unt = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model=gpt2-untrained_hf,*.pkl'))
    schirmpf_data_unt = pd.read_pickle(shcrimpf_unt[0])['data']
    schirmpf_data=pd.read_pickle(shcrimpf[0])['data']
    hf_untrained_data = pd.read_pickle(file_untrained_hf[0])['data']
    schrimpf_mean=schirmpf_data.values[:,0]
    schrimpf_std=schirmpf_data.values[:,1]
    scores_mean=[]
    scors_std=[]
    score_data=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data']
        score_data.append(x)
        num_subj=x.raw.raw
        scores_mean.append(x.values[:,0])
        scors_std.append(x.values[:,1])


    # read precomputed scores
    #precomputed=pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    #precomputed_bench=precomputed[precomputed['benchmark']==benchmark]
    #model_bench=precomputed_bench[precomputed_bench['model']==precomputed_model]

    #model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model+'-untrained']
    # untrained scores
    untrained_hf=pd.read_pickle(file_untrained_hf[0])['data'].values
    untrained_manual = pd.read_pickle(file_untrained[0])['data'].values

    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    for idx,scr in enumerate(scores_mean):
        r3 = np.arange(len(scr))
        ax.plot(r3, scr, color=all_col[idx,:],linewidth=2,marker='.',markersize=10,label=f'{chkpoints_srt[idx]}')
        ax.fill_between(r3, scr-scors_std[idx],scr+scors_std[idx], facecolor=all_col[idx, :],alpha=0.1)
    # add precomputed
    ax.plot(r3,schrimpf_mean,linestyle='-',linewidth=2,color=(.5,.5,.5,1),label='trained(Schrimpf)',zorder=1)
    ax.fill_between(r3, schrimpf_mean - schrimpf_std, schrimpf_mean + schrimpf_std, facecolor=(.5,.5,.5,1), alpha=0.1)
    # ax.plot(r3,model_bench['score'],linestyle='-',linewidth=2,color=(.5,.5,.5,1),label='trained(Schrimpf)',zorder=1)
    #ax.plot(r3, model_unt_bench['score'], linestyle='--',linewidth=2, color=(.5,.5,.5,1), label='untrained(Schrimpf)',zorder=1)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.3, .8), frameon=True,fontsize=8)
    ax.set_xlim((0-.5,len(l_names)-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scr)))
    ax.set_xlabel('Layer')
    ax.set_ylim(ylims)
    plt.grid(True, which="both", ls="-", color='0.9')
    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}_v3.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}_v3.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
    #%%
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    plt.rcParams.update({'font.size': 8})
    ax = plt.axes((.05, .2, .9, .35))
    scores_mean_np = np.stack(scores_mean).transpose()
    scores_std_np = np.stack(scors_std).transpose()
    for idx, scr in enumerate(scores_mean_np):
        r3 = np.arange(len(scr))
        std_v=scores_std_np[idx,:]
        r3 = .95 * r3 / len(r3)
        r3= r3-np.mean(r3)
        for idy ,sc in enumerate(scr):
            ax.plot(r3[idy]+idx, sc, color=all_col[idy, :], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,  zorder=5)
            ax.errorbar(r3[idy]+idx, sc, yerr=std_v[idy], color='k', zorder=3)
        ax.plot(r3 + idx, scr, color=(.5,.5,.5), linewidth=1,zorder=4)

    ax.axhline(y=0, color='k', linestyle='-',zorder=2)
    #ax.legend(bbox_to_anchor=(1.4, 2), frameon=True, fontsize=8)
    ax.set_xlim((0 - .5, len(l_names) - .5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(scores_mean_np)))
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    #ax.set_xticklabels(l_names, rotation=90, fontsize=12)

    #ax.set_xticklabels(['untrained', '10M', '100M', '1B','Schrimpf\n(2021)'], rotation=0)
    ax.set_ylim(ylims)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark} - layerwise')
    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}_layerwise.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}_layerwise.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
#%%
    # plot for best layer of Schirmpf study
    #layer_id=np.argmax(model_bench['score'])
    layer_id = np.argmax(schrimpf_mean)
    layer_name=schirmpf_data['layer'][layer_id]
    scr_layer=[x[layer_id] for x in scores_mean]
    scr_layer_std = [x[layer_id] for x in scors_std]
    scr_schirmpf= schirmpf_data.sel(layer=(schirmpf_data.layer==layer_name).values)


    scr_hf=hf_untrained_data.sel(layer=(hf_untrained_data.layer==layer_name).values)

    hf_untrained_score = [x for idx, x in scr_hf.raw.raw.groupby('layer') if idx == layer_name]
    ttest_1samp(hf_untrained_score[0].groupby('subject').median().values,popmean=0)
    #n_sub=len(np.unique(score_data[0].raw.raw.subject))
    ttest_1samp(scr_hf,popmean=0)
    #for idx, x in enumerate(scr_layer):
    #    [h,pval]=ttest_ind_from_stats(x,scr_layer_std[idx],n_sub,scr_layer[-1],scr_layer_std[-1],n_sub)
    #    print(f'{idx}, {h}, {pval} \n')
    if benchmark=='Blank2014fROI-encoding':
        voxel_scores=[[x for idx, x in y.raw.raw.groupby('layer') if idx ==layer_name] for y in score_data]
        schrimpf_score=[x for idx, x in scr_schirmpf.raw.raw.groupby('layer') if idx ==layer_name]
        for idx, x in enumerate(voxel_scores):
            [h,pval]=ttest_ind(x[0].groupby('subject_UID').median(),voxel_scores[-1][0].groupby('subject_UID').median(),nan_policy='omit',axis=1)
            print(f'{idx}, {h}, {pval} \n')
    else:
        voxel_scores=[[x for idx, x in y.raw.raw.groupby('layer') if idx ==layer_name] for y in score_data]
        score_name=[x.attrs['model'] for x in score_data]
        schrimpf_score = [x for idx, x in scr_schirmpf.raw.raw.groupby('layer') if idx == layer_name]
        for idx, x in enumerate(voxel_scores):
            #[h,pval]=ttest_ind(x[0].groupby('subject').median(),voxel_scores[-1][0].groupby('subject').median(),nan_policy='omit',axis=1,alternative='less')
            [h, pval] = ttest_ind(x[0].groupby('subject').median().values.squeeze(), schrimpf_score[0].groupby('subject').median().values,
                                   nan_policy='omit', axis=0, alternative='less')
            print(f'{score_name[idx]},vs 1B {idx}, {h}, {pval*4} \n')

    ttest_1samp(voxel_scores[0][0].groupby('subject').median().values.squeeze(),popmean=0)
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .45, .35))
    x_coords=[1e5,1e6,10e6,100e6,1000e6]
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
    minor_ticks = np.concatenate([np.arange(1,11)*1e5,np.arange(1,11)*1e6,np.arange(1,11)*1e7,np.arange(1,11)*1e8])

    #ax.plot(8000e6, np.asarray(model_bench['score'])[layer_id], color=(.3,.3,.3,1), linewidth=2, marker='o', markersize=10,
    #        label=f'Schrimpf(2021)', zorder=2)
    #ax.errorbar(8000e6, np.asarray(model_bench['score'])[layer_id], yerr=np.asarray(model_bench['error'])[layer_id], color='k', zorder=1)

    ax.plot(8000e6, scr_schirmpf.values[0][0], color=(.3,.3,.3,1), linewidth=2, marker='o', markersize=10,
           label=f'Schrimpf(2021)', zorder=2)
    ax.errorbar(8000e6, scr_schirmpf.values[0][0], yerr=scr_schirmpf.values[0][1], color='k', zorder=1)

    ax.plot(.8e5, untrained_hf[layer_id][0], color=all_col[0, :], linewidth=2, marker='o',markeredgecolor='w',markersize=10,label=f'HF_untrained', zorder=2)
    ax.errorbar(.8e5, untrained_hf[layer_id][0], yerr=untrained_hf[layer_id][1],color='k', zorder=1)

    ax.set_xticks(np.concatenate([major_ticks,[8000e6]]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B', 'Schrimpf\n(2021)'], rotation=0)
    #ax.set_xticklabels(['untrained', '10M', '100M', '1B','Schrimpf\n(2021)'], rotation=0)
    ax.set_ylim(ylims)

    # ax.set_xticks((-.5,0,1,2,3,3.5))
    # ax.set_xticks((-.5, 0, 1, 2, 3, 3.5))
    # ax.set_xticklabels(['untrained(Shrimpf)','untrained','10M','100M','1B','trained(Schrimpf)'],rotation=90)
    # ax.set_xticks(np.asarray(x_coords))
    ax.legend(bbox_to_anchor=(1.7, .8), frameon=True, fontsize=8)
    # ax.set_xlim((min(x_coords),max(x_coords)))
    # ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_for_layer_{benchmark}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_for_layer_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

#%% plot for best layer of current study
    layer_ids = [np.argmax(x) for x in scores_mean]

    scr_layer = [x[layer_ids[idx]] for idx, x in enumerate(scores_mean)]
    scr_layer_std = [x[layer_ids[idx]] for idx, x in enumerate(scors_std)]

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # ax = plt.axes((.1, .4, .45, .35))
    ax = plt.axes((.1, .4, .35, .35))
    x_coords = [1e5, 1e6, 10e6, 100e6, 1000e6]
    for idx, scr in enumerate(scr_layer):
        ax.plot(x_coords[idx], scr, color=all_col[idx, :], linewidth=2, marker='.', markersize=20,
                label=f'{chkpoints_srt[idx]}', zorder=5)
        ax.errorbar(x_coords[idx], scr, yerr=scr_layer_std[idx], color='k', zorder=4)
    # add precomputed
    # ax.errorbar(idx+.5,model_bench['score'],yerr=model_bench['error'],linestyle='--',fmt='.',markersize=20,linewidth=2,color=(0,0,0,1),label='trained(Schrimpf)',zorder=1)
    # ax.errorbar(-0.5, model_unt_bench['score'], yerr=model_unt_bench['error'], linestyle='--', fmt='.', markersize=20,
    #            linewidth=2, color=(.5, .5, .5, 1), label='untrained(Schrimpf)', zorder=1)
    ax.set_xscale('log')
    ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
    ax.axhline(y=0, color='k', linestyle='-',)

    # ax.set_xlim((-1,len(scores_mean)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    major_ticks = x_coords
    minor_ticks = np.concatenate(
        [np.arange(1, 11) * 1e5, np.arange(1, 11) * 1e6, np.arange(1, 11) * 1e7, np.arange(1, 11) * 1e8])

    ax.plot(8000e6, np.asarray(model_bench['score'])[layer_id], color=(.3, .3, .3, 1), linewidth=2, marker='.',
            markersize=20,
            label=f'Schrimpf(2021)', zorder=2)
    ax.errorbar(8000e6, np.asarray(model_bench['score'])[layer_id], yerr=np.asarray(model_bench['error'])[layer_id],
                color='k', zorder=1)

    ax.set_xticks(np.concatenate([major_ticks, [8000e6]]))
    ax.set_xticks(minor_ticks, minor=True)
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B', 'Schrimpf\n(2021)'], rotation=0)
    # ax.set_xticklabels(['untrained', '10M', '100M', '1B','Schrimpf\n(2021)'], rotation=0)
    ax.set_ylim(ylims)

    # ax.set_xticks((-.5,0,1,2,3,3.5))
    # ax.set_xticks((-.5, 0, 1, 2, 3, 3.5))
    # ax.set_xticklabels(['untrained(Shrimpf)','untrained','10M','100M','1B','trained(Schrimpf)'],rotation=90)
    # ax.set_xticks(np.asarray(x_coords))
    ax.legend(bbox_to_anchor=(1.7, .8), frameon=True, fontsize=8)
    # ax.set_xlim((min(x_coords),max(x_coords)))
    # ax.set_xticklabels(l_names, rotation=90, fontsize=12)
    ax.set_ylabel('Pearson Corr')
    ax.set_title(f'benchmark {benchmark}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_best_loss_gpt_neox{permuted}_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


