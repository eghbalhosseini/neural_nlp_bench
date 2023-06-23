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
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# import gridspec
import matplotlib.gridspec as gridspec



if user=='eghbalhosseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

def compute_error(score):
    subject_values = np.nan_to_num(score.values,
                                   nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
    subject_axis = score.dims.index(score['subject'].dims[0])
    error = median_abs_deviation(subject_values, axis=subject_axis)
    return error

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
    #loss_10M_ckpnt = '2000'
    file_1B_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{2500}-untrained*.pkl'))
    file_1B=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{loss_1B_ckpnt}{permuted}*.pkl'))
    file_100M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                f'benchmark={benchmark},model={model_100M}-v2-ckpnt-{loss_100M_ckpnt}{permuted}*.pkl'))
    file_10M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                  f'benchmark={benchmark},model={model_10M}-v2-ckpnt-{loss_10M_ckpnt}{permuted}*.pkl'))
    file_1M = glob(os.path.join(result_caching, 'neural_nlp.score',
                                 f'benchmark={benchmark},model={model_1M}-v2-ckpnt-{loss_1M_ckpnt}{permuted}*.pkl'))
    files_srt = [file_1B_untrained[0], file_1M[0], file_10M[0], file_100M[0], file_1B[0]]
    chkpoints_srt = ['untrained', '1M', '10M', '100M', '1B']
    # order files

    file_untrained = glob(os.path.join(result_caching, 'neural_nlp.score',
                                          f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{310000}-untrained,*.pkl'))

    file_untrained_hf = glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model={model_1B}-v2-ckpnt-{310000}-untrained_hf,*.pkl'))

    #shcrimpf= glob(os.path.join(result_caching, 'neural_nlp.score',
    #                                   f'benchmark={benchmark},model=gpt2,*.pkl'))
    #schirmpf_data=pd.read_pickle(shcrimpf[0])['data']
    hf_untrained_data = pd.read_pickle(file_untrained_hf[0])['data']

    scores_mean=[]
    scors_std=[]
    score_data=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data']
        score_data.append(x)
        num_subj=x.raw.raw
        scores_mean.append(x.values[:,0])
        scors_std.append(x.values[:,1])

    scores_mean_per_roi=[]
    scores_std_per_roi=[]

    for x in score_data:
        roi_mean=[]
        roi_std=[]
        roi_names = []
        for roi_id, roi_x in x.raw.raw.groupby('roi'):
            score_=roi_x.groupby('subject').median().median('subject')
            score_err=compute_error(roi_x.groupby('subject').median())
            roi_mean.append(score_.values)
            roi_std.append(score_err)
            roi_names.append(roi_id)
        scores_mean_per_roi.append(roi_mean)
        scores_std_per_roi.append(roi_std)

    scores_mean_per_roi=np.stack([np.stack(x) for x in scores_mean_per_roi],axis=2)
    scores_std_per_roi=np.stack([np.stack(x) for x in scores_std_per_roi],axis=2)

    # score per atlas
    scores_mean_per_atlas = []
    scores_std_per_atlas = []

    for x in score_data:
        atlas_mean = []
        atlas_std = []
        atlas_names = []
        for atlas_id, atlas_x in x.raw.raw.raw.groupby('atlas'):
            True
            score_ = atlas_x.mean('split').groupby('subject').median().mean('experiment').median('subject')
            score_err = compute_error(atlas_x.mean('split').groupby('subject').median().mean('experiment'))
            atlas_mean.append(score_.values)
            atlas_std.append(score_err)
            atlas_names.append(atlas_id)
        scores_mean_per_atlas.append(atlas_mean)
        scores_std_per_atlas.append(atlas_std)

    scores_mean_per_atlas = np.stack([np.stack(x) for x in scores_mean_per_atlas], axis=2)
    scores_std_per_atlas = np.stack([np.stack(x) for x in scores_std_per_atlas], axis=2)

    # read precomputed scores
    precomputed=pd.read_csv('/om/weka/evlab/ehoseini/neural-nlp-2022/precomputed-scores.csv')
    precomputed_bench=precomputed[precomputed['benchmark']==benchmark]
    model_bench=precomputed_bench[precomputed_bench['model']==precomputed_model]

    model_unt_bench = precomputed_bench[precomputed_bench['model'] == precomputed_model+'-untrained']
    # untrained scores
    untrained_hf=pd.read_pickle(file_untrained_hf[0])['data'].values
    untrained_manual = pd.read_pickle(file_untrained[0])['data'].values

    l_names=pd.read_pickle(file)['data'].layer.values
    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(files_srt)), len(files_srt)))
    #%%
    # plot for best layer of Schirmpf study
    layer_id=np.argmax(model_bench['score'])
    scr_layer_roi=scores_mean_per_roi[:,layer_id,:]
    scr_layer_roi_std=scores_std_per_roi[:,layer_id,:]
    scr_layer_atlas=scores_mean_per_atlas[:,layer_id,:]
    scr_layer_atlas_std=scores_std_per_atlas[:,layer_id,:]
    layer_name=model_bench['layer'].iloc[layer_id]

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # create 12 subplots 4 by 3, for each roi
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=.5, hspace=.5)
    for idx, roi in enumerate(roi_names):
        ax=fig.add_subplot(gs[idx])
        scr_layer=scr_layer_roi[idx,:]
        scr_layer_std=scr_layer_roi_std[idx,:]
        x_coords=[1e5,1e6,10e6,100e6,1000e6]
        for id, scr in enumerate(scr_layer):
            ax.plot(x_coords[id], scr, color=all_col[id, :], linewidth=2, marker='o', markersize=8,markeredgecolor='k',markeredgewidth=1,
                label=f'{chkpoints_srt[id]}', zorder=2)
            ax.errorbar(x_coords[id], scr, yerr=scr_layer_std[id], color='k', zorder=1)
        ax.set_xscale('log')
        ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
        ax.axhline(y=0, color='k', linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        major_ticks = x_coords
        minor_ticks = np.concatenate([np.arange(1,11)*1e5,np.arange(1,11)*1e6,np.arange(1,11)*1e7,np.arange(1,11)*1e8])

        ax.set_xticks(np.concatenate([major_ticks]))
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(True, which="major", ls="-", color='0.9', zorder=0)
        ax.set_axisbelow(True)
        if idx==11:
            ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B'], rotation=0)
            ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
            ax.set_ylabel('Pearson Corr')
        ax.set_title(f'{roi}')
    fig.show()

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_per_roi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_per_roi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%%
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # create 12 subplots 4 by 3, for each roi
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=.5, hspace=.5)
    for idx, atlas in enumerate(atlas_names):
        ax = fig.add_subplot(gs[idx])
        scr_layer = scr_layer_atlas[idx, :]
        scr_layer_std = scr_layer_atlas_std[idx, :]
        x_coords = [1e5, 1e6, 10e6, 100e6, 1000e6]
        for id, scr in enumerate(scr_layer):
            ax.plot(x_coords[id], scr, color=all_col[id, :], linewidth=2, marker='o', markersize=8, markeredgecolor='k',
                    markeredgewidth=1,
                    label=f'{chkpoints_srt[id]}', zorder=2)
            ax.errorbar(x_coords[id], scr, yerr=scr_layer_std[id], color='k', zorder=1)
        ax.set_xscale('log')
        ax.plot(x_coords, scr_layer, color='k', linewidth=2, zorder=1)
        ax.axhline(y=0, color='k', linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        major_ticks = x_coords
        minor_ticks = np.concatenate(
            [np.arange(1, 11) * 1e5, np.arange(1, 11) * 1e6, np.arange(1, 11) * 1e7, np.arange(1, 11) * 1e8])

        ax.set_xticks(np.concatenate([major_ticks]))
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(True, which="major", ls="-", color='0.9', zorder=0)
        ax.set_axisbelow(True)
        if idx == 11:
            ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B'], rotation=0)
            ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
            ax.set_ylabel('Pearson Corr')
        ax.set_title(f'{atlas}')
    fig.show()
