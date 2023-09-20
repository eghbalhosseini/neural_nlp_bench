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
    result_dir=('/om/weka/evlab/ehoseini/MyData/NeuroBi'
                ''
                'oLang_2022/')
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

    shcrimpf= glob(os.path.join(result_caching, 'neural_nlp.score',
                                       f'benchmark={benchmark},model=gpt2,*.pkl'))
    schirmpf_data=pd.read_pickle(shcrimpf[0])['data']
    hf_untrained_data = pd.read_pickle(file_untrained_hf[0])['data']

    # compute score per roi in schrimpf_data
    schirmpf_data_roi_mean=[]
    schirmpf_data_roi_std=[]
    roi_id_shrimpf=[]
    for roi_id, roi_x in schirmpf_data.raw.raw.groupby('roi'):
        score_=roi_x.groupby('subject').median().median('subject')
        score_err=compute_error(roi_x.groupby('subject').median())
        schirmpf_data_roi_mean.append(score_.values)
        schirmpf_data_roi_std.append(score_err)
        roi_id_shrimpf.append(roi_id)

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
    score_roi=[]
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
        score_roi.append(x)
    # score per hemisphere
    # compute for schrimpf_data
    schirmpf_data_hemi_mean=[]
    schirmpf_data_hemi_std=[]
    score_=schirmpf_data.raw.raw
    hemi_coord=[x.split('_')[0] for x in score_.roi.values]
    score_=score_.assign_coords({'hemi': ('neuroid', hemi_coord)})
    for hemi_id, hemi_x in score_.groupby('hemi'):
        score_=hemi_x.groupby('subject').median().median('subject')
        score_err=compute_error(hemi_x.groupby('subject').median())
        schirmpf_data_hemi_mean.append(score_.values)
        schirmpf_data_hemi_std.append(score_err)
    schirmpf_data_hemi_mean=np.array(schirmpf_data_hemi_mean)
    schirmpf_data_hemi_std=np.array(schirmpf_data_hemi_std)
    schirmpf_data_hemi_mean=np.expand_dims(schirmpf_data_hemi_mean,-1)
    schirmpf_data_hemi_std=np.expand_dims(schirmpf_data_hemi_std,-1)

    scores_per_hemi=[]
    scores_std_per_hemi=[]
    score_hemi=[]
    for x in score_data:
        hemi_mean=[]
        hemi_std=[]
        hemi_names=[]
        score_=x.raw.raw
        hemi_coord=[x.split('_')[0] for x in score_.roi.values]
        score_x=score_.assign_coords({'hemi': ('neuroid', hemi_coord)})
        for hemi_id, hemi_x in score_x.groupby('hemi'):
            score_=hemi_x.groupby('subject').median().median('subject')
            score_err=compute_error(hemi_x.groupby('subject').median())
            hemi_mean.append(score_.values)
            hemi_std.append(score_err)
            hemi_names.append(hemi_id)
        scores_per_hemi.append(hemi_mean)
        scores_std_per_hemi.append(hemi_std)
        score_hemi.append(score_x)

    scores_mean_per_hemi = np.stack([np.stack(x) for x in scores_per_hemi], axis=2)
    scores_std_per_hemi = np.stack([np.stack(x) for x in scores_std_per_hemi], axis=2)

    #scores_mean_per_roi=np.stack([np.stack(x) for x in scores_mean_per_roi],axis=2)
    #scores_std_per_roi=np.stack([np.stack(x) for x in scores_std_per_roi],axis=2)

    # score per atlas
    # compute for schrimpf_data
    schirmpf_data_mean=[]
    schirmpf_data_std=[]
    for atlas_id, atlas_x in schirmpf_data.raw.raw.raw.groupby('atlas'):
        score_=atlas_x.mean('split').groupby('subject').median().mean('experiment').median('subject')
        score_err=compute_error(atlas_x.mean('split').groupby('subject').median().mean('experiment'))
        schirmpf_data_mean.append(score_.values)
        schirmpf_data_std.append(score_err)
    schirmpf_atlas_mean=np.array(schirmpf_data_mean)
    schirmpf_atlas_std=np.array(schirmpf_data_std)
    schirmpf_data_atlas_mean=np.expand_dims(schirmpf_atlas_mean,-1)
    schirmpf_data_atlas_std=np.expand_dims(schirmpf_atlas_std,-1)


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
    #
    roi_order= ['LH_IFGorb','LH_IFG','LH_MFG',
                'LH_AntTemp','LH_PostTemp','LH_AngG',
                'RH_IFGorb','RH_IFG','RH_MFG',
                'RH_AntTemp','RH_PostTemp','RH_AngG']
    # reoder the score roi date based on roi_order
    scores_mean_per_roi_reordered=[]
    scores_std_per_roi_reordered=[]
    for roi in roi_order:
        ix=roi_names.index(roi)
        scores_mean_per_roi_reordered.append([x[ix] for x in scores_mean_per_roi])
        scores_std_per_roi_reordered.append([x[ix] for x in scores_std_per_roi])
    # replace the scores with reordered scores
    scores_mean_per_roi=scores_mean_per_roi_reordered
    scores_std_per_roi=scores_std_per_roi_reordered

    scores_mean_per_roi = np.stack([np.stack(x) for x in scores_mean_per_roi], axis=2)
    scores_std_per_roi = np.stack([np.stack(x) for x in scores_std_per_roi], axis=2)
    # reorder the dimensions to be [roi,layer, model]
    scores_mean_per_roi=np.transpose(scores_mean_per_roi,[2,1,0])
    scores_std_per_roi=np.transpose(scores_std_per_roi,[2,1,0])

    # reoder shrimp_roi_data
    schirmpf_scores_mean_per_roi_reordered=[]
    schirmpf_scores_std_per_roi_reordered=[]
    for roi in roi_order:
        ix=roi_id_shrimpf.index(roi)
        schirmpf_scores_mean_per_roi_reordered.append(schirmpf_data_roi_mean[ix])
        schirmpf_scores_std_per_roi_reordered.append(schirmpf_data_roi_std[ix])
    # replace the scores with reordered scores
    schirmpf_data_roi_mean=schirmpf_scores_mean_per_roi_reordered
    schirmpf_data_roi_std=schirmpf_scores_std_per_roi_reordered

    schirmpf_data_roi_mean = np.stack(schirmpf_data_roi_mean)
    schirmpf_data_roi_std = np.stack(schirmpf_data_roi_std)
    # add a new dimension so that it is roi_layer, model
    schirmpf_data_roi_mean=np.expand_dims(schirmpf_data_roi_mean,-1)
    schirmpf_data_roi_std=np.expand_dims(schirmpf_data_roi_std,-1)


    #%% read precomputed scores
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
    # plot for best layer of Schirmpf study
    layer_id=np.argmax(model_bench['score'])
    scr_layer_roi=scores_mean_per_roi[:,layer_id,:]
    scr_layer_roi_std=scores_std_per_roi[:,layer_id,:]
    scr_layer_atlas=scores_mean_per_atlas[:,layer_id,:]
    scr_layer_atlas_std=scores_std_per_atlas[:,layer_id,:]
    # hemi
    scr_layer_hemi=scores_mean_per_hemi[:,layer_id,:]
    scr_layer_hemi_std=scores_std_per_hemi[:,layer_id,:]
    # schirmpf
    scr_layer_schirmpf_roi=schirmpf_data_roi_mean[:,layer_id,:]
    scr_layer_schirmpf_std_roi=schirmpf_data_roi_std[:,layer_id,:]
    # atlas for schirmpf
    scr_layer_schirmpf_atlas=schirmpf_data_atlas_mean[:,layer_id,:]
    scr_layer_schirmpf_std_atlas=schirmpf_data_atlas_std[:,layer_id,:]

    # hemi
    scr_layer_schirmpf_hemi=schirmpf_data_hemi_mean[:,layer_id,:]
    scr_layer_schirmpf_std_hemi=schirmpf_data_hemi_std[:,layer_id,:]




    layer_name=model_bench['layer'].iloc[layer_id]

    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # create 12 subplots 4 by 3, for each roi
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=.5, hspace=.5)
    for idx, roi in enumerate(roi_order):
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

        ax.plot(8000e6, scr_layer_schirmpf_roi[idx], color=(.3, .3, .3, 1), linewidth=2, marker='o',
                markersize=10,
                label=f'Schrimpf(2021)', zorder=2)
        ax.errorbar(8000e6, scr_layer_schirmpf_roi[idx], yerr=scr_layer_schirmpf_std_roi[idx],
                    color='k', zorder=1)

        ax.set_xticks(np.concatenate([major_ticks, [8000e6]]))
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(True, which="major", ls="-", color='0.9', zorder=0)
        ax.set_axisbelow(True)
        if idx==11:
            ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B','Schrimpf\n(2021)'], rotation=0)
            ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
            ax.set_ylabel('Pearson Corr')
        ax.set_title(f'{roi}')
        # set ylim
        ax.set_ylim([-0.1, 0.35])
    fig.show()
    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_per_roi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.png'), dpi=250, format='png',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_per_roi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.eps'), format='eps',
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    #%% plot per hemi for best layer of Schirmpf study
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # create 12 subplots 4 by 3, for each roi
    ys_=[.6,.1]
    hemi_names=['left','right']
    for idx, atlas in enumerate(hemi_names):
        ax = plt.axes((.1, ys_[idx], .35, .35))
        scr_layer = scr_layer_hemi[idx, :]
        scr_layer_std = scr_layer_hemi_std[idx, :]
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
        minor_ticks = np.concatenate([np.arange(1,11)*1e5,np.arange(1,11)*1e6,np.arange(1,11)*1e7,np.arange(1,11)*1e8])
        # add schrimpf
        ax.plot(8000e6, scr_layer_schirmpf_hemi[idx], color=(.3, .3, .3, 1), linewidth=2, marker='o',
                markersize=10,
                label=f'Schrimpf(2021)', zorder=2)
        ax.errorbar(8000e6, scr_layer_schirmpf_hemi[idx], yerr=scr_layer_schirmpf_std_hemi[idx],
                    color='k', zorder=1)

        ax.set_xticks(np.concatenate([major_ticks, [8000e6]]))
        ax.set_xticks(minor_ticks, minor=True)
        plt.grid(True, which="major", ls="-", color='0.9', zorder=0)
        ax.set_axisbelow(True)
        if idx == 11:
            ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B'], rotation=0)
            ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
            ax.set_ylabel('Pearson Corr')
        ax.set_title(f'{atlas}')
        # set y lim to be the same for all
        ax.set_ylim(-.1, .3)
    fig.show()
    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_per_hemi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.png'),
        dpi=250, format='png',
        metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_per_hemi_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.eps'),
        format='eps',
        metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


    #%%
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    # create 12 subplots 4 by 3, for each roi
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=.5, hspace=.5)
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
        minor_ticks = np.concatenate([np.arange(1,11)*1e5,np.arange(1,11)*1e6,np.arange(1,11)*1e7,np.arange(1,11)*1e8])

        ax.set_xticks(np.concatenate([major_ticks]))
        ax.set_xticks(minor_ticks, minor=True)
        # add schrimpf for atlas
        ax.plot(8000e6, scr_layer_schirmpf_atlas[idx], color=(.3, .3, .3, 1), linewidth=2, marker='o',
                markersize=10,
                label=f'Schrimpf(2021)', zorder=2)
        ax.errorbar(8000e6, scr_layer_schirmpf_atlas[idx], yerr=scr_layer_schirmpf_std_atlas[idx],
                    color='k', zorder=1)


        plt.grid(True, which="major", ls="-", color='0.9', zorder=0)
        ax.set_axisbelow(True)
        if idx == 11:
            ax.set_xticklabels(['untrained', '1M', '10M', '100M', '1B'], rotation=0)
            ax.legend(bbox_to_anchor=(1., .8), frameon=True, fontsize=8)
            ax.set_ylabel('Pearson Corr')
        ax.set_title(f'{atlas}')
        # set y lim to be the same for all
        ax.set_ylim(-.1, .3)
    fig.show()
    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_per_atlas_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.png'),
        dpi=250, format='png',
        metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(
        os.path.join(analysis_dir, f'chpnt_score_per_atlas_gpt_neox{permuted}_for_schrimpf_layer_{benchmark}.eps'),
        format='eps',
        metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%% do a ttest for posttem and IFGorb
    len(score_roi)
    voxel_scores = [[x for idx, x in y.raw.raw.groupby('layer') if idx == layer_name][0] for y in score_roi]
    # voxel for posttem
    roi_name='LH_PostTemp'
    posttemp_scores = [[x for idx, x in y.groupby('roi') if idx == roi_name][0] for y in voxel_scores]
    roi_name='LH_IFGorb'
    IFGorb_scores = [[x for idx, x in y.groupby('roi') if idx == roi_name][0] for y in voxel_scores]

    voxel_name = [y.attrs['model'] for y in score_roi]
    for idx, x in enumerate(posttemp_scores):
        [h, pval] = ttest_ind(x.groupby('subject').median().values.squeeze(),
                              IFGorb_scores[idx].groupby('subject').median().values.squeeze(),
                              nan_policy='omit')
        print(f'{voxel_name[idx]},{idx}, {h}, {pval} \n')

    [h, pval] = ttest_ind(voxel_scores[-1][0].groupby('subject').median().values.squeeze(),
                      voxel_scores[-2][0].groupby('subject').median().values.squeeze(),
                      nan_policy='omit')
    print(f'{idx}, {h}, {pval} \n')
    #%%
    len(score_roi)
    voxel_scores = [[x for idx, x in y.raw.groupby('layer') if idx == layer_name][0] for y in score_hemi]
    # add hemi as a coordiante to each voxel_score
    voxel_scores_mode=[]
    for idx, x in enumerate(voxel_scores):
        # get only languge atlass
        x=x.sel(atlas='language')
        voxel_scores_mode.append(x.assign_coords({'hemi':('neuroid',score_hemi[idx].hemi.values)}))
    # voxel for posttem
    hemi_name = 'LH'
    posttemp_scores = [[x for idx, x in y.groupby('hemi') if idx == hemi_name][0] for y in voxel_scores_mode]
    roi_name = 'RH'
    IFGorb_scores = [[x for idx, x in y.groupby('hemi') if idx == roi_name][0] for y in voxel_scores_mode]

    voxel_name = [y.attrs['model'] for y in score_roi]
    for idx, x in enumerate(posttemp_scores):
        [h, pval] = ttest_ind(x.groupby('subject').mean('split').median('experiment').values.squeeze(),
                              IFGorb_scores[idx].groupby('subject').mean('split').median('experiment').values.squeeze(),
                              nan_policy='omit')
        print(f'{voxel_name[idx]},{idx}, {h}, {pval} \n')

    [h, pval] = ttest_ind(voxel_scores[-1][0].groupby('subject').median().values.squeeze(),
                          voxel_scores[-2][0].groupby('subject').median().values.squeeze(),
                          nan_policy='omit')
    print(f'{idx}, {h}, {pval} \n')
