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
from pathlib import Path
import seaborn as sns
import matplotlib.gridspec as gridspec
ROOTDIR = (Path('/om/weka/evlab/ehoseini/MyData/fmri_DNN/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'outputs')).resolve()
PLOTDIR = (Path(OUTDIR / 'plots')).resolve()
from glob import glob
from scipy import stats
from scipy.stats import median_abs_deviation
if user=='eghbalhosseini':
    #analysis_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    True
elif user=='ehoseini':
    analysis_dir='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/analysis/'
    #result_dir='/om/weka/evlab/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmarks=['DsParametricfMRI_v1-min-RidgeEncoding', 'DsParametricfMRI_v1-rand-RidgeEncoding','DsParametricfMRI_v1-max-RidgeEncoding']
    bench_key='DsParametricfMRI_v1_RidgeEncoding'
    #benchmark = 'LangLocECoGv2-encoding'

    colors = [np.divide((51, 153, 255), 255), np.divide((160, 160, 160), 256), np.divide((255, 153, 51), 255),
              np.divide((55, 76, 128), 256)]

    '''ANN result across models'''
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    layer_ids = (2, 24, 12, 12, 44, 5, 47)
    model_sh = ['RoBERTa', 'XLNet-L', 'BERT-L', 'XLM', 'GPT2-XL', 'ALBERT-XXL', 'CTRL']
    models_scores=[]
    score_dat=[]
    for model,layer in model_layers:
        benchmark_score=[]
        benchmark_dat=[]
        for benchmark in benchmarks:
            files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
            assert len(files)>0
            scores_mean=[]
            scors_std=[]
            x=pd.read_pickle(files[0])['data']
            scores_mean=x.values[:,0]
            scores_std=x.values[:,1]
            l_names=x.layer.values
            ann_layer=int(np.argwhere(l_names ==layer))
            benchmark_score.append([scores_mean[ann_layer],scores_std[ann_layer]])
            x_raw=x.raw.raw
            # select the layer
            selected_layer=int(np.argwhere((x_raw.layer==layer).values))
            # use xarray to select the layer data
            x_raw=x_raw.isel(layer=selected_layer)
            benchmark_dat.append(x_raw)
        models_scores.append(benchmark_score)
        score_dat.append(benchmark_dat)

    models_scores=np.stack(models_scores)
    width = 0.25  # the width of the bars
    x = np.arange(models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in model_layers]



    #%% investigate individual layers
    x_=xr.concat(score_dat[4],dim='benchmark').squeeze().mean('split')
    # use sns to plot one distribution for each row of x_
    fig = plt.figure(figsize=(11, 8))

    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(7, 8, figure=fig, wspace=.1, hspace=.1)
    for i in range(7):
        x_ = xr.concat(score_dat[i], dim='benchmark').squeeze().mean('split')
        for j, (id,grp) in enumerate(x_.groupby('subject')):
            ax = fig.add_subplot(gs[i, j])
            # create a dataframe with 1 coloum for each row in grp, and 1 colurm that is the benchmark
            df=pd.DataFrame(grp.values.T,columns=grp.benchmark.values)
            sns.ecdfplot(df[0],ax=ax, color=custom_palette_rgb[0])
            sns.ecdfplot(df[1], ax=ax, color=custom_palette_rgb[1])
            sns.ecdfplot(df[2], ax=ax, color=custom_palette_rgb[2])
        # j is zero add a y label, which is the model,
            if j==0:
                ax.set_ylabel(model_sh[i], fontsize=6)
            else:
                ax.set_ylabel('')

        # if i is zero then add a title which is the id
            if i==0:
                ax.set_title(id, fontsize=6)
        # turn off the x axis
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    fig.show()
    #%%
    fig = plt.figure(figsize=(11, 8))
    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(7, 8, figure=fig, wspace=.1, hspace=.1)
    for i in range(7):
        x_ = xr.concat(score_dat[i], dim='benchmark').squeeze().mean('split')
        # drop voxels with repettion_corr_ratio of <0.95
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        for j, (id,grp) in enumerate(x_.groupby('subject')):
            ax = fig.add_subplot(gs[i, j])
            # make axes to be square
            # do a scatter of row 0 vs row 2
            sns.scatterplot(x=grp[0,:].values,y=grp[2,:].values,ax=ax,s=2)
            # get the limits of axes from sns plot
            x0,x1=ax.get_xlim()
            y0,y1=ax.get_ylim()
            # turn off the legend
            # ax.get_legend().remove()
            # set y lim to be equal to xlim
            xlim=min(x0,y0)
            ylim=max(x1,y1)
            ax.set_xlim((xlim,ylim))
            ax.set_ylim((xlim,ylim))
            # plot a unity line
            ax.plot([xlim,ylim],[xlim,ylim],color='k',linewidth=.5)
            ax.set_aspect('equal', 'box')
            # turn off axis ticks and labels
            if j == 0:
                ax.set_ylabel(model_sh[i], fontsize=6)
            else:
                ax.set_ylabel('')

            # if i is zero then add a title which is the id
            if i == 0:
                ax.set_title(id, fontsize=6)
            # turn off the x axis
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            # limit
            # for kk in range(grp.shape[0]):
            #     sns.displot(grp[kk,:].values, hist=False, rug=True, label=id, color=custom_palette_rgb[kk])
        # plot a line for each subject in x_mean
        # turn off

    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'vox_max_min_regression_relation_{bench_key}_full.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'vox_max_min_regression_relation_{bench_key}_full.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%%

    fig = plt.figure(figsize=(11, 8))
    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=.1, hspace=.1)
    for i in range(7):
        x_ = xr.concat(score_dat[i], dim='benchmark').squeeze().mean('split')
        # select roi 1
        #x_ = x_.where(x_.roi == np.unique(x_.roi)[5], drop=True)
        # drop voxels with repettion_corr_ratio of <0.95
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        ax = fig.add_subplot(gs[i])
            # make axes to be square
            # do a scatter of row 0 vs row 2
        # do a displot
        sns.distplot(x_[0, :].values, rug=True, color=custom_palette_rgb[0],ax=ax)
        sns.distplot(x_[2, :].values, rug=True, color=custom_palette_rgb[2],ax=ax)
        # plot the mean of each row
        ax.axvline(x_[0, :].median().values, color=custom_palette_rgb[0], linestyle='-')
        ax.axvline(x_[2, :].median().values, color=custom_palette_rgb[2], linestyle='-')
    fig.show()

    fig = plt.figure(figsize=(11, 8))
    x_ = xr.concat(score_dat[1], dim='benchmark').squeeze()
    # plot the distirbution of reptetion corr for each subject
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=.5, hspace=.3)
    for i, (id,grp) in enumerate(x_.groupby('subject')):
        ax = fig.add_subplot(gs[i])
        sns.distplot(grp.repetition_corr.values, rug=True, ax=ax)
        # plot mean of each subject
        ax.axvline(grp.repetition_corr.mean().values, color='k', linestyle='-')
        ax.set_title(f'{id}, vox={np.sum(grp.repetition_corr_ratio>0).squeeze().values}')
        # remove y label
        ax.set_ylabel('')
        ax.set_xlabel('')
        # remove spines
        sns.despine(ax=ax)

    fig.show()

    #%%
    score_xr= [xr.concat(x, dim='benchmark').squeeze().mean('split') for x in score_dat]
    fig = plt.figure(figsize=(11, 8))
    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=.2, hspace=.2)
    for i in range(7):
        x_ = score_xr[i]
        # selet x_ to be only voxels with repettion_corr_ratio of >0.95
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        #x_ = x_.sel(neuroid=(x_.subject != '1047').values)
        # select roi 1
        ax = fig.add_subplot(gs[i])
            # make axes to be square
            # do a scatter of row 0 vs row 2
        # do a displot
        sns.distplot(x_[0, :].values, rug=True, color=custom_palette_rgb[0],ax=ax)
        sns.distplot(x_[2, :].values, rug=True, color=custom_palette_rgb[2],ax=ax)
        # test if the distribution x_[0, :] is larger than  x_[2, :]


        [t,p] = stats.ttest_ind(x_[0, :].values, x_[2, :].values)
        # if the difference is significant then plot a star
        if p < .05:
            ax.text(.5,.5,'*', fontsize=20)
            # also show the p value
            ax.text(.5,.4,f'p={p:.3f}', fontsize=10)
        # plot the mean of each row
        ax.axvline(x_[0, :].mean().values, color=custom_palette_rgb[0], linestyle='-')
        ax.axvline(x_[2, :].mean().values, color=custom_palette_rgb[2], linestyle='-')
        # remove spines and yticks
        sns.despine(ax=ax)
        ax.set_yticks([])
        # set y label to be model
        ax.set_ylabel(model_sh[i])

    fig.show()
    # save figure
    fig.savefig(os.path.join(PLOTDIR, f'model_voxels_relation_{bench_key}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'model_voxels_relation_{bench_key}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

#%%
    #%%
    score_xr= [xr.concat(x, dim='benchmark').squeeze().mean('split') for x in score_dat]
    fig = plt.figure(figsize=(11, 8))
    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=.2, hspace=.2)
    for i in range(7):
        x_ = score_xr[i]
        # selet x_ to be only voxels with repettion_corr_ratio of >0.95
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        # select roi 1
        ax = fig.add_subplot(gs[i])
            # make axes to be square
            # do a scatter of row 0 vs row 2
        # do a displot
        sns.scatterplot(x=x_[0, :].values, y=x_[2, :].values, ax=ax, s=2)
        # get the limits of axes from sns plot
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # turn off the legend
        # ax.get_legend().remove()
        # set y lim to be equal to xlim
        xlim = min(x0, y0)
        ylim = max(x1, y1)
        ax.set_xlim((xlim, ylim))
        ax.set_ylim((xlim, ylim))
        # plot a unity line
        ax.plot([xlim, ylim], [xlim, ylim], color='k', linewidth=.5)
        ax.set_aspect('equal', 'box')
        # turn off axis ticks and labels
        # turn off the x axis
        if i==6:
            ax.set_xlabel('min')
            ax.set_ylabel('max')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        # limit


        sns.despine(ax=ax)
        ax.set_yticks([])
        # set y label to be model
        ax.set_title(model_sh[i])

    fig.show()
    # save figure
    fig.savefig(os.path.join(PLOTDIR, f'vox_max_min_relation_{bench_key}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'vox_max_min_relation_{bench_key}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    #%%
    fig = plt.figure(figsize=(11, 8))
    custom_palette_rgb = sns.color_palette(colors)
    # create 7 subplots
    gs = gridspec.GridSpec(7, 8, figure=fig, wspace=.1, hspace=.1)
    for i in range(7):
        x_ = score_xr[i]
        x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        # drop subject 1047
        #x_ = x_.sel(neuroid=(x_.subject!='1047').values)
        for j, (id,grp) in enumerate(x_.groupby('subject')):
            ax = fig.add_subplot(gs[i, j])
            # create a dataframe with 1 coloum for each row in grp, and 1 colurm that is the benchmark
            df=pd.DataFrame(grp.values.T,columns=grp.benchmark.values)
            sns.ecdfplot(df[0],ax=ax, color=custom_palette_rgb[0])
            #sns.ecdfplot(df[1], ax=ax, color=custom_palette_rgb[1])
            sns.ecdfplot(df[2], ax=ax, color=custom_palette_rgb[2])
        # j is zero add a y label, which is the model,
            if j==0:
                ax.set_ylabel(model_sh[i], fontsize=8)
            else:
                ax.set_ylabel('')

        # if i is zero then add a title which is the id
            if i==0:
                #ax.set_title(id, fontsize=6)
                # count number of voxels
                ax.set_title(f'{id}, #vox={grp.shape[1]}', fontsize=8)

        # turn off the x axis
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

    fig.show()
    fig.savefig(os.path.join(PLOTDIR, f'model_voxels_relation_per_subject_{bench_key}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'model_voxels_relation_per_subject_{bench_key}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%%
    models_scores = []
    for i in range(7):
        x_ = score_xr[i]
        # selet x_ to be only voxels with repettion_corr_ratio of >0.95
        # x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        #x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        # do it based on voxel repetition_corr
        #x_ = x_.where(x_.repetition_corr > 0, drop=True)
        # drop subject 1047
        score=x_.groupby('subject').median('neuroid')
        center = score.median('subject')
        subject_values = np.nan_to_num(score.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = score.dims.index(score['subject'].dims[0])
        error = median_abs_deviation(subject_values, axis=subject_axis)
        benchmark_score=np.concatenate([center.values[:,None],error[:,None]],axis=1)
        models_scores.append(benchmark_score)

    models_scores = np.stack(models_scores)
    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    # fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(models_scores.shape[0])

    model_name = model_sh

    rects2 = ax.bar(x - 0.25, models_scores[:, 0, 0], width, label='min', color=colors[0], linewidth=.5, edgecolor='k')
    ax.errorbar(x - 0.25, models_scores[:, 0, 0], yerr=models_scores[:, 0, 1], linestyle='', color='k')
    # plot the second item
    rects2 = ax.bar(x, models_scores[:, 1, 0], width, label='rand', color=colors[1], linewidth=.5, edgecolor='k')
    ax.errorbar(x, models_scores[:, 1, 0], yerr=models_scores[:, 1, 1], linestyle='', color='k')
    # plot the third item
    rects3 = ax.bar(x + 0.25, models_scores[:, 2, 0], width, label='max', color=colors[2], linewidth=.5, edgecolor='k')
    ax.errorbar(x + 0.25, models_scores[:, 2, 0], yerr=models_scores[:, 2, 1], linestyle='', color='k')

    # ax.errorbar(x  , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation (un-normalized)')
    ax.set_title(f'Layer performance for models used ANNSet1 \n on DsParametric')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=45)
    ax.set_ylim((-.175, 0.175))
    ax.set_xlim((-.5, 6.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{bench_key}_reliable_voxels_ratio_95.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{bench_key}_reliable_voxels_ratio_95.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


    #%%
    fig = plt.figure(figsize=(11, 8))
    x_ = xr.concat(score_dat[1], dim='benchmark').squeeze()
    # plot the distirbution of reptetion corr for each subject
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=.5, hspace=.5)
    for i, (id, grp) in enumerate(x_.groupby('subject')):
        ax = fig.add_subplot(gs[i])
        sns.distplot(grp.repetition_corr.values, rug=True,color=(.5,.5,.5), ax=ax)
        grp_cr=grp.where(grp.repetition_corr_ratio > .95, drop=True)
        sns.distplot(grp_cr.repetition_corr.values, rug=True,color=(.1,.1,.1), ax=ax)
        # plot mean of each subject
        ax.axvline(grp.repetition_corr.mean().values, color='k', linestyle='-')
        ax.set_title(f'{id}, vox={np.sum(grp.repetition_corr_ratio > 0).squeeze().values},\n relaible vox={np.sum(grp.repetition_corr_ratio > .95).squeeze().values}')
        # remove y label
        ax.set_ylabel('')
        ax.set_xlabel('')
        # remove spines
        sns.despine(ax=ax)

    fig.show()

    fig.savefig(os.path.join(PLOTDIR, f'voxel_relaible_values_{bench_key}_ratio_95.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'voxel_relaible_values_{bench_key}_ratio_95.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    #%%
    models_scores = []
    for i in range(7):
        x_ = score_xr[i]
        # selet x_ to be only voxels with repettion_corr_ratio of >0.95
        # x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        x_ = x_.where(x_.repetition_corr_ratio > .95, drop=True)
        benchmark_score=dict()
        for roi,x_roi in x_.groupby('roi'):

        # do it based on voxel repetition_corr
        # x_ = x_.where(x_.repetition_corr > 0, drop=True)
        # drop subject 1047
            score = x_roi.groupby('subject').median('neuroid')
            center = score.median('subject')
            subject_values = np.nan_to_num(score.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
            subject_axis = score.dims.index(score['subject'].dims[0])
            error = median_abs_deviation(subject_values, axis=subject_axis)
            benchmark_score[roi]=np.concatenate([center.values[:, None], error[:, None]], axis=1)
        models_scores.append(benchmark_score)


    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    # reduce figure page margin to fit all subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=.2, hspace=.2)
    # fig_length = 0.055 * len(models_scores)
    rois = list(models_scores[0].keys())
    for i in range(12):
        ax=fig.add_subplot(gs[i])
        model_score=[x[rois[i]] for x in models_scores]

        model_val=np.array(model_score)

        x = np.arange(model_val.shape[0])
        #model_name = model_sh
        rects2 = ax.bar(x - 0.25, model_val[:, 0, 0], width, label='min', color=colors[0], linewidth=.5, edgecolor='k')
        #ax.errorbar(x - 0.25, model_val[:, 0, 0], yerr=model_val[:, 0, 1], linestyle='', color='k')
        # plot the second item
        rects2 = ax.bar(x, model_val[:, 1, 0], width, label='rand', color=colors[1], linewidth=.5, edgecolor='k')
        #ax.errorbar(x, model_val[:, 1, 0], yerr=model_val[:, 1, 1], linestyle='', color='k')
        # plot the third item
        rects3 = ax.bar(x + 0.25, model_val[:, 2, 0], width, label='max', color=colors[2], linewidth=.5, edgecolor='k')
        #ax.errorbar(x + 0.25, model_val[:, 2, 0], yerr=model_val[:, 2, 1], linestyle='', color='k')
        # ax.errorbar(x  , models_scores[:, 0], yerr=models_scores[:, 1], linestyle='', color='k')
        ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Pearson correlation (un-normalized)')
        ax.set_title(f'{rois[i]}',fontsize=6)
        # set font size to 6
        ax.tick_params(axis='both', which='major', labelsize=6)
        if i in [9,10,11]:
            #ax.set_xlabel('Model',fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(model_sh, rotation=45)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

        ax.set_ylim((-.2, 0.2))
        ax.set_xlim((-.5, 6.5))
        #ax.legend()
    #ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.show()

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{bench_key}_reliable_voxels_ratio_95.png'),
                dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    fig.savefig(os.path.join(PLOTDIR, f'ANN_models_scores_DsParametric_{bench_key}_reliable_voxels_ratio_95.eps'),
                format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)