import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import getpass
import matplotlib
user=getpass.getuser()
print(user)
# add root directory to python path
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from pathlib import Path
ROOTDIR = (Path('/Users/eghbalhosseini/MyData/neural_nlp_bench/analysis/LangLoc_ECoG/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'models')).resolve()
PLOTDIR = (Path(ROOTDIR / 'plots')).resolve()
from glob import glob
if user=='eghbalhosseini':
    result_caching='/Users/eghbalhosseini/.result_caching/'
elif user=='ehoseini':
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    benchmark = 'LangLocECoG-uni-gaus-Encoding'
    model= 'gpt2-large'
    # create model directory
    model_dir = Path(OUTDIR, model)
    # make sure parent directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    # make directory for this model
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model},*.pkl'))
    assert len(files)>0
    scores_mean=[]
    scors_std=[]
    model_data=pd.read_pickle(files[0])['data']
    scores_mean=model_data.values[:,0]
    scores_std=model_data.values[:,1]
    l_names=model_data.layer.values
    cmap_all = matplotlib.colormaps['plasma']
    all_col = cmap_all(np.divide(np.arange(len(scores_mean)), len(scores_mean)))
    width = 0.7  # the width of the bars
    fig = plt.figure(figsize=(11, 8),dpi=250,frameon=False)
    fig_length=0.0135*len(l_names)
    ax = plt.axes((.1, .6, fig_length, .35))
    x = np.arange(scores_mean.shape[0])
    y = scores_mean
    y_err = scores_std
    layer_name = l_names
    rects1 = ax.bar(x, y, width, color=np.divide((188, 80, 144), 255))
    ax.errorbar(x, y, yerr=y_err, linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'{model} \n performance on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_name,rotation=90)
    ax.set_ylim((-.05, .1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # find layer with maximum score
    max_score_idx = np.argmax(scores_mean)
    score_raw=model_data.raw
    # select layer with maximum score
    max_score_layer = score_raw[max_score_idx]
    # take the mean ove splits for max score layer
    max_score_layer= max_score_layer.mean('split')
    # split by subject
    max_score_layer_by_subject = max_score_layer.groupby('subject')
    # for each subject compute mean score across all electrodes

    width = 0.7  # the width of the bars
    spread= 0.1
    # create a colormap with the size of number of subjects
    cmap_all = matplotlib.colormaps['Dark2']
    all_col = cmap_all(np.divide(np.arange(len(max_score_layer_by_subject)), len(max_score_layer_by_subject)))
    fig_length = 0.02 * len(max_score_layer_by_subject)
    ax = plt.axes((.1, .1, fig_length, .35))
    # for each subject plot the mean score and the individual electrode score
    for idx, (sub_id,grp) in enumerate(max_score_layer_by_subject):
        # do a box plot for each subject
        ax.boxplot(grp.values, positions=[idx], widths=0.6,showfliers=False,showmeans=True,meanprops={'marker':'o','markerfacecolor':'black','markeredgecolor':'black'})
        # Generating positions for each point
        x_positions = np.random.uniform(-spread, spread, size=len(grp.values))
        ax.scatter(x_positions+ idx, np.sort(grp.values), alpha=0.5, s=5,color=all_col[idx])
    # plot a horizontal line at zero
    ax.axhline(y=0, color='k', linestyle='-')
    # add title
    ax.set_title(f'{model} ; on {benchmark}\n performance of best layer on subjects')
    ax.set_ylabel('Pearson correlation')
    # make the title such that mode land benchmarks are bold

    ax.set_xticks(np.arange(len(max_score_layer_by_subject)))
    ax.set_xticklabels(max_score_layer_by_subject.groups.keys(),rotation=90)

    # in a new sbplot scatter plot of max_score_layer.values vs s_v_n_ratio
    ax = plt.axes((fig_length+.2, .1, .25, .35))
    ax.scatter(max_score_layer.s_v_n_ratio.values,max_score_layer.values , alpha=0.5, s=5)
    # set xaxis to log scale
    ax.set_xscale('log')
    # onlt show points [0.95,0.99,1] on x axis



    # set x ticks to go from .95 to 1
    # make xtick labels vertical
    ax.set_xticks([.95,.99,1])
    ax.set_xticklabels([.95,.99,1],rotation=90)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    # add title and labels
    ax.set_title('electrode score vs s_v_n_ratio')
    ax.set_xlabel('s_v_n_ratio')
    ax.set_ylabel('electrode score')
    # save figure
    fig.savefig(os.path.join(model_dir, f'score_{model}_{benchmark}.png'), dpi=250, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    fig.savefig(os.path.join(model_dir, f'score_{model}_{benchmark}.eps'), format='eps', metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)