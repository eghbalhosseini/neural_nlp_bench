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
from scipy.ndimage import gaussian_filter1d

if user=='eghbalhosseini':
    analysis_dir='/Users/eghbalhosseini/MyData/NOL_paper/suppl_fig_7/'
    result_dir='/Users/eghbalhosseini/MyData/NOL_paper/suppl_fig_7/'



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth
if __name__ == "__main__":
    model_1B='gpt2-neox-pos_learned-1B'
    loss_1B_ckpnt='310000'
    model_100M = 'gpt2-neox-pos_learned-100M'
    #loss_100M_ckpnt='11600'
    loss_100M_ckpnt='14250'
    model_10M = 'gpt2-neox-pos_learned-10M'
    #loss_10m_ckpnt='1900'
    loss_10M_ckpnt = '2000'
    model_1M = 'gpt2-neox-pos_learned-1M'
    loss_1M_ckpnt = '1000'
    chkpoints_srt = ['1M', '10M', '100M', '1B']
    train_select=[1000,2000,14250,310000]
    train_stoppings = [2000, 8000, 40000, 320000]

    training_results = pd.read_pickle(os.path.join(result_dir, 'data', 'miniberta_train_valid_set.pkl'))
    # find the location of perplexity score closest to checkpoint:
    valid_chkpnt = [training_results[key]['validation_result'][:, 0].astype(int) for key in training_results.keys()]
    valid_loss = [training_results[key]['validation_result'][:, 1] for key in training_results.keys()]

    training_chkpnt = [training_results[key]['train_result'][:, 0].astype(int) for key in training_results.keys()]
    training_loss = [training_results[key]['train_result'][:, 1] for key in training_results.keys()]

    # reorder them to be 10m, 100m, 1B
    training_chkpnt.reverse()
    training_loss.reverse()

    valid_chkpnt.reverse()
    valid_loss.reverse()
    train_cuts=[]
    valid_cuts=[]
    for idx, x in enumerate(train_stoppings):
        train_cuts.append(np.stack([training_chkpnt[idx][training_chkpnt[idx] <= x],
                                    training_loss[idx][training_chkpnt[idx] <= x]]))

        valid_cuts.append(np.stack([valid_chkpnt[idx][valid_chkpnt[idx] <= x],
                                    valid_loss[idx][valid_chkpnt[idx] <= x]]))

    #

    cmap_all = cm.get_cmap('viridis')
    all_col = cmap_all(np.divide(np.arange(len(train_cuts)+1), len(train_cuts)+1))
    all_col= all_col[1:,:]
    fig = plt.figure(figsize=(11, 8), dpi=300, frameon=False)
    ax = plt.axes((.1, .2, .35, .35))
    for idx in range(len(train_cuts)):
        #ax.plot(smooth(train_cuts[idx][0],20), smooth(train_cuts[idx][1],20), c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.plot(train_cuts[idx][0], gaussian_filter1d(train_cuts[idx][1],10), c=(all_col[idx, :]), zorder=2,
                label=chkpoints_srt[idx],linewidth=2)
        ax.axvline(train_select[idx],c=(all_col[idx, :]))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylabel('training loss')
    ax.set_xlabel('training step')
    ax.legend(bbox_to_anchor=(1.2, .8), frameon=True, fontsize=8)

    ax = plt.axes((.6, .2, .35, .35))
    for idx in range(len(train_cuts)):
        # ax.plot(smooth(train_cuts[idx][0],20), smooth(train_cuts[idx][1],20), c=(all_col[idx,:]), zorder=2,label=chkpoints_srt[idx])
        ax.plot(valid_cuts[idx][0], valid_cuts[idx][1], c=(all_col[idx, :]), zorder=2,
                label=chkpoints_srt[idx], linewidth=2)
        ax.axvline(train_select[idx], c=(all_col[idx, :]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    plt.grid(True, which="both", ls="-", color='0.9', zorder=0)
    ax.set_ylabel('validation loss')
    ax.set_xlabel('training step')


    fig.show()
    fig.savefig(os.path.join(analysis_dir,f'gpt_neox_training_performance.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'gpt_neox_training_performance.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)
