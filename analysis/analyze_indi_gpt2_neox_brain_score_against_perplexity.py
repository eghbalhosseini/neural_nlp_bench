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
    #benchmark = 'Blank2014fROI-encoding'
    benchmark='Fedorenko2016v3-encoding'
    model='gpt2-neox-pos_learned-1B'
    training_key='miniberta_1b'
    files=glob(os.path.join(result_caching,'neural_nlp.score',f'benchmark={benchmark},model={model}*.pkl'))
    # order files
    chkpoints=[re.findall(r'ckpnt-\d+',x)[0] for x in files]
    chkpoints=[int(x.replace('ckpnt-','')) for x in chkpoints]
    reorder=np.argsort(chkpoints)
    chkpoints_srt=[chkpoints[x] for x in reorder]
    files_srt=[files[x] for x in reorder]

    reorder_new=np.concatenate((np.asarray([0,]),np.squeeze(np.argwhere(np.mod(np.asarray(chkpoints_srt)[:,],10000)==0))))
    files_select=[files_srt[x] for x in reorder_new]
    chkpoints_select=[chkpoints_srt[x] for x in reorder_new]

    scores_mean=[]
    scors_std=[]
    for ix, file in tqdm(enumerate(files_select)):
        x=pd.read_pickle(file)['data'].values
        scores_mean.append(x[:,0])
        scors_std.append(x[:,1])

    training_results=pd.read_pickle(os.path.join(result_dir,'data','miniberta_train_valid_set.pkl'))
    # get perplexities on validation sets
    # get max score per step
    max_score=[max(x) for x in scores_mean]
    max_score_loc = [np.argmax(x) for x in scores_mean]
    best_overall_layer=max_score_loc[np.argmax(max_score)]
    score_of_best_layer=[x[best_overall_layer] for x in scores_mean]
    # training checkpionts add 50 to it for 100M and 10M
    training_chkpnt=list(training_results[training_key]['validation_result'][:,0].astype(int))
    ckpnt_overlap=[]
    for idx, x in enumerate(chkpoints_select):
        if x in training_chkpnt:
            ckpnt_overlap.append((idx, training_chkpnt.index(x)))
    ckpnt_overlap=np.asarray(ckpnt_overlap)
    validation_perpelxity=np.asarray([training_results[training_key]['validation_result'][x,2] for x in ckpnt_overlap[:,1]])
    validation_score=np.asarray([max_score[x] for x in ckpnt_overlap[:,0]])
    validation_score_best_layer = np.asarray([score_of_best_layer[x] for x in ckpnt_overlap[:, 0]])
    cmap_all = cm.get_cmap('plasma')
    all_col = cmap_all(np.divide(np.arange(len(validation_score)), len(validation_score)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .1, .45, .35))
    ax.plot(validation_perpelxity, validation_score,zorder=1,color='k')
    ax.scatter(validation_perpelxity, validation_score, c=all_col,zorder=2)
    #ax.set_xlim([max(1.05*validation_perpelxity),min(.8*validation_perpelxity)])
    ax.set_xlim([ min(.7 * validation_perpelxity),max(1.05 * validation_perpelxity),])
    ax.set_ylim([-.1, 1])
    ax.set_xlim([20, 50])

    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr \n for best score')
    ax.set_xlabel('perplexity')

    ax = plt.axes((.1, .5, .45, .35))
    ax.plot(validation_perpelxity, validation_score_best_layer, zorder=1, color='k')
    ax.scatter(validation_perpelxity, validation_score_best_layer, c=all_col, zorder=2)
    # ax.set_xlim([max(1.05*validation_perpelxity),min(.8*validation_perpelxity)])
    ax.set_xlim([min(.7 * validation_perpelxity), max(1.05 * validation_perpelxity), ])
    ax.set_ylim([-.1, 1])
    ax.set_xlim([20, 50])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Pearson Corr \n for best layer')
    ax.set_xlabel('perplexity')

    ax.set_title(f'model:{model}\n benchmark {benchmark} against perplexity')
    fig.show()

    fig.savefig(os.path.join(analysis_dir,f'chpnt_score_{model}_{benchmark}_against_perplexity.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    fig.savefig(os.path.join(analysis_dir, f'chpnt_score_{model}_{benchmark}_against_perplexity.eps'), format='eps',metadata=None,
                bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)