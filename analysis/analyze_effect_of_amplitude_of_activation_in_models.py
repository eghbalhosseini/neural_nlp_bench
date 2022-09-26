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

if user=='eghbalhosseini':
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
elif user=='ehoseini':
    analysis_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022//analysis/'
    result_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    stimuli_name='Pereira2018-*'
    model='mistral-caprica-gpt2-small-x81-ckpnt-400000-untrained-4'
    print(model)
    activtiy_folder='neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored'
    files=glob(os.path.join(result_caching,activtiy_folder,f'identifier={model},stimuli_identifier={stimuli_name}*.pkl'))
    # order files

    files_srt=files
    # remove chekcpoint zero if exist
    norms=[]
    for ix, file in tqdm(enumerate(files_srt)):
        x=pd.read_pickle(file)['data']
        layer_norm=[]
        for grp in x.groupby('layer'):
            layer_norm.append(linalg.norm(grp[1], axis=1))
        norms.append(np.stack(layer_norm))

    norms=np.concatenate(norms,axis=1)

    cmap_all = cm.get_cmap('plasma')

    all_col = cmap_all(np.divide(np.arange(len(norms)), len(norms)))
    fig = plt.figure(figsize=(11, 8), dpi=250, frameon=False)
    ax = plt.axes((.1, .2, .45, .35))
    idx=0
    r3 = np.arange(norms.shape[0])
    ax.plot(r3, np.mean(norms,axis=1), color=all_col[idx,:],linewidth=2)
    ax.errorbar(r3, np.mean(norms,axis=1), yerr=np.std(norms,axis=1), linewidth=2, color=all_col[idx, :],marker='.', markersize=10)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.legend(bbox_to_anchor=(1.4, 2), frameon=True,fontsize=8)
    ax.set_xlim((0-.5,norms.shape[0]-.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(norms.shape[0]))
    #fig.show()

    fig.savefig(os.path.join(analysis_dir,f'activations_{model}_.png'), dpi=250, format='png', metadata=None,
        bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)

    #fig.savefig(os.path.join(analysis_dir, f'effect_of_mu_{model}_{benchmark}.eps'), format='eps',metadata=None,
    #            bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)