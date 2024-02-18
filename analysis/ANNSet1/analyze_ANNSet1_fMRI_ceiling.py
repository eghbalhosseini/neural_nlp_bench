import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import getpass
import matplotlib
import xarray as xr
user=getpass.getuser()
print(user)
# add root directory to python path
from pathlib import Path
ROOTDIR = (Path('/Users/eghbalhosseini/MyData/neural_nlp_bench/analysis/LangLoc_ECoG/') ).resolve()
OUTDIR = (Path(ROOTDIR / 'models')).resolve()
PLOTDIR = (Path(ROOTDIR / 'plots')).resolve()
from glob import glob
if user=='eghbalhosseini':
    result_caching=('/Users/eghbalhosseini/.result_caching/')
    ceiling_dir='/Users/eghbalhosseini/.result_caching/neural_nlp.benchmarks.ceiling.FewSubjectExtrapolation.__call__/'
elif user=='ehoseini':
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'
    ceiling_dir = '/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.benchmarks.ceiling.FewSubjectExtrapolation.__call__/'

if __name__ == "__main__":
    benchmark= 'ANNSet1fMRI-wordForm-encoding'
    ceiling_file = glob(os.path.join(ceiling_dir, f'identifier={benchmark}*.pkl'))
    assert len(ceiling_file) > 0
    ceiling_data = pd.read_pickle(ceiling_file[0])['data']
    ceiling_raw=ceiling_data.raw
    vox_means=[]
    for grp_id, grp in ceiling_raw.groupby('subject'):
        vox_mean=grp.groupby('subsample').mean()
        vox_means.append(vox_mean)

    cmap_all = matplotlib.colormaps['plasma']
    all_col = cmap_all(np.divide(np.arange(len(vox_means)), len(vox_means)))
    fig = plt.figure(figsize=(11,8), dpi=200, frameon=False)
    pap_ratio=8/11
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax = plt.axes((.1, .65, .35, .35*pap_ratio))
    spread=.01
    ofsets=0.1
    for idx , grp in enumerate(vox_means):
        # do a box plot for each subject
        True
        vox_medians=grp.median('neuroid')
        vox_std = grp.std('neuroid')
        sub_samples=grp.subsample.values
        #
        # Generating positions for each point
        off=ofsets*(idx-(len(vox_means)/2)+1)
        x_positions = np.random.uniform(-spread, spread, size=grp.values.shape[1])

        #ax.boxplot(grp.values.T, positions=sub_samples+off, widths=0.2, showfliers=False, showmeans=True,
        #           meanprops={'marker': 's', 'markerfacecolor': 'red', 'markeredgecolor': 'none', 'markersize': 4})
        for id_,sample_ in enumerate(sub_samples):
            True
            ax.scatter(x_positions+(sample_+off), np.sort(grp[id_,:].values), alpha=0.5, s=1,color=all_col[idx])
        # plot a line with the median of the subject
        ax.plot(sub_samples,vox_medians.values, color=all_col[idx],linewidth=2)


    fig.show()