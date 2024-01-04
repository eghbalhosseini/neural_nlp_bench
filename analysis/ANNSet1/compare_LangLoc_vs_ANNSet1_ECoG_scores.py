import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import getpass
user=getpass.getuser()
print(user)

from glob import glob
if user=='ehoseini':
    analysis_dir='/rdma/vast-rdma/vast/evlab/ehoseini/MyData/brain-score-language/analysis/'
    result_caching='/om5/group/evlab/u/ehoseini/.result_caching/'

if __name__ == "__main__":
    ANN_bench='ANNSet1ECoG-uni-band-Encoding'
    langloc_bench='LangLocECoG-uni-band-Encoding'
    model_layers = [('roberta-base', 'encoder.layer.1'),
                    ('xlnet-large-cased', 'encoder.layer.23'),
                    ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                    ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                    ('gpt2-xl', 'encoder.h.43'),
                    ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                    ('ctrl', 'h.46')]
    model, layer = model_layers[4]
    files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={ANN_bench},model={model},*.pkl'))
    assert len(files) > 0
    # read file
    x_ann = pd.read_pickle(files[0])['data']
    # read langloc file
    langloc_files = glob(os.path.join(result_caching, 'neural_nlp.score', f'benchmark={langloc_bench},model={model},*.pkl'))
    assert len(langloc_files) > 0
    x_langloc = pd.read_pickle(langloc_files[0])['data']

    x_ann=x_ann.raw
    x_langloc=x_langloc.raw
    # find the location x_ann neuroids in x_langloc
    electrodes= x_ann['neuroid_id'].astype(str).values
    electrodes_langloc=x_langloc['neuroid_id'].astype(str).values
    # find the intersection of electrodes
    electrodes_intersect=np.intersect1d(electrodes,electrodes_langloc)
    # for each dataset only select the neuroids that are in the intersection
    index_overlap_ann=np.in1d(electrodes,electrodes_intersect)
    index_overlap_langloc=np.in1d(electrodes_langloc,electrodes_intersect)
    x_ann=x_ann[:,:,index_overlap_ann]
    x_langloc=x_langloc[:,:,index_overlap_langloc]
    # align them so that they have the same order
    assert ( x_ann['neuroid_id'].astype(str).values==x_langloc['neuroid_id'].astype(str).values).all()
    # for both compute average over split dimension
    x_ann=x_ann.mean('split')
    x_langloc=x_langloc.mean('split')
    # first find index of layer in x_ann
    layer_index_ann=np.where(x_ann['layer'].values==layer)[0][0]
    x_ann_layer=x_ann.isel(layer=layer_index_ann)
    # find the index of layer in x_langloc
    layer_index_langloc=np.where(x_langloc['layer'].values==layer)[0][0]
    x_langloc_layer=x_langloc.isel(layer=layer_index_langloc)

    elec_ann_vec = np.transpose(x_ann_layer.values)
    elec_langloc_vec = np.transpose(x_langloc_layer.values)


    colors = [ np.divide((128, 128, 128), 256), np.divide((255, 98, 0), 255)]
    fig = plt.figure(figsize=(8, 11), dpi=300, frameon=False)
    pap_ratio = 11 / 8
    ax = plt.axes((.1, .05, .15, .3 * pap_ratio))

    elec_vec = np.vstack((elec_langloc_vec, elec_ann_vec))
    # plot one line per column in rdm_vec
    for i in range(elec_vec.shape[1]):
        ax.plot([1, 2], elec_vec[:, i], color='k', alpha=.3, linewidth=.5)
        # plot a scatter with each point color same as color_set
        ax.scatter([1, 2], elec_vec[:, i], color=colors, s=10, marker='o', alpha=.5)
    # use a boxplot to show the distribution of rdm values per row, with colors matching above scatter plot

    ax.boxplot(elec_vec.transpose(), vert=True, showfliers=False, showmeans=False,
               meanprops={'marker': 'o', 'markerfacecolor': 'r', 'markeredgecolor': 'k'})
    # set xtick labels to ds_min, ds_rand, ds_max
    ax.set_xticklabels(['langloc', 'ANN'], fontsize=8)
    ax.set_ylabel('Ds')
    #ax.set_ylim((0, 1.3))
    ax.set_title('Ds distribution')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim((.75, 2.25))

    fig.show()






    ann_models_scores = np.stack(ann_models_scores)
    langloc_models_scores = np.stack(langloc_models_scores)
    width = 0.25  # the width of the bars
    fig = plt.figure(figsize=(11, 8))
    # fig_length = 0.055 * len(models_scores)
    ax = plt.axes((.1, .4, .35, .35))
    x = np.arange(ann_models_scores.shape[0])

    model_name = [f'{x[0]} \n {x[1]}' for x in model_layers]

    rects1 = ax.bar(x, langloc_models_scores[:, 0], width, color=np.divide((55, 76, 128), 256), label='LangLocECoG')
    ax.errorbar(x, langloc_models_scores[:, 0], yerr=langloc_models_scores[:, 1], linestyle='', color='k')

    rects2 = ax.bar(x + width, ann_models_scores[:, 0], width, label='ANNSet1_ECoG',
                    color=np.divide((188, 80, 144), 255))
    ax.errorbar(x + width, ann_models_scores[:, 0], yerr=ann_models_scores[:, 1], linestyle='', color='k')
    ax.axhline(y=0, color='k', linestyle='-')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Pearson correlation')
    #ax.set_title(f'Layer performance for models used ANNSet1 \n on {benchmark}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_name, rotation=90)
    #ax.set_ylim((-.15, 1.15))
    ax.set_xlim((-.5, 6.5))
    ax.legend()
    ax.legend(bbox_to_anchor=(1.5, .8), frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.show()
    # fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_{benchmark}.png'), dpi=250, format='png',
    #             metadata=None,
    #             bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
    #
    # fig.savefig(os.path.join(analysis_dir, f'ANN_models_scores_{benchmark}.eps'), format='eps', metadata=None,
    #             bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


