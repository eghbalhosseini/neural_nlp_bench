#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for data in 1M 10M 100M 1B  ; do
            data_list[$i]="${data}"
            i=$[$i+1]
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running data ${data_list[$SLURM_ARRAY_TASK_ID]}"



. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

which python

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/analysis/NOL_paper/analyze_unigram_bigram_trigram_freq_in_miniberta_dataset.py --data "${data_list[$SLURM_ARRAY_TASK_ID]}"