#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-52
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for ngram in 1 2 3 4  ; do
  for data_id in 0 1 2 3 4 5 6 7 8 9 10 11 12 ; do
            ngram_list[$i]="${ngram}"
            data_id_list[$i]="${data_id}"
            i=$[$i+1]
done
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running ngram ${ngram_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running data ${data_id_list[$SLURM_ARRAY_TASK_ID]}"



. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

which python

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/analysis/NOL_paper/analyze_unigram_bigram_trigram_freq_in_openwebtext_dataset.py --ngram "${ngram_list[$SLURM_ARRAY_TASK_ID]}" --data_id "${data_id_list[$SLURM_ARRAY_TASK_ID]}"