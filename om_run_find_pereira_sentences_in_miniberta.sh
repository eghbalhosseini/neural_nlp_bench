#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-50
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for data in 1B  ; do
  # create a sequence of 50 numbers
  for chunk in $(seq 0 49); do
            data_list[$i]="${data}"
            chunk_list[$i]="${chunk}"
            i=$[$i+1]
  done
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running data ${data_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
conda activate neural_nlp_2022
which python
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/analysis/NOL_paper/find_pereira_sentences_in_miniberta_dataset.py --data "${data_list[$SLURM_ARRAY_TASK_ID]}" --chunk_id "${chunk_list[$SLURM_ARRAY_TASK_ID]}"