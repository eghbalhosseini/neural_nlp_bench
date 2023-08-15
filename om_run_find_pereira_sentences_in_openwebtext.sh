#!/bin/bash
#SBATCH --job-name=PerOpenWebText
#SBATCH --array=0-650%200
#SBATCH --time=5:00:00
#SBATCH --mem=10G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for data_id in 0 1 2 3 4 5 6 7 8 9 10 11 12 ; do
      for chunk in $(seq 0 49); do
            data_list[$i]="${data_id}"
            chunk_list[$i]="${chunk}"
            i=$[$i+1]
  done
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running data ${data_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
conda activate neural_nlp_2022
which python
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/analysis/NOL_paper/find_pereira_sentences_in_openwebtext_dataset.py --data_id "${data_list[$SLURM_ARRAY_TASK_ID]}" --chunk_id "${chunk_list[$SLURM_ARRAY_TASK_ID]}"