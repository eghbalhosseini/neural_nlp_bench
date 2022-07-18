#!/bin/bash
#SBATCH --job-name=MISTRAL
#SBATCH --array=0-1
#SBATCH --time=6-23:00:00
#SBATCH --mem=80G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

#i=0
#for benchmark in Fedorenko2016v3-encoding ; do
#  for model in mistral/caprica-gpt2-small-x81  ; do
#      for checkpoint in `seq 0 20000 400000`; do
#            model_list[$i]="${model}/ckpt_${checkpoint}"
#            benchmark_list[$i]="$benchmark"
#            i=$[$i+1]
#      done
#    done
#done

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural_nlp_bench/compute_model_perplexity.py