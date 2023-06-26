#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=1-33
#SBATCH --time=32:00:00
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=1
for benchmark in  DsParametricfMRI_v1-max-RidgeEncoding \
                  DsParametricfMRI_v1-min-RidgeEncoding \
                  DsParametricfMRI_v1-rand-RidgeEncoding ; do
  for model in  roberta-base \
                xlnet-large-cased \
                bert-large-uncased-whole-word-masking \
                xlm-mlm-en-2048 \
                gpt2-xl \
                albert-xxlarge-v2 \
                ctrl \
                distilgpt2 \
                gpt2 \
                gpt2-medium \
                gpt2-large ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

#Blank2014fROI-encoding
module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
echo "cache id " $RESULTCACHING_HOME


. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

which python

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
