#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-66
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in DsParametricfMRI-80-first-max-Encoding DsParametricfMRI-80-first-min-Encoding DsParametricfMRI-80-first-rand-Encoding \
  DsParametricfMRI-80-second-max-Encoding DsParametricfMRI-80-second-min-Encoding DsParametricfMRI-80-second-rand-Encoding ; do
  #DsParametricfMRI-first-max-RidgeEncoding DsParametricfMRI-first-min-RidgeEncoding DsParametricfMRI-first-rand-RidgeEncoding \
  #DsParametricfMRI-second-max-RidgeEncoding DsParametricfMRI-second-min-RidgeEncoding DsParametricfMRI-second-rand-RidgeEncoding
  for model in bert-large-uncased-whole-word-masking \
               xlnet-large-cased \
               roberta-base \
               xlm-mlm-en-2048 \
               gpt2-xl \
               distilgpt2 \
               gpt2 \
               gpt2-medium \
               gpt2-large \
               albert-xxlarge-v2 \
               ctrl ; do

            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
echo "cache id " $RESULTCACHING_HOME

. ~/.bash_profile
conda activate neural_nlp_2022

which python

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
