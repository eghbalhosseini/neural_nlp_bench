#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-99
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in DsParametricfMRI-second-max-Encoding DsParametricfMRI-second-min-Encoding DsParametricfMRI-second-rand-Encoding \
  DsParametricfMRI-first-max-StrictEncoding DsParametricfMRI-first-min-StrictEncoding DsParametricfMRI-first-rand-StrictEncoding \
  DsParametricfMRI-second-max-StrictEncoding DsParametricfMRI-second-min-StrictEncoding DsParametricfMRI-second-rand-StrictEncoding ; do
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
