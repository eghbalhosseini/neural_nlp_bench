#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  DsParametricfMRI-first-all-all-Encoding_sep2024 DsParametricfMRI-second-all-all-Encoding_sep2024 \
  DsParametricfMRI-first-all-max-Encoding_sep2024 DsParametricfMRI-first-all-min-Encoding_sep2024 DsParametricfMRI-first-all-rand-Encoding_sep2024 \
  DsParametricfMRI-second-all-max-Encoding_sep2024 DsParametricfMRI-second-all-min-Encoding_sep2024 DsParametricfMRI-second-all-rand-Encoding_sep2024 ; do
  #DsParametricfMRI-first-all-max-auditory-Encoding_sep2024 DsParametricfMRI-first-all-min-auditory-Encoding_sep2024 DsParametricfMRI-first-all-rand-auditory-Encoding_sep2024 \
  #DsParametricfMRI-second-all-max-auditory-Encoding_sep2024 DsParametricfMRI-second-all-min-auditory-Encoding_sep2024 DsParametricfMRI-second-all-rand-auditory-Encoding_sep2024 \

  #DsParametricfMRI-full-90-max-RidgeEncoding_sep2024 DsParametricfMRI-full-90-min-RidgeEncoding_sep2024 DsParametricfMRI-full-90-rand-RidgeEncoding_sep2024 \
  #DsParametricfMRI-shared-90-max-RidgeEncoding_sep2024 DsParametricfMRI-shared-90-min-RidgeEncoding_sep2024 DsParametricfMRI-shared-90-rand-RidgeEncoding_sep2024 \

  #DsParametricfMRI-first-min-RidgeEncoding_sep2024 DsParametricfMRI-first-rand-RidgeEncoding_sep2024 DsParametricfMRI-first-max-RidgeEncoding_sep2024 \
  #DsParametricfMRI-second-min-RidgeEncoding_sep2024 DsParametricfMRI-second-rand-RidgeEncoding_sep2024 DsParametricfMRI-second-max-RidgeEncoding_sep2024 \
  #DsParametricfMRI-first-max-Encoding_sep2024 DsParametricfMRI-first-min-Encoding_sep2024 DsParametricfMRI-first-rand-Encoding_sep2024 \
  #DsParametricfMRI-second-max-Encoding_sep2024 DsParametricfMRI-second-min-Encoding_sep2024 DsParametricfMRI-second-rand-Encoding_sep2024 \
  #DsParametricfMRI-first-reliable-max-Encoding_sep2024 DsParametricfMRI-first-reliable-min-Encoding_sep2024 DsParametricfMRI-first-reliable-rand-Encoding_sep2024 \
  #DsParametricfMRI-second-reliable-max-Encoding_sep2024 DsParametricfMRI-second-reliable-min-Encoding_sep2024 DsParametricfMRI-second-reliable-rand-Encoding_sep2024
  #DsParametricfMRI-shared-90-max-encoding_sep2024 DsParametricfMRI-shared-90-min-encoding_sep2024 DsParametricfMRI-shared-90-rand-encoding_sep2024 \
  #DsParametricfMRI-full-90-max-RidgeEncoding_sep2024 DsParametricfMRI-full-90-min-RidgeEncoding_sep2024 DsParametricfMRI-full-90-rand-RidgeEncoding_sep2024 \


 #bert-large-uncased-whole-word-masking \
               #xlnet-large-cased \
               #roberta-base \
               #xlm-mlm-en-2048 \
               #gpt2-xl \
               #distilgpt2 \
               #gpt2 \
               #gpt2-medium \
               #gpt2-large \
               #albert-xxlarge-v2 \
               #ctrl
               # ; do
               #sentence-length word-position random-embedding skip-thoughts lm_1b word2vec glove
  for model in lm_1b ; do
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
