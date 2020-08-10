#!/bin/bash
#SBATCH --job-name=nlp-weight
#SBATCH --array=0-90
#SBATCH --time=56:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --exclude node017,node018



i=0
for trained in "" "-untrained" ; do
for benchmark in Fedorenko2016v3-encoding-weights ; do
for model in  bert-base-uncased bert-base-multilingual-cased bert-large-uncased \
  openaigpt gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 \
  transfo-xl-wt103 \
  xlnet-base-cased xlnet-large-cased xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 xlm-clm-enfr-1024 xlm-mlm-100-1280 \
  roberta-base roberta-large distilroberta-base distilbert-base-uncased \
  albert-base-v1 albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 \
  xlm-roberta-base xlm-roberta-large ; do
  model_list[$i]="$model$trained"
  benchmark_list[$i]="$benchmark"
  i=$[$i +1]
done
done
done

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec -B /om:/om /om/user/`whoami`/simg_images/arch_search.simg python ~/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --log_level DEBUG
