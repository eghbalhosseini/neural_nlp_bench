#!/bin/bash

#SBATCH --job-name=nlp_NYU
#SBATCH --array=0-12
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=267G

i=0

for benchmark in Blank2014fROI-encoding ; do
for model in roberta-base \
  nyu-mll/roberta-base-1B-1 nyu-mll/roberta-base-1B-2 nyu-mll/roberta-base-1B-3 \
  nyu-mll/roberta-base-100M-1 nyu-mll/roberta-base-100M-2 nyu-mll/roberta-base-100M-3 \
  nyu-mll/roberta-base-10M-1 nyu-mll/roberta-base-10M-2 nyu-mll/roberta-base-10M-3 \
  nyu-mll/roberta-med-small-1M-1 nyu-mll/roberta-med-small-1M-2 nyu-mll/roberta-med-small-1M-3 ; do
  model_list[$i]="$model"
  benchmark_list[$i]="$benchmark"
  i=$[$i +1]
done
done

#echo ${#model_list[@]}
#exit

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec -B /om:/om /om/user/`whoami`/simg_images/arch_search.simg python ~/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --log_level DEBUG

