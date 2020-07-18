#!/bin/bash
#SBATCH --job-name=arch_search
#SBATCH --array=0
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=267G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in Pereira2018-encoding ; do
for model in distilgpt2 ; do
  model_list[$i]="$model"
  benchmark_list[$i]="$benchmark"
  i=$i+1
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

singularity exec -B /om:/om /om/user/`whoami`/simg_images/arch_search.simg python ~/MyCodes/arch_search/test_tranformer_lib_on_nlp_pipeline.py "${model_list[$SLURM_ARRAY_TASK_ID]}" "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
#singularity exec -B /om:/om /om/user/`whoami`/simg_images/arch_search.simg python ~/neural-nlp/neural_nlp run --model gpt2-xl --benchmark Blank2014fROI-encoding --log_level DEBUG
