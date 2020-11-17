#!/bin/bash
#SBATCH --job-name=arch_search
#SBATCH --array=0-9
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=267G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexso@mit.edu

i=0
for benchmark in Fedorenko2016v3-encoding-weights Pereira2018-encoding-weights ; do
for model in gpt2 transfo-xl-wt103 roberta-base xlm-mlm-xnli15-1024 albert-base-v2  ; do
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

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master.simg python /home/`whoami`/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
