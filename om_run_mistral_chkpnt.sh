#!/bin/bash
#SBATCH --job-name=nlpMistral
#SBATCH --array=0-3
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0
for benchmark in Pereira2018-encoding ; do
  for model in mistral/eowyn-gpt2-medium-x777  ; do
      for checkpoint in `seq 0 10000 20000`; do
            model_list[$i]="${model}/ckpt_${checkpoint}-untrained"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
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

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /home/`whoami`/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
