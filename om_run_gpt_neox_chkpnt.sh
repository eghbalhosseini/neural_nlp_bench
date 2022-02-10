#!/bin/bash
#SBATCH --job-name=nlpGptNeoX
#SBATCH --array=0-7
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --output=R-%x.%j.out

i=0
for benchmark in Fedorenko2016v3-encoding Pereira2018-encoding  ; do
  for model in gpt2-neox-100M-v2 gpt2-neox-100M-v2-untrained \
                gpt2-neox-10B-v2 gpt2-neox-10B-v2-untrained ; do

            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
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

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /om/user/ehoseini/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
