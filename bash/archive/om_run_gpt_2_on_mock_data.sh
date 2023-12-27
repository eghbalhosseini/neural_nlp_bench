#!/bin/bash
#SBATCH --job-name=nlpMock
#SBATCH --array=0-1
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in MghMockLang-encoding ; do
  for model in distilgpt2 distilgpt2-untrained ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om2/user/`whoami`/st
export XDG_CACHE_HOME


singularity exec -B /om:/om,/om2:/om2,/om5:/om5 /om/user/`whoami`/simg_images/test/neural_nlp_2020 python /om/weka/evlab/ehoseini/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
