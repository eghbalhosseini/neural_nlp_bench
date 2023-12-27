#!/bin/bash

#SBATCH --job-name=MghLang
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0
ROOT_DIR=/om/weka/evlab/ehoseini/neural-nlp-1/

for benchmark in MghLanglocAudioV1-encoding ; do
    for model in gpt2 gpt2-untrained gpt2-medium gpt2-medium-untrained gpt2-large gpt2-large-untrained gpt2-xl gpt2-xl-untrained ; do
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
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om2/user/`whoami`/st
export XDG_CACHE_HOME


singularity exec -B /om:/om,/om2:/om2,/om5:/om5 /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /om/weka/evlab/ehoseini/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"


