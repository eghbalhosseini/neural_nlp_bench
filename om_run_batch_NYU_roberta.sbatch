#!/bin/bash

#SBATCH --job-name=nlp_NYU
#SBATCH --array=0-8
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0

for benchmark in Pereira2018-encoding ; do
    for model in roberta-base , roberta-base-untrained \
        nyu-mll/roberta-base-1B-1 nyu-mll/roberta-base-1B-1-untrained \
        nyu-mll/roberta-base-100M-1 nyu-mll/nyu-mll/roberta-base-100M-1-untrained \
        nyu-mll/roberta-base-10M-1 nyu-mll/roberta-base-10M-1-untrained  ; do
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



#singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_fz.simg python neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

#singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /om/user/ehoseini/neural-nlp-nyu/neural_nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
singularity exec -B /om:/om,/om5:/om5 /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /home/`whoami`/neural-nlp-master/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"