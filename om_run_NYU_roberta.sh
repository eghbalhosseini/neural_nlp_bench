#!/bin/bash
#SBATCH --job-name=NYU_roberta
#SBATCH --array=0-5
#SBATCH --time=56:00:00
#SBATCH --mem=60G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in Pereira2018-encoding Blank2014fROI-encoding Futrell2018-encoding Futrell2018-stories_encoding Futrell2018-sentences_encoding ; do
for model in nyu-mll/roberta-base-1B-1-untrained ; do

#for model in nyu-mll/roberta-base-1B-1 nyu-mll/roberta-base-1B-2 nyu-mll/roberta-base-1B-3 nyu-mll/roberta-base-100M-1 nyu-mll/roberta-base-100M-2 nyu-mll/roberta-base-100M-3 \

# nyu-mll/roberta-base-10M-2 nyu-mll/roberta-base-10M-1 nyu-mll/roberta-base-10M-3 ; do
  model_list[$i]="$model"
  benchmark_list[$i]="$benchmark"
  i=$i+1
done
done


RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"



. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022
/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

#singularity exec -B /om:/om /om/user/`whoami`/simg_images/arch_search.simg python ~/neural-nlp/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --log_level DEBUG
