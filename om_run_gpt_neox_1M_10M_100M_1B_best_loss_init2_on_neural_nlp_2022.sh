#!/bin/bash
#SBATCH --job-name=INIT2
#SBATCH --array=0-19
#SBATCH --time=6-23:00:00
#SBATCH --mem=60G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  Pereira2018-encoding Blank2014fROI-encoding Futrell2018-encoding Futrell2018-stories_encoding Futrell2018-sentences_encoding  ; do
  for model in gpt2-neox-pos_learned-1M-v2-init2-ckpnt-1000 \
              gpt2-neox-pos_learned-10M-v2-init2-ckpnt-2000 \
               gpt2-neox-pos_learned-100M-v2-init2-ckpnt-17500 \
               gpt2-neox-pos_learned-1B-v2-init2-ckpnt-107500 ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

#Futrell2018-stories_encoding
#Futrell2018-sentences_encoding

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
