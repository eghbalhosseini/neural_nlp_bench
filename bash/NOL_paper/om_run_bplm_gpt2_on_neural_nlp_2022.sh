#!/bin/bash
#SBATCH --job-name=nlp_bplm
#SBATCH --array=0-23
#SBATCH --time=6-23:00:00
#SBATCH --mem=60G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  Pereira2018-encoding Futrell2018-encoding Blank2014fROI-encoding ; do
  for model in bplm-gpt2-gauss-init-ckpnt-0 \
               bplm-gpt2-gauss-init-ckpnt-43750 \
               bplm-gpt2-gauss-init-1-ckpnt-0 \
               bplm-gpt2-gauss-init-1-ckpnt-41000 \
               bplm-gpt2-gauss-init-1-ckpnt-44500 \
               bplm-gpt2-hf-init-ckpnt-0 \
               bplm-gpt2-hf-init-ckpnt-16750 \
               bplm-gpt2-hf-init-ckpnt-39000 ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

#Futrell2018-stories_encoding
#Futrell2018-sentences_encoding

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
