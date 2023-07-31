#!/bin/bash
#SBATCH --job-name=MISTRAL
#SBATCH --array=1-12
#SBATCH --time=40:00:00
#SBATCH --mem=40G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
i=1
for benchmark in  Futrell2018-encoding Futrell2018-norm-encoding ; do
  for model in mistral-caprica-gpt2-small-x81  ; do
      for checkpoint in 0 20 200 2000 20000 200000 ; do
            model_list[$i]="${model}-ckpnt-${checkpoint}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
      done
    done
done



RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
conda activate neural_nlp_2022

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"