#!/bin/bash

#SBATCH --job-name=MGH
#SBATCH --array=0-27
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0

for benchmark in LangLocECoG-bip-band-Encoding LangLocECoG-uni-band-Encoding \
                  ANNSet1ECoG-bip-band-Encoding ANNSet1ECoG-uni-band-Encoding ; do

    for model in  roberta-base \
                  bert-large-uncased-whole-word-masking \
                  xlnet-large-cased \
                  xlm-mlm-en-2048 \
                  albert-xxlarge-v2 \
                  gpt2-xl \
                  ctrl  ; do
           model_list[$i]="$model"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om2/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"


