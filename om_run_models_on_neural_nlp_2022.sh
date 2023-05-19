#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-60
#SBATCH --time=32:00:00
#SBATCH --mem=20G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in LangLocECoG-sentence-encoding ANNSet1ECoG-Sentence-encoding ; do
  for model in roberta-base roberta-large distilroberta-base \
      xlnet-large-cased xlnet-base-cased \
      bert-base-uncased bert-base-multilingual-cased bert-large-uncased bert-large-uncased-whole-word-masking \
      xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 xlm-clm-enfr-1024 xlm-mlm-100-1280 \
      openaigpt gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 \
      albert-base-v1 albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 \
      ctrl ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

#Blank2014fROI-encoding
module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
echo "cache id " $RESULTCACHING_HOME


. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

which python

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
