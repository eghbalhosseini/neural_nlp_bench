#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-91
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  ANNSet1ECoG-encoding ; do
  for model in sentence-length word-position random-embedding skip-thoughts skip-thoughts-untrained lm_1b lm_1b-untrained \
   word2vec word2vec-untrained glove glove-untrained transformer transformer-untrained ETM ETM-untrained bert-base-uncased \
    bert-base-multilingual-cased bert-large-uncased bert-large-uncased-whole-word-masking openaigpt gpt2 gpt2-medium gpt2-large \
     gpt2-xl distilgpt2 transfo-xl-wt103 xlnet-base-cased xlnet-large-cased xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 \
      xlm-clm-enfr-1024 xlm-mlm-100-1280 roberta-base roberta-large distilroberta-base distilbert-base-uncased ctrl albert-base-v1 \
       albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 t5-small \
        t5-base t5-large t5-3b t5-11b xlm-roberta-base xlm-roberta-large bert-base-uncased-untrained bert-base-multilingual-cased-untrained \
         bert-large-uncased-untrained bert-large-uncased-whole-word-masking-untrained openaigpt-untrained gpt2-untrained gpt2-medium-untrained \
          gpt2-large-untrained gpt2-xl-untrained distilgpt2-untrained transfo-xl-wt103-untrained xlnet-base-cased-untrained xlnet-large-cased-untrained \
           xlm-mlm-en-2048-untrained xlm-mlm-enfr-1024-untrained xlm-mlm-xnli15-1024-untrained xlm-clm-enfr-1024-untrained xlm-mlm-100-1280-untrained \
            roberta-base-untrained roberta-large-untrained distilroberta-base-untrained distilbert-base-uncased-untrained ctrl-untrained \
             albert-base-v1-untrained albert-base-v2-untrained albert-large-v1-untrained albert-large-v2-untrained albert-xlarge-v1-untrained \
              albert-xlarge-v2-untrained albert-xxlarge-v1-untrained albert-xxlarge-v2-untrained t5-small-untrained t5-base-untrained t5-large-untrained \
               t5-3b-untrained t5-11b-untrained xlm-roberta-base-untrained xlm-roberta-large-untrained ; do
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
