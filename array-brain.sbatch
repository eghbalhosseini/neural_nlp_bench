#!/bin/bash

#SBATCH --job-name=nnlp-neural-array
#SBATCH --array=0-90
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=267G

i=0
for trained in "" "-untrained" ; do
for benchmark in Blank2014fROI-encoding ; do
for model in sentence-length ETM word2vec skip-thoughts glove lm_1b \
  transformer \
  bert-base-uncased bert-base-multilingual-cased bert-large-uncased bert-large-uncased-whole-word-masking \
  openaigpt gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 \
  transfo-xl-wt103 \
  xlnet-base-cased xlnet-large-cased xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 xlm-clm-enfr-1024 xlm-mlm-100-1280 \
  roberta-base roberta-large distilroberta-base \
  distilbert-base-uncased \
  ctrl \
  albert-base-v1 albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 \
  t5-small t5-base t5-large t5-3b t5-11b \
  xlm-roberta-base xlm-roberta-large ; do
  model_list[$i]="$model$trained"
  benchmark_list[$i]="$benchmark"
  i=$[$i +1]
done
done
done

#echo ${#model_list[@]}
#exit

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

CUDA_VISIBLE_DEVICES= RESULTCACHING_HOME=/braintree/home/msch/.result_caching PYTHONPATH=.:../result_caching/:../brain-score:../brainio_collection:../brainio_base:/braintree/home/msch/miniconda3/envs/neural-nlp/lib/python3.7/site-packages/ /braintree/home/msch/miniconda3/envs/neural-nlp/bin/python neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --log_level DEBUG

