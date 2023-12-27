#!/bin/bash

i=0
for trained in "" "-untrained" ; do
for benchmark in Fedorenko2016v3-encoding-weights ; do
for model in gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 \
    bert-base-uncased bert-large-uncased \
    xlnet-base-cased xlnet-large-cased xlm-mlm-en-2048 xlm-mlm-enfr-1024 xlm-mlm-xnli15-1024 xlm-clm-enfr-1024 xlm-mlm-100-1280 \
    roberta-base roberta-large distilroberta-base \
    distilbert-base-uncased \
    ctrl \
    albert-base-v1 albert-base-v2 albert-large-v1 albert-large-v2 albert-xlarge-v1 albert-xlarge-v2 albert-xxlarge-v1 albert-xxlarge-v2 \
    xlm-roberta-base xlm-roberta-large ; do
  model_list[$i]="$model$trained"
  benchmark_list[$i]="$benchmark"
  python /Users/eghbalhosseini/MyCodes/neural-nlp/neural_nlp run --model "${model_list[$i]}" --benchmark "${benchmark_list[$i]}"
  i=$[$i +1]
done
done
done


