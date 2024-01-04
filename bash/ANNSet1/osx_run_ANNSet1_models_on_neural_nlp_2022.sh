#!/bin/bash

neural_nlp_bench() {
    model="$1"
    benchmark="$2"

    # Set a memory limit for this process (soft limit)
    ulimit -v 10G # 10GB

    # Your code here
    # Assuming you have Python and Conda installed locally with the necessary environment
    # Activate your Conda environment (modify the environment name if needed)
    conda activate neural_align
    # echo python
    which python

    # Replace the path with the correct path to your Python script
    /Users/eghbalhosseini/miniconda3/envs/neural_align/bin/python /Users/eghbalhosseini/MyCodes/neural-nlp-2022/neural_nlp run --model "$model" --benchmark "$benchmark"
}

# Export the function so it's available to parallel
export -f neural_nlp_bench

# Define an array of model and benchmark combinations
# Each entry in the format "model_name benchmark"
models=(
    #"xlnet-large-cased"
    "xlm-mlm-en-2048"
    "albert-xxlarge-v2"
    "bert-large-uncased-whole-word-masking"
    "roberta-base"
    "gpt2-xl"
    "ctrl"
)

  tasks=(
    "ANNSet1ECoG-encoding"
    # Add more tasks as needed
)

# Run the tasks in parallel with 4 processes
parallel -j 4 neural_nlp_bench ::: "${models[@]}" ::: "${tasks[@]}" ::: --joblog joblog.txt --linebuffer