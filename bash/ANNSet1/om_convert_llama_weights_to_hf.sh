#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --array=0-8
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --partition=evlab

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022



i=0
#for model_size in 7B 13B 30B 65B ; do
#           model_list[$i]="$model_size"
#            i=$[$i +1]
#    done

for model_size in 7B 7Bf 13B 13Bf 30B 34B 65B 70B 70Bf ; do
           model_list[$i]="$model_size"
            i=$[$i +1]
    done


python /rdma/vast-rdma/vast/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir /nese/mit/group/evlab/u/ehoseini/MyData/LLAMA_2/ --model_size "${model_list[$SLURM_ARRAY_TASK_ID]}" --output_dir "/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA_2_hf/${model_list[$SLURM_ARRAY_TASK_ID]}"

