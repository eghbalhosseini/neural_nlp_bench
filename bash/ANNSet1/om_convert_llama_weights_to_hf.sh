#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --partition=evlab

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/rdma/vast-rdma/vast/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py


i=0
for model_size in 7B 13B 30B 65B ; do
           model_list[$i]="$model_size"
            i=$[$i +1]
    done


python /rdma/vast-rdma/vast/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir /nese/mit/group/evlab/u/Shared/LLAMA/ --model_size "${model_list[$SLURM_ARRAY_TASK_ID]}" --output_dir "/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/${model_list[$SLURM_ARRAY_TASK_ID]}"

