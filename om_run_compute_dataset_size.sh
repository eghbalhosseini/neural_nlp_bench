#!/bin/bash
#SBATCH --job-name=CNT
#SBATCH --array=0
#SBATCH --time=6-23:00:00
#SBATCH --mem=180G
#SBATCH -c 16
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/compute_dataset_size.py
