#!/bin/bash
#SBATCH --job-name=download_
#SBATCH --array=0
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural_nlp_bench/download_openwebtext.py

# save model file for pytorch < 1.6.0
