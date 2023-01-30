#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-1
#SBATCH --time=56:00:00
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  LangLocECoGv2-encoding ; do
    benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done

#Blank2014fROI-encoding
module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
echo "cache id " $RESULTCACHING_HOME


. /om/user/ehoseini/.bash_profile
. /om/user/ehoseini/.bashrc
conda activate neural_nlp_2022

which python
# run compute_benchamrk_ceiling.py
/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural_nlp_bench/compute_benchamrk_ceiling.py --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
