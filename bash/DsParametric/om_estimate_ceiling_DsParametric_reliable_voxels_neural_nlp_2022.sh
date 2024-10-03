#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
# define num_subsamples as 200
num_subsamples=150
# define number of bootstrap samples as 100
num_bootstrap_samples=200


i=0
for benchmark in DsParametricfMRI-first-all-min-Encoding_sep2024 \
                DsParametricfMRI-first-all-rand-Encoding_sep2024 \
                DsParametricfMRI-first-all-max-Encoding_sep2024  ; do
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


. /om/weka/evlab/ehoseini/.bash_profile
. /om/weka/evlab/ehoseini/.bashrc
conda activate neural_nlp_2022

which python


/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /weka/scratch/weka/evlab/ehoseini/neural-nlp-2022/compute_ceiling_for_DsParametric_benchmark.py --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --num_subsamples $num_subsamples --num_bootstraps $num_bootstrap_samples
