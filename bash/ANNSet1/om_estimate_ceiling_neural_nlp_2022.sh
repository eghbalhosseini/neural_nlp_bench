#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --array=0-25
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


# define num_subsamples as 200
num_subsamples=500
# define number of bootstrap samples as 100
num_bootstrap_samples=100

i=0
for benchmark in   LangLocECoG-bip-gaus-strict-Encoding \
                   LangLocECoG-uni-gaus-strict-Encoding \
                   LangLocECoG-bip-gaus-zs-strict-Encoding \
                   LangLocECoG-uni-gaus-zs-strict-Encoding \
                   LangLocECoG-bip-band-strict-Encoding \
                   LangLocECoG-uni-band-strict-Encoding \
                   LangLocECoG-bip-gaus-shared-ANN-strict-Encoding \
                   LangLocECoG-uni-gaus-shared-ANN-strict-Encoding \
                   LangLocECoG-bip-band-shared-ANN-strict-Encoding \
                   LangLocECoG-uni-band-shared-ANN-strict-Encoding \
                   ANNSet1ECoG-bip-gaus-Encoding \
                   ANNSet1ECoG-bip-gaus-strict-Encoding \
                   ANNSet1ECoG-bip-band-Encoding \
                   ANNSet1ECoG-bip-band-strict-Encoding \
                   ANNSet1ECoG-uni-gaus-Encoding \
                   ANNSet1ECoG-uni-gaus-strict-Encoding \
                   ANNSet1ECoG-uni-band-Encoding \
                   ANNSet1ECoG-uni-band-strict-Encoding \
                   ANNSet1ECoG-bip-gaus-shared-LangLoc-Encoding \
                   ANNSet1ECoG-bip-gaus-shared-LangLoc-strict-Encoding \
                   ANNSet1ECoG-bip-band-shared-LangLoc-Encoding \
                   ANNSet1ECoG-bip-band-shared-LangLoc-strict-Encoding \
                   ANNSet1ECoG-uni-gaus-shared-LangLoc-Encoding \
                   ANNSet1ECoG-uni-gaus-shared-LangLoc-strict-Encoding \
                   ANNSet1ECoG-uni-band-shared-LangLoc-Encoding \
                   ANNSet1ECoG-uni-band-shared-LangLoc-strict-Encoding ; do
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


/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om2/user/ehoseini/neural-nlp-2022/compute_ceiling_for_benchmark.py --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}" --num_subsamples $num_subsamples --num_bootstraps $num_bootstrap_samples
