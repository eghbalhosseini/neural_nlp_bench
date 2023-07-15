#!/bin/bash
#SBATCH --job-name=nlp2022
#SBATCH --array=0-8
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for benchmark in  Pereira2018-encoding Pereira2018-norm-encoding Pereira2018-norm-v2-encoding  Futrell2018-norm-v2-encoding Futrell2018-norm-encoding ; do
  for model in  gpt2-neox-pos_learned-1B-v3-ckpnt-310000-untrained-ln-hf \
                gpt2-untrained-ln-hf; do
                #gpt2-neox-pos_learned-10M-v3-ckpnt-2000 gpt2-neox-pos_learned-10M-v3-ckpnt-2000-untrained gpt2-neox-pos_learned-10M-v3-ckpnt-2000-untrained_hf \
                #gpt2-neox-pos_learned-100M-v3-ckpnt-14250 gpt2-neox-pos_learned-100M-v3-ckpnt-14250-untrained gpt2-neox-pos_learned-100M-v3-ckpnt-14250-untrained_hf \
                #gpt2 gpt2-untrained gpt2-untrained_hf
                  #gpt2-neox-pos_learned-1M-v3-ckpnt-1000 gpt2-neox-pos_learned-1M-v3-ckpnt-1000-untrained gpt2-neox-pos_learned-1M-v3-ckpnt-1000-untrained_hf \
                  #gpt2-neox-pos_learned-1B-v3-ckpnt-310000 gpt2-neox-pos_learned-1B-v3-ckpnt-310000-untrained gpt2-neox-pos_learned-1B-v3-ckpnt-310000-untrained_hf
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
    done
done

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
