#!/bin/bash
#SBATCH --job-name=1B
#SBATCH --array=0-3
#SBATCH --time=6-23:00:00
#SBATCH --mem=40G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
overwrite=true
for benchmark in Pereira2018-encoding Blank2014fROI-encoding Futrell2018-encoding ; do
  for model in gpt2-neox-pos_learned-1B-v2-ckpnt  ; do
      for checkpoint in `seq 2500 2500 2500`; do
            model_list[$i]="${model}-${checkpoint}-untrained"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
      done
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


if [ $overwrite ]
then
  x=${benchmark_list[$SLURM_ARRAY_TASK_ID]}
  original='-encoding'
  correction=''
  activity_name="${x/$original/$correction}"
  ACT_DIR="${RESULTCACHING_HOME}/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/"
  act_name="identifier=${model_list[$SLURM_ARRAY_TASK_ID]},stimuli_identifier=${activity_name}*"
  find $ACT_DIR -type f -iname $act_name -exec rm -rf {} \;

  SCORE_DIR="${RESULTCACHING_HOME}/neural_nlp.score/"
  score_name="benchmark=${benchmark_list[$SLURM_ARRAY_TASK_ID]},model=${model_list[$SLURM_ARRAY_TASK_ID]}*"
  find $SCORE_DIR -type f -iname $score_name -exec rm -rf {} \;

  echo " removed prior data "
fi


. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022
/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
