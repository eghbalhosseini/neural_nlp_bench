#!/bin/bash
#SBATCH --job-name=MISTRAL
#SBATCH --array=0-4
#SBATCH --time=6-23:00:00
#SBATCH --mem=40G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
overwrite=true

for benchmark in Fedorenko2016v3-encoding  ; do
  for model in mistral/caprica-gpt2-small-x81  ; do
      for checkpoint in 40 400 4000 40000 400000; do
            model_list[$i]="${model}/ckpt_${checkpoint}"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
      done
    done
done

#Futrell2018-encoding

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"


if [ $overwrite ]
then
  x=${benchmark_list[$SLURM_ARRAY_TASK_ID]}
  original='-encoding'
  correction=''
  activity_name="${x/$original/$correction}"

  x=${model_list[$SLURM_ARRAY_TASK_ID]}
  original='/ckpt'
  correction='_ckpt'
  model_name="${x/$original/$correction}"

  ACT_DIR="${RESULTCACHING_HOME}/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/"
  act_name="identifier=${model_name},stimuli_identifier=${activity_name}*"

  echo "searching for ${act_name}"
  find $ACT_DIR -type f -iname $act_name -printf x | wc -c
  find $ACT_DIR -type f -iname $act_name -exec rm -rf {} \;
  SCORE_DIR="${RESULTCACHING_HOME}/neural_nlp.score/"
  score_name="benchmark=${benchmark_list[$SLURM_ARRAY_TASK_ID]},model=${model_list[$SLURM_ARRAY_TASK_ID]}*"
  find $ACT_DIR -type f -iname $act_name -printf x | wc -c
  find $SCORE_DIR -type f -iname $score_name -exec rm -rf {} \;
  echo " removed prior data "
fi

# Blank2014fROI-encoding


#. ~/.bash_profile
#. ~/.bashrc
#conda activate neural_nlp_2022
#
#/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"