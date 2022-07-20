#!/bin/bash
#SBATCH --job-name=MISTRAL
#SBATCH --array=0-4
#SBATCH --time=6-23:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:RTXA6000:4
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH -p evlab

i=0
overwrite=false

#activity_id_list="Fedorenko2016.ecog"
activity_id_list="naturalistic_stories"
activity_arr=($activity_id_list)

for benchmark in wikitext-2 ; do
  for model in mistral-caprica-gpt2-small-x81  ; do
      for checkpoint in 40 400 4000 40000 400000; do
            model_list[$i]="${model}-ckpnt-${checkpoint}"
            benchmark_list[$i]="$benchmark"
            activity_list[$i]="${activity_arr[$idx]}"
            i=$[$i+1]
      done
    done
done


#Futrell2018-encoding
#Blank2014fROI-encoding
#wikitext-2

RESULTCACHING_HOME=/om5/group/evlab/u/ehoseini/.result_caching
export RESULTCACHING_HOME
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running benchmark ${benchmark_list[$SLURM_ARRAY_TASK_ID]}"

#if [ $overwrite ]
#then
#  #original='mistral/'
#  #correction='mistral_'
#  #model_fix="${x/$original/$correction}"
#
#  #x=$model_fix
#  #original='/ckpt'
#  #correction='_ckpt'
#  #model_name="${x/$original/$correction}"
#
#  activity_name=${activity_list[$SLURM_ARRAY_TASK_ID]}
#  model_name=${model_list[$SLURM_ARRAY_TASK_ID]}
#  ACT_DIR="${RESULTCACHING_HOME}/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/"
#  act_name="identifier=${model_name},stimuli_identifier=${activity_name}*"
#  echo "searching for ${act_name}"
#  find $ACT_DIR -type f -iname $act_name -printf x | wc -c
#  find $ACT_DIR -type f -iname $act_name -exec rm -rf {} \;
#
#  SCORE_DIR="${RESULTCACHING_HOME}/neural_nlp.score/"
#  score_name="benchmark=${benchmark_list[$SLURM_ARRAY_TASK_ID]},model=${model_list[$SLURM_ARRAY_TASK_ID]}*"
#  echo "searching for ${score_name}"
#  find $ACT_DIR -type f -iname $act_name -printf x | wc -c
#  find $SCORE_DIR -type f -iname $score_name -exec rm -rf {} \;
#  echo " removed prior data "
#fi

. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022

/om/user/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/user/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"