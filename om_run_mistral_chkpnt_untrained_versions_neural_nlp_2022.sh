#!/bin/bash
#SBATCH --job-name=MISTRAL
#SBATCH --array=0-16
#SBATCH --time=23:00:00
#SBATCH --mem=60G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0
overwrite=false

#activity_id_list="Fedorenko2016.ecog"
#activity_id_list="naturalStories naturalStories naturalStories"
#activity_arr=($activity_id_list)

for benchmark in Pereira2018-v2-encoding Pereira2018-norm-encoding Futrell2018-encoding Futrell2018-norm-encoding ; do
#for benchmark in Futrell2018-encoding Futrell2018-stories_encoding Futrell2018-sentences_encoding ; do
for model in mistral-caprica-gpt2-small-x81-ckpnt-200000-untrained_hf \
              mistral-caprica-gpt2-small-x81-ckpnt-0-untrained_hf \
              mistral-caprica-gpt2-small-x81-ckpnt-0-untrained-ln-uniform \
              mistral-caprica-gpt2-small-x81-ckpnt-200000-untrained-ln-uniform ; do
            model_list[$i]="${model}"
            benchmark_list[$i]="$benchmark"
 #           activity_list[$i]="${activity_arr[$idx]}"
            i=$[$i+1]
      #done
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
#
#if [ "$overwrite" = true ]
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
#  score_name="benchmark=${benchmark_list[$SLURM_ARRAY_TASK_ID]},model=${model_list[$SLURM_ARRAY_TASK_ID]},*"
#  echo "searching for ${score_name}"
#  find $ACT_DIR -type f -iname $act_name -printf x | wc -c
#  find $SCORE_DIR -type f -iname $score_name -exec rm -rf {} \;
#  echo " removed prior data "
#fi

. ~/.bash_profile
conda activate neural_nlp_2022
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"