#!/bin/bash
#SBATCH --job-name=MIS
#SBATCH --array=0-2
#SBATCH --time=6-23:00:00
#SBATCH --mem=40G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
overwrite=false
for benchmark in Futrell2018-stories_encoding Futrell2018-sentences_encoding ; do
    for model in mistral-caprica-gpt2-small-x81 ; do
      for checkpoint in `seq 0 1 0` ; do
            model_list[$i]="${model}-ckpnt-${checkpoint}-untrained"
            benchmark_list[$i]="$benchmark"
            i=$[$i+1]
      done
    done
done


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
  ACT_DIR="${RESULTCACHING_HOME}/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored/"
  act_name="identifier=${model_list[$SLURM_ARRAY_TASK_ID]},*"

  find $ACT_DIR -type f -iname $act_name -printf x | wc -c

  find $ACT_DIR -type f -iname $act_name -exec rm -rf {} \;

  SCORE_DIR="${RESULTCACHING_HOME}/neural_nlp.score/"
  score_name="benchmark=${benchmark_list[$SLURM_ARRAY_TASK_ID]},model=${model_list[$SLURM_ARRAY_TASK_ID]}*"

  find $ACT_DIR -type f -iname $act_name -printf x | wc -c

  find $SCORE_DIR -type f -iname $score_name -exec rm -rf {} \;

  echo " removed prior data "
fi


. ~/.bash_profile
. ~/.bashrc
conda activate neural_nlp_2022
/om/weka/evlab/ehoseini/miniconda3/envs/neural_nlp_2022/bin/python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp run --model "${model_list[$SLURM_ARRAY_TASK_ID]}" --benchmark "${benchmark_list[$SLURM_ARRAY_TASK_ID]}"
