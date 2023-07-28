#!/bin/bash
#SBATCH -c 8
#SBATCH --exclude node[017-018]
#SBATCH -t 5:00:00

GRAND_FILE=$1
OVERWRITE='false' # or 'true'
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2       # Taking the task ID as an input parameter.
fi
echo "${GRAND_FILE}"
echo $JID

while IFS=, read -r line_count ckpoint_name ckpont_dir ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"
      run_ckpoint_name=$ckpoint_name
      run_ckpont_dir=$ckpont_dir
      do_run=true
      break
    else
      do_run=false
  fi
done <"${GRAND_FILE}"
if $do_run; then
  echo "run_ckpoint_name:${run_ckpoint_name}"
  echo "run_ckpont_dir:${run_ckpont_dir}"
  . /home/ehoseini/.bash_profile
  conda activate neural_nlp_2022
  echo $(which python)
  python /om/weka/evlab/ehoseini/neural-nlp-2022/neural_nlp/models/gpt_neox_model/convert_ckpt_to_gpt_neox.py --checkpoint_dir "$run_ckpont_dir" --hf_save_dir "$run_ckpont_dir"

fi