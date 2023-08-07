#!/bin/bash
#SBATCH --job-name=download_
#SBATCH --array=0-12
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

#!/bin/bash
i=0
ROOT_DIR=/om/weka/evlab/ehoseini/MyData/mistral/

CHECKPOINT_FILE="${ROOT_DIR}/mistral_checkpoints.txt"
rm -f $CHECKPOINT_FILE
touch $CHECKPOINT_FILE
printf "%s,%s,%s,%s\n" "row" "model_id" "checkpoint_id" "save_dir"    >> $CHECKPOINT_FILE

for model_id in expanse-gpt2-small-x777 alias-gpt2-small-x21 ; do
    for checkpoint in 0 20 200 2000 20000 200000 ; do
            model_id_list[$i]="$model_id"
            checkpoint_list[$i]="$checkpoint"
            save_loc="${ROOT_DIR}/${model_id}/ckpt_${checkpoint}"
            saving_list[$i]=$save_loc
            i=$[$i +1]
            printf "%d,%s,%s,%s\n" "$i" "$model_id" "$checkpoint" "$save_loc"  >> $CHECKPOINT_FILE
    done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_id_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running checkpoint ${checkpoint_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running saving directory ${saving_list[$SLURM_ARRAY_TASK_ID]}"

module load openmind/git-lfs/2.13.2

  git clone "https://huggingface.co/stanford-crfm/${model_id_list[$SLURM_ARRAY_TASK_ID]}" --branch "checkpoint-${checkpoint_list[$SLURM_ARRAY_TASK_ID]}" --single-branch "${saving_list[$SLURM_ARRAY_TASK_ID]}"
  cd "${saving_list[$SLURM_ARRAY_TASK_ID]}"
  git lfs pull

#copy config file from the main branch in to the checkpoint folder ( not needed anymore )
Model_config_file="${ROOT_DIR}/${model_id_list[$SLURM_ARRAY_TASK_ID]}/config.json"
cp "${Model_config_file}" "${saving_list[$SLURM_ARRAY_TASK_ID]}"

. ~/.bash_profile
conda activate base

python fix_pytorch_version.py "${saving_list[$SLURM_ARRAY_TASK_ID]}"

# save model file for pytorch < 1.6.0
