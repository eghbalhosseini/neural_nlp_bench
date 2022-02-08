#!/bin/bash
#SBATCH --job-name=download_
#SBATCH --array=0-40
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

#!/bin/bash
/om/user/ehoseini/MyData/miniBERTa_training/
i=0
ROOT_DIR=/om/user/ehoseini/MyData/miniBERTa_training/

CHECKPOINT_FILE="${ROOT_DIR}/minberta_checkpoints.txt"
rm -f $CHECKPOINT_FILE
touch $CHECKPOINT_FILE
printf "%s,%s,%s\n"  "checkpoint_id" "checkpoint" "save_dir"   >> $CHECKPOINT_FILE

for checkpoint in miniBERTa_100m_v2/gpt2/checkpoints_4/global_step20000 \
                  miniBERTa_1b_v2/gpt2/checkpoints_4/global_step92500 ; do
    checkpoint_list[$i]="$checkpoint"
    save_loc="${ROOT_DIR}/${checkpoint}"
    saving_list[$i]=$save_loc
    i=$[$i +1]
    printf "%d,%s,%s\n" "$i" "$model_id" "$checkpoint" "$save_loc"  >> $CHECKPOINT_FILE
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running checkpoint ${checkpoint_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running saving directory ${saving_list[$SLURM_ARRAY_TASK_ID]}"




. ~/.bash_profile
. ~/.bashrc
conda activate base

python fix_pytorch_version.py --ckpt_dir "${saving_list[$SLURM_ARRAY_TASK_ID]}" --ckpt_type "gpt_neox"

# save model file for pytorch < 1.6.0
