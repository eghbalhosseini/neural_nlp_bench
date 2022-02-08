#!/bin/bash
#SBATCH --job-name=download_
#SBATCH --array=0
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


i=0
ROOT_DIR=/om/user/ehoseini/MyData/miniBERTa_training/

CHECKPOINT_FILE="${ROOT_DIR}/minberta_checkpoints.txt"
rm -f $CHECKPOINT_FILE
touch $CHECKPOINT_FILE
printf "%s,%s,%s,%s\n"  "checkpoint_id" "checkpoint" "src_dir" "target_dir"   >> $CHECKPOINT_FILE

for checkpoint in miniBERTa_1b_v2/gpt2/checkpoints_4/global_step92500 ; do
  #miniBERTa_100m_v2/gpt2/checkpoints_4/global_step20000 ; do
    checkpoint_list[$i]="$checkpoint"
    srt_loc="${ROOT_DIR}/${checkpoint}"
    target_loc="${ROOT_DIR}/${checkpoint}_torch_1_5"
    src_list[$i]=$srt_loc
    target_list[$i]=$target_loc
    i=$[$i +1]
    printf "%d,%s,%s,%s\n" "$i" "$checkpoint" "$src_loc" "$target_loc"  >> $CHECKPOINT_FILE
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running checkpoint ${checkpoint_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running saving directory ${src_list[$SLURM_ARRAY_TASK_ID]}"




#. ~/.bash_profile
#. ~/.bashrc
#conda activate base
#
#python fix_pytorch_version.py --ckpt_dir "${src_list[$SLURM_ARRAY_TASK_ID]}" --ckpt_type "gpt_neox"

echo "Running target directory ${target_loc[$SLURM_ARRAY_TASK_ID]}"
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_nlp_master_fz.simg python /om/user/ehoseini/neural-nlp-master/neural-nlp/neural_nlp/models/gpt_neox_model/convert_ckpt_to_hf.py --checkpoint_dir "${target_list[$SLURM_ARRAY_TASK_ID]}"

# save model file for pytorch < 1.6.0
