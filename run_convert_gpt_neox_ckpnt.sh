#!/bin/bash
ROOT_DIR='/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_1b_v2/gpt2/checkpoints_4/'

ckpnt_conv_='ckpnt_conv_list'
i=0
LINE_COUNT=0
CKPNT_FILE="${ROOT_DIR}/${ckpnt_conv_}.txt"
rm -f $CKPNT_FILE
touch $CKPNT_FILE
printf "%s,%s,%s\n" "row" "ckpnt_name" "location"   >> $CKPNT_FILE

echo "looking at ${ROOT_DIR} "
SUBJ_LINE=0
while read x; do
      # check if file already exist in labels dir
      original=$ROOT_DIR
      correction=''
      ckpont_name="${x/$original/$correction}"
      ckpont_file="${x}/pytorch_model.bin"
      if [ ! -f "$ckpont_file" ]
      then
        LINE_COUNT=$(expr ${LINE_COUNT} + 1)
        # folder to find the file
        printf "%d,%s,%s\n" "$LINE_COUNT" "$ckpont_name" "$x" >> $CKPNT_FILE
      fi
done < <(find $ROOT_DIR -maxdepth 1 -type d -name "global*")

run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
  echo "running  ${LINE_COUNT} jobs"

   #nohup /cm/shared/admin/bin/submit-many-jobs 3 2 3 1 convert_gpt_neox_ckpoint.sh  $CKPNT_FILE
   nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 75 100 25 convert_gpt_neox_ckpnt.sh  $CKPNT_FILE &
  else
    echo $LINE_COUNT
fi