#!/bin/bash
set -e
array=(
# configs/selfsup/ai4arctic/pretrain_50/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50.py       # 6:30
# configs/selfsup/ai4arctic/pretrain_50/mae_vit-large-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50.py
configs/selfsup/ai4arctic/pretrain_50/mae_vit-huge-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50.py

# configs/selfsup/ai4arctic/pretrain_50/mae_vit-base-p16_4xb8-amp-coslr-100ki_ai4arctic_pt50.py
)

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   # sbatch pretrain.sh $i ${array[i]}
   sbatch pretrain.sh ${array[i]} $i
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 2
done
