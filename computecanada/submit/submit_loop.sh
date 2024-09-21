#!/bin/bash
set -e
array=(
configs/selfsup/ai4arctic/pretrain_50/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_pt50.py
configs/selfsup/ai4arctic/pretrain_50/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic_pt50.py
configs/selfsup/ai4arctic/pretrain_50/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic_pt50.py

configs/selfsup/ai4arctic/pretrain_70/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_pt70.py
configs/selfsup/ai4arctic/pretrain_70/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic_pt70.py
configs/selfsup/ai4arctic/pretrain_70/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic_pt70.py

configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_pt80.py
configs/selfsup/ai4arctic/pretrain_80/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic_pt80.py
configs/selfsup/ai4arctic/pretrain_80/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic_pt80.py

configs/selfsup/ai4arctic/pretrain_90/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_pt90.py
configs/selfsup/ai4arctic/pretrain_90/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic_pt90.py
configs/selfsup/ai4arctic/pretrain_90/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic_pt90.py

configs/selfsup/ai4arctic/pretrain_95/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_pt95.py
configs/selfsup/ai4arctic/pretrain_95/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic_pt95.py
configs/selfsup/ai4arctic/pretrain_95/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic_pt95.py
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
