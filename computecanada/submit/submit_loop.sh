#!/bin/bash
set -e
array=(
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-400e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-800e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-1600e_ai4arctic.py

configs/selfsup/ai4arctic/mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-large-p16_8xb512-amp-coslr-400e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-large-p16_8xb512-amp-coslr-800e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-large-p16_8xb512-amp-coslr-1600e_ai4arctic.py

configs/selfsup/ai4arctic/mae_vit-huge-p16_8xb512-amp-coslr-1600e_ai4arctic.py
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
