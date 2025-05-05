_base_ = [
    '../pretrain_80/mae_vit-base-p16_4xb8-coslr-30ki_ai4arctic_pt80.py'
]

# ============== DATASET ==============
# Update train file
file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/pretrain_60.txt'
concat_dataset = _base_.concat_dataset
train_dataloader = _base_.train_dataloader
for i in range(len(concat_dataset.datasets)):
    concat_dataset.datasets[i].ann_file = file_train
train_dataloader.dataset = concat_dataset

# ============== RUNTIME ==============
wandb_config = _base_.wandb_config
wandb_config.init_kwargs.group = "ft_{:03d}".format(int(file_train.split('.')[0].split('_')[-1]))
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

