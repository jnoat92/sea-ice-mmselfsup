_base_ = 'mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py'

mask_ratio = 0.65
model = dict(backbone=dict(mask_ratio=mask_ratio))

# ============== DATASET ==============
train_dataloader = dict(batch_size=16, num_workers=8)   # use 4 GPUs

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = _base_.visualizer
visualizer.vis_backends = vis_backends
