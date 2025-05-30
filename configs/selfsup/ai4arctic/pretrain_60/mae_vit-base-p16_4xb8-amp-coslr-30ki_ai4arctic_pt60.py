_base_ = 'mae_vit-base-p16_4xb8-coslr-30ki_ai4arctic_pt60.py'

# ============== DATASET ==============
train_dataloader = dict(batch_size=64, num_workers=16)

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')
