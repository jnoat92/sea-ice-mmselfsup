_base_ = '../mae_vit-base-p16_4xb8-coslr-30ki_ai4arctic_pt80.py'

# ============== DATASET ==============
train_dataloader = dict(batch_size=32, num_workers=16)

# ============== MODEL ==============
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='h', out_indices=[7, 15, 23, 31]),
    neck=dict(type='MAEPretrainDecoder_custom', embed_dim=1280))

# ============== SCHEDULE ==============
# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# ============== RUNTIME ==============
wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')
