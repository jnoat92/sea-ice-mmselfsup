_base_ = 'mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80.py'

# model settings
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='h', out_indices=[7, 15, 23, 31]),
    neck=dict(type='MAEPretrainDecoder_custom', embed_dim=1280))

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')
