_base_ = 'mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt20.py'

# model settings
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='l', out_indices=[5, 11, 17, 23]),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1024))

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')
