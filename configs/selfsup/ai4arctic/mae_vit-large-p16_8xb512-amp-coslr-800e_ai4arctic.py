_base_ = 'mae_vit-base-p16_8xb512-amp-coslr-800e_ai4arctic.py'

# model settings
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='l'),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1024))
