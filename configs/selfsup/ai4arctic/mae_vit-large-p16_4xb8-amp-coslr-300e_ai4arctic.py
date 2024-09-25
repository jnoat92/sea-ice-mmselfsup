_base_ = 'mae_vit-base-p16_4xb8-amp-coslr-300e_ai4arctic.py'

# model settings
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='l'),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1024))
