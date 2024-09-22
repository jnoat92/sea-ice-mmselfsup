_base_ = 'mae_vit-large-p16_8xb512-amp-coslr-300e_ai4arctic.py'

# pre-train for 100 epochs
max_epochs = 1600
train_cfg = dict(max_epochs=max_epochs)

# model settings
model = dict(
    backbone=dict(type='MAEViT_CCH', arch='h'),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1280))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=1560,
        by_epoch=True,
        begin=40,
        end=max_epochs,
        convert_to_iter_based=True)
]

vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         group= "max_epochs_{:04d}".format(max_epochs),
                        )
                    )
                ]
