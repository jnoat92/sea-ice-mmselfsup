_base_ = 'mae_vit-base-p16_4xb8-amp-coslr-400e_ai4arctic.py'

# pre-train for 800 epochs
max_epochs = 800
train_cfg = dict(max_epochs=max_epochs)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=760,
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
