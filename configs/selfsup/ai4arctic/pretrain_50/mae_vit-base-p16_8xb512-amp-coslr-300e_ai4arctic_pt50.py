_base_ = 'mae_vit-base-p16_8xb512-amp-coslr-400e_ai4arctic_pt50.py'

# pre-train for 300 epochs
max_epochs = 300
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
        T_max=260,
        by_epoch=True,
        begin=40,
        end=max_epochs,
        convert_to_iter_based=True)
]

vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         group= "max_epochs_{:04d}.pkl".format(max_epochs),
                        )
                    )
                ]
