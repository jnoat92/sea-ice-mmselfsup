_base_ = 'mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50.py'

n_iterations = 100000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=n_iterations)

# # learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1/3,
        by_epoch=False,
        begin=0,
        end=n_iterations//10
        ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=n_iterations//10,
        end=n_iterations
        )
]

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')
