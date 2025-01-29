_base_ = 'mae_vit-base-p16_4xb8-coslr-50ki_ai4arctic_pt40.py'

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')

# mixed precision
optim_wrapper = dict(type='CustomAmpOptimWrapper', loss_scale='dynamic')
custom_imports = _base_.custom_imports
custom_imports['imports'].append('mmselfsup.engine.optimizers.custom_amp_optimizar_wrapper')
