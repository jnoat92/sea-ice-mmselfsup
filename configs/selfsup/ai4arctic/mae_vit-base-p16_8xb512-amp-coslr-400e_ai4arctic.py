_base_ = 'mae_vit-base-p16_8xb512-coslr-400e_ai4arctic.py'

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
