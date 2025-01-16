_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py'
]

# ============== DATASET ==============
import numpy as np

crop_size = (384, 384)

downsample_factor_train = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # List all downsampling factors from 2X to 10X to include during training


possible_channels = ['nersc_sar_primary', 'nersc_sar_secondary', 
                    'distance_map', 
                    'btemp_6_9h', 'btemp_6_9v', 'btemp_7_3h', 'btemp_7_3v', 'btemp_10_7h', 'btemp_10_7v', 'btemp_18_7h',
                    'btemp_18_7v', 'btemp_23_8h', 'btemp_23_8v', 'btemp_36_5h', 'btemp_36_5v', 'btemp_89_0h', 'btemp_89_0v',
                    'u10m_rotated', 'v10m_rotated', 't2m', 'skt', 'tcwv', 'tclw', 
                    'sar_grid_incidenceangle', 
                    'sar_grid_latitude', 'sar_grid_longitude', 'month', 'day']

# dataset settings
dataset_type_train = 'AI4ArcticPatches'
dataset_type_val = 'AI4Arctic'

data_root_train_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
data_root_test_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3'
data_root_patches = '/home/jnoat92/scratch/dataset/ai4arctic/'

gt_root_test = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3_segmaps'

# file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/pretrain_20.txt'
# file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/pretrain_40.txt'
# file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/pretrain_60.txt'
file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/pretrain_80.txt'

# load normalization params
global_meanstd = np.load('/'.join([data_root_train_nc, 'global_meanstd.npy']), allow_pickle=True).item()
mean, std = {}, {}
for i in possible_channels:
    ch = i if i != 'sar_grid_incidenceangle' else 'sar_incidenceangle'
    if ch not in global_meanstd.keys(): continue
    mean[i] = global_meanstd[ch]['mean']
    std[i]  = global_meanstd[ch]['std']

mean['sar_grid_latitude'] = 69.14857395508363;   std['sar_grid_latitude']  = 7.023603113019076
mean['sar_grid_longitude']= -56.351130746236606; std['sar_grid_longitude'] = 31.263271402859893
mean['month'] = 6; std['month']  = 3.245930125274979
mean['day'] = 182; std['day']  = 99.55635507719892


# channels to use
channels = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',

    # -- incidence angle -- #
    'sar_grid_incidenceangle',

    # -- Geographical variables -- #
    'sar_grid_latitude',
    'sar_grid_longitude',
    'distance_map',

    # # -- AMSR2 channels -- #
    'btemp_6_9h', 'btemp_6_9v',
    'btemp_7_3h', 'btemp_7_3v',
    'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    'btemp_89_0h', 'btemp_89_0v',

    # # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m', 'skt', 'tcwv', 'tclw',

    # -- acquisition time
    'month', 'day'
]


train_pipeline = [
    dict(type='LoadPatchFromPKLFile', channels=channels, mean=mean, std=std, 
        to_float32=True, nan=255, with_seg=False),
    dict(
        type='RandomResizedCrop',   # copy from ../_base_/datasets/imagenet_mae.py
        size=crop_size,
        scale=(0.9, 1.0),
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

concat_dataset = dict(type='ConcatDataset', 
                    datasets= [dict(type=dataset_type_train,
                                    data_root = '/'.join([data_root_patches, 'down_scale_%dX'%(i)]),
                                    ann_file = file_train,
                                    pipeline = train_pipeline) for i in downsample_factor_train])

train_dataloader = dict(batch_size=8,
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='WeightedInfiniteSampler', use_weights=True),
                        collate_fn=dict(type='default_collate'),
                        dataset=concat_dataset)


# ============== MODEL ==============
patch_size = 16
model = dict(
    type='MAE_CCH',
    data_preprocessor=dict(mean=None, std=None, bgr_to_rgb=False),
    backbone=dict(type='MAEViT_CCH', 
                  arch='b', 
                  img_size=crop_size,
                  patch_size=patch_size, 
                  in_channels=len(channels),
                  mask_ratio=0.75, 
                  out_indices=[2, 5, 8, 11],
                  drop_path_rate=0.1
                ),
    neck=dict(type='MAEPretrainDecoder_custom', 
              num_patches=(crop_size[0]//patch_size)**2, 
              in_chans=len(channels),
              attn_drop_rate=0.3),
    head=dict(
        type='MAEPretrainHeadNonFixInChannels',
        norm_pix=True,
        patch_size=16,
        channels=len(channels),
        loss=dict(type='MAEReconstructionLoss')),
)


# ============== SCHEDULE ==============
# optimizer wrapper
# Using 4 GPUs
optimizer = dict(
    type='AdamW', lr=1e-4 * 4, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# runtime settings
n_iterations = 50000
train_cfg = dict(type='IterBasedTrainLoop', 
                 max_iters=n_iterations,
                 _delete_=True)

# learning rate scheduler
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


# ============== RUNTIME ==============

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', interval=5000, max_keep_ckpts=10, by_epoch=False),
)

wandb_config = dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         entity='jnoat92',
                         project='MAE-selfsup',
                         group="pt_{:03d}".format(int(file_train.split('/')[-1].split('.')[0].split('_')[1])),
                         name='{{fileBasenameNoExtension}}',),
                     #  name='filename',),
                     define_metric_cfg=None,
                     commit=True,
                     log_code_name=None,
                     watch_kwargs=None)

vis_backends = [wandb_config,
                dict(type='LocalVisBackend')]

visualizer = dict(type='SelfSupVisualizer', 
                  vis_backends=vis_backends, 
                  name='visualizer')

# ============== CUSTOM IMPORTS ==============

custom_imports = dict(
    imports=['mmselfsup.datasets.ai4arctic_patches',
            'mmselfsup.datasets.transforms.loading_ai4arctic_patches',
            'mmselfsup.datasets.samplers.ai4arctic_multires_sampler',
            'mmselfsup.models.algorithms.mae_ai4arctic',
            'mmselfsup.models.backbones.mae_vit_custom_ichannels',
            'mmselfsup.models.necks.mae_neck_custom',
            'mmselfsup.models.heads.mae_head_non_fix_in_channels',
            ],
    allow_failed_imports=False)


# randomness

randomness = dict(seed=0, diff_rank_seed=True)
