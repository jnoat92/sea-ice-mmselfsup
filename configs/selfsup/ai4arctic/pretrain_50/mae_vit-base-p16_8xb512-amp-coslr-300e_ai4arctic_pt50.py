_base_ = '../mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py'

file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/pretrain_50.txt'

vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         name='{{fileBasenameNoExtension}}',
                        )
                    )
                ]
