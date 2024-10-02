# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://colab.research.google.com/github/facebookresearch/mae
# /blob/main/demo/mae_visualize.ipynb
import random
from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.dataset import Compose, default_collate

from mmselfsup.apis import inference_model, init_model
from mmengine.config import Config
import time
import os

def show_image(img: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert img.shape[2] == 3
    img = np.uint8(img.detach().cpu().numpy())

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def save_images(original_img: torch.Tensor, img_masked: torch.Tensor,
                pred_img: torch.Tensor, img_paste: torch.Tensor,
                out_file: str) -> None:
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]

    plt.subplot(1, 4, 1)
    show_image(original_img, 'original')

    plt.subplot(1, 4, 2)
    show_image(img_masked, 'masked')

    plt.subplot(1, 4, 3)
    show_image(pred_img, 'reconstruction')

    plt.subplot(1, 4, 4)
    show_image(img_paste, 'reconstruction + visible')

    plt.savefig(out_file)
    print(f'Images are saved to {out_file}')


def recover_norm(img: torch.Tensor,
                 mean: np.ndarray = 0,
                 std: np.ndarray = 1):
    if mean is not None and std is not None:
        # img = torch.clip((img * std + mean) * 255, 0, 255).int()
        img = img * std + mean

    min_ = img.min(0).values.min(0).values
    max_ = img.max(0).values.max(0).values
    img = 255*(img - min_) / (max_-min_)
    return img.int()


def post_process(
    original_img: torch.Tensor,
    pred_img: torch.Tensor,
    mask: torch.Tensor,
    mean: np.ndarray = 0,
    std: np.ndarray = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # channel conversion
    original_img = torch.einsum('nchw->nhwc', original_img.cpu())
    # masked image
    img_masked = original_img * (1 - mask)
    # reconstructed image pasted with visible patches
    img_paste = original_img * (1 - mask) + pred_img * mask

    # muptiply std and add mean to each image
    original_img = recover_norm(original_img[0], mean=mean, std=std)
    img_masked = recover_norm(img_masked[0], mean=mean, std=std)

    pred_img = recover_norm(pred_img[0], mean=mean, std=std)
    img_paste = recover_norm(img_paste[0], mean=mean, std=std)

    return original_img, img_masked, pred_img, img_paste


def main():
    parser = ArgumentParser()
    # parser.add_argument('config', help='Model config file')
    # parser.add_argument('--checkpoint', help='Checkpoint file')
    # parser.add_argument('--img-path', help='Image file path')
    # parser.add_argument('--out-file', help='The output image file path')
    parser.add_argument('--config', default='/home/jnoat92/projects/def-l44xu-ab/ai4arctic/sea-ice-mmselfsup/configs/selfsup/ai4arctic/pretrain_50/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50.py', help='Model config file')
    parser.add_argument('--checkpoint', default='/home/jnoat92/projects/def-l44xu-ab/ai4arctic/sea-ice-mmselfsup/work_dirs/selfsup/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50/iter_45750.pth',help='Checkpoint file')
    parser.add_argument('--img-path', default='/home/jnoat92/scratch/dataset/ai4arctic/down_scale_9X/S1A_EW_GRDM_1SDH_20180814T120158_20180814T120258_023242_0286BE_36EF_icechart_cis_SGRDIHA_20180814T1201Z_pl_a/00007.pkl', help='Image file path')
    parser.add_argument('--out-file', default='/home/jnoat92/projects/def-l44xu-ab/ai4arctic/sea-ice-mmselfsup/work_dirs/selfsup/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt50/20180814T120158_00010', help='The output image file path')
    parser.add_argument(
        '--use-vis-pipeline',
        action='store_true',
        help='Use vis_pipeline defined in config. For some algorithms, such '
        'as SimMIM and MaskFeat, they generate mask in data pipeline, thus '
        'the visualization process applies vis_pipeline in config to obtain '
        'the mask.')
    parser.add_argument(
        '--norm-pix',
        action='store_true',
        help='MAE uses `norm_pix_loss` for optimization in pre-training, thus '
        'the visualization process also need to compute mean and std of each '
        'patch embedding while reconstructing the original images.')
    parser.add_argument(
        '--target-generator',
        action='store_true',
        help='Some algorithms use target_generator for optimization in '
        'pre-training, such as MaskFeat, thus the visualization process could '
        'turn this on to visualize the target instead of RGB image.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='The random seed for visualization')
    args = parser.parse_args()

    # Taken from train.py
    cfg = Config.fromfile(args.config)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    print('Reconstruction visualization.')

    if args.use_vis_pipeline:
        model.cfg.test_dataloader = dict(
            dataset=dict(pipeline=model.cfg.vis_pipeline))
    else:
        model.cfg.test_dataloader = dict(
            dataset=dict(pipeline=[
                dict(type='LoadPatchFromPKLFile', channels=cfg.channels, 
                     mean=cfg.mean, std=cfg.std),
                dict(type='PackSelfSupInputs', meta_keys=['img_path'])
            ]))

    # get original image
    vis_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    data = dict(img_path=args.img_path)
    data = vis_pipeline(data)
    data = default_collate([data])
    img, data_samples = model.data_preprocessor(data, False)

    # if args.norm_pix:
    #     # for MAE reconstruction
    #     img_embedding = model.head.patchify(img[0])
    #     # normalize the target image
    #     mean = img_embedding.mean(dim=-1, keepdim=True)
    #     std = (img_embedding.var(dim=-1, keepdim=True) + 1.e-6)**.5
    # else:
    #     mean = imagenet_mean
    #     std = imagenet_std
    imgs_mean =  np.array([cfg.mean[key] for key in cfg.channels])
    imgs_std  =  np.array([cfg.std [key] for key in cfg.channels])

    from torchsummary import summary
    # get reconstruction image
    with torch.no_grad():
        features = model(img, data_samples, mode='tensor')
    results = model.reconstruct(features, mean=0, std=1)

    original_target = model.target if args.target_generator else img[0]

    original_img, img_masked, pred_img, img_paste = post_process(
        original_target,
        results.pred.value,
        results.mask.value,
        mean=imgs_mean,
        std=imgs_std)
        
    # min_ = original_img.min(0).values.min(0).values
    # max_ = original_img.max(0).values.max(0).values
    # original_img = torch.clip(255*(original_img - min_) / (max_-min_), 0, 255).int()
    # img_masked = torch.clip(255*(img_masked - min_) / (max_-min_), 0, 255).int()
    # pred_img = torch.clip(255*(pred_img - min_) / (max_-min_), 0, 255).int()
    # img_paste = torch.clip(255*(img_paste - min_) / (max_-min_), 0, 255).int()
    

    for ch, name in enumerate(cfg.channels):
        os.makedirs(args.out_file, exist_ok=True)
        save_images(original_img[:,:,ch].unsqueeze(-1).repeat(1, 1, 3), 
                    img_masked[:,:,ch].unsqueeze(-1).repeat(1, 1, 3), 
                    pred_img[:,:,ch].unsqueeze(-1).repeat(1, 1, 3), 
                    img_paste[:,:,ch].unsqueeze(-1).repeat(1, 1, 3), 
                    args.out_file + '/' + name)
        if ch == 1: break


if __name__ == '__main__':
    main()
