"""
No@
"""
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MAEPretrainHeadNonFixInChannels(BaseModule):
    """Pre-training head for MAE.
        MODIFIED by No@ FROM mae_head.py
            IT INCLUDES:
                - Customized output channels

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16,
                 channels: int = 3) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.channels = channels
        self.loss = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images, of shape B x H x W x C.

        Returns:
            torch.Tensor: Patchified images. The shape is B x L x D.
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.channels))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 * channels)
        Returns:
            imgs (torch.Tensor): The shape is (N, channels, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.channels, h * p, h * p))
        return imgs

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.

        Args:
            target (torch.Tensor): Image with the shape of B x channels x H x W

        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE head.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        target = self.construct_target(target)
        loss = self.loss(pred, target, mask)

        return loss
