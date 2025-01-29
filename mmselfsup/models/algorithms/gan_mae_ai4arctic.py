'''
No@
January 20th, 2025
'''
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from mmengine.structures import BaseDataElement

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel
from .mae_ai4arctic import MAE_CCH
from mmengine.optim import OptimWrapper
import numpy as np


@MODELS.register_module()
class GAN_MAE_CCH(MAE_CCH):
    """GAN-MAE.
        Implementation of `Masked Auto-Encoders Meet Generative Adversarial Networks and Beyond
        <DOI:10.1109/CVPR52729.2023.02342>`_.
    """
    def __init__(self, *args, **kwargs):

        # Generator (MAE) specifics
        super().__init__(*args, **kwargs)
        # Discriminator Specifics
        self.discriminate = nn.Linear(self.backbone.embed_dims, 1, bias = True)

    def discriminator(self, x):
        """
        Same network (parameters shared) as the ViT encoder, but excluding the masking.

        Args:
            x (torch.Tensor): Input tokens, which are of shape B x L x C.
        Returns:
            torch.Tensor: prediction (scalar)
        """
        B = x.shape[0]
        x = self.backbone.patch_embed(x)[0]
        # print("after patch embed = "+str(x.size()))
        # add pos embed w/o cls token
        x = x + self.backbone.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for _, layer in enumerate(self.backbone.layers):
            x = layer(x)
        # Use final norm
        x = self.backbone.norm1(x)

        # x = x.view(x.size(0), -1)

        x = self.discriminate(x)

        return x
    
    def discriminator_loss(self, x, mask):
        # Real and fake discriminator outputs
        logits = self.discriminator(x)
        logits = logits[:, 1:, 0]
        target = 1 - mask
        target = target.double()

        disc_loss = torch.nn.BCEWithLogitsLoss()
        return disc_loss(logits, target)
    
    def adv_loss(self, currupt_img, mask):
        target = 1 - mask  # This flips the mask values
        output = torch.sigmoid(self.discriminator(currupt_img))
        disc_preds = output[:, 1:, 0]

        # Reshape target to match the discriminator output shape
        target = target.view(disc_preds.shape)
        target = target.float()

        # Calculate the number of correct predictions for original and reconstructed patches
        corr_orig = (torch.log(disc_preds + 1e-8) * target).sum()/(target.sum())
        corr_recons = (torch.log((1-disc_preds + 1e-8))*(1 - target)).sum()/((1-target).sum())
        # print(corr_orig)
        # print(corr_orig + corr_recons)
        return (corr_orig) + (corr_recons) 

    def aw_loss(self, L_mae, L_adv, Gen_opt):
        # resetting gradient back to zero
        Gen_opt.zero_grad()

        # computing real batch gradient 
        L_mae.backward(retain_graph=True)
        # tensor with real gradients
        grad_real_tensor = [param.grad.clone() for _, param in self.named_parameters() if param.grad is not None]
        grad_real_list = torch.cat([grad.reshape(-1) for grad in grad_real_tensor], dim=0)
        # calculating the norm of the real gradient
        rdotr = torch.dot(grad_real_list, grad_real_list).item() 
        mae_norm = np.sqrt(rdotr)
        # resetting gradient back to zero
        Gen_opt.zero_grad()

        # computing fake batch gradient 
        L_adv.backward(retain_graph = True)#(retain_graph=True)
        # tensor with real gradients
        grad_fake_tensor = [param.grad.clone() for _, param in self.named_parameters() if param.grad is not None]
        grad_fake_list = torch.cat([grad.reshape(-1) for grad in grad_fake_tensor], dim=0)
        # calculating the norm of the fake gradient
        fdotf = torch.dot(grad_fake_list, grad_fake_list).item() + 1e-6 # 1e-4 added to avoid division by zero
        adv_norm = np.sqrt(fdotf)
        
        # resetting gradient back to zero
        Gen_opt.zero_grad()

        # dot product between real and fake gradients
        adaptive_weight = mae_norm/adv_norm
        # print(adaptive_weight)
        # print(L_mae)
        # print(L_adv)
        # calculating aw_loss
        aw_loss = L_mae + adaptive_weight * L_adv

        # updating gradient, i.e. getting aw_loss gradient
        for index, (_, param) in enumerate(self.named_parameters()):
            # print(grad_real_tensor[index])
            # print(grad_fake_tensor[index])
            if param.grad is not None:
                param.grad = grad_real_tensor[index] + adaptive_weight * grad_fake_tensor[index]

        return aw_loss

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.

        # =============== MAE
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        mae_loss = self.head(pred, inputs[0], mask)

        # =============== GAN
        img_patched = self.head.patchify(inputs[0])
        currupt_img = torch.zeros(img_patched.size())
        mask1 = mask.unsqueeze(-1).expand_as(pred)
        currupt_img = torch.where(mask1 == 1, pred, img_patched)
        currupt_img = self.head.unpatchify(currupt_img)

        # discriminator loss        
        disc_loss = self.discriminator_loss(currupt_img, mask)
        # generator loss        
        adv_loss = self.adv_loss(currupt_img, mask)

        losses = dict(mae_loss=mae_loss, 
                      disc_loss=disc_loss, 
                      adv_loss=adv_loss)
        return losses

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)

            # Train Generator (MAE)
            losses = self._run_forward(data, mode='loss')
            gen_loss = self.aw_loss(losses['mae_loss'], losses['adv_loss'], optim_wrapper)
            parsed_gen_loss, _ = self.parse_losses(Dict(gen_loss=gen_loss))
            optim_wrapper.update_params(parsed_gen_loss, Dict(retain_graph=True))   # retain computational graph for the subsequent backward pass.

            # Train Discriminator (MAE's encoder)
            losses = self._run_forward(data, mode='loss')
            parsed_disc_loss, _ = self.parse_losses(Dict(disc_loss=losses['disc_loss']))
            optim_wrapper.update_params(parsed_disc_loss)

            _, log_vars = self.parse_losses(losses)

        return log_vars



