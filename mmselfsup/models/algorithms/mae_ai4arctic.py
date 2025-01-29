# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import BaseDataElement

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .mae import MAE


@MODELS.register_module()
class MAE_CCH(MAE):
    """MAE.
        def reconstruct MODIFIED by No@ FROM mae.py
            IT INCLUDES:
                - Customized head predicted channels
    """

    def reconstruct(self,
                    features: torch.Tensor,
                    data_samples: Optional[List[SelfSupDataSample]] = None,
                    **kwargs) -> SelfSupDataSample:
        """The function is for image reconstruction.
        MODIFIED by No@ FROM mae.py
            IT INCLUDES:
                - Customized head predicted channels

        Args:
            features (torch.Tensor): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            SelfSupDataSample: The prediction from model.
        """
        mean = kwargs['mean']
        std = kwargs['std']
        features = features * std + mean

        pred = self.head.unpatchify(features)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = self.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         self.head.channels)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        results = SelfSupDataSample()
        results.mask = BaseDataElement(**dict(value=mask))
        results.pred = BaseDataElement(**dict(value=pred))

        return results
