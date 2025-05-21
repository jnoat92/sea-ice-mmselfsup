'''
No@
January 24th, 2025
Incomplete
'''
from typing import Optional, Dict
import torch
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer import AmpOptimWrapper

@OPTIM_WRAPPERS.register_module()
class CustomAmpOptimWrapper(AmpOptimWrapper):
    """
    Customized AmpOptimWrapper
    method 'update_params' modified by No@:
        include optional backward arguments 'back_kwargs', ex: retain_graph=True
    """

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            back_kwargs: Optional[Dict] = None,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            back_kwargs (dict): Arguments for backward.
                Defaults to None.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss, **back_kwargs)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)


