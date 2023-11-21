"""
Goal Conditioned Behavior Cloning
see RT2, RoboCat, STEVE-1
"""


from collections import OrderedDict

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.algo.bc import BC, BC_Transformer


@register_algo_factory_func("gcbc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the GCBC algo class to instantiate,
    along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = "gaussian" in algo_config and algo_config.gaussian.enabled
    gmm_enabled = "gmm" in algo_config and algo_config.gmm.enabled
    vae_enabled = "vae" in algo_config and algo_config.vae.enabled

    rnn_enabled = False
    transformer_enabled = True

    if gaussian_enabled:
        raise NotImplementedError
    elif gmm_enabled:
        raise NotImplementedError
        # algo_class, algo_kwargs = BC_Transformer_GMM, {}
    elif vae_enabled:
        raise NotImplementedError
    else:
        algo_class, algo_kwargs = GCBC_Transformer, {}

    return algo_class, algo_kwargs


class GoalConditioned():
    """ Goal Conditioned mixin """
    
    def __init__(self, ):
        pass


class GCBC_Transformer(BC_Transformer, GoalConditioned):
    """GCBC training with a Transformer policy."""

    def _create_networks(self):
        """Creates networks and places them into @self.nets."""
        BC_Transformer._create_networks(self)

    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        BC_Transformer._set_params_from_config(self)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        # always supervise all steps
        return BC_Transformer.process_batch_for_training(self, batch)

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg=f"Error: expect temporal dimension of obs batch to match transformer context length {self.context_length}",
        )
        return super()._forward_training(batch, epoch=None)

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        super()._compute_losses(predictions, batch)

    def get_action(self, obs_dict, goal_dict):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        return BC_Transformer.get_action(self, obs_dict, goal_dict)

    def reset(self):
        """prepare transformer for autoregressive inference in rollout"""
        BC_Transformer.reset(self)


