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


class GCBC_Transformer(BC_Transformer):
    """GCBC training with a Transformer policy."""

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cached_goals = {}

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
        # no need to do this for goal_dict since self.nets does this 
        TensorUtils.assert_size_at_dim(
            batch['obs'],
            size=(self.context_length),
            dim=1,
            msg=f"Error: expect temporal dimension of obs batch to match transformer context length {self.context_length}",
        )
        return super()._forward_training(batch, epoch=None)

    def get_cached_goals(self, env_name):
        """ 
        returns the surrogate goal_dict 
        since envs don't seem to return goals
        """
        return self.cached_goals[env_name]

    def cache_goal_dict(self, cache_dict):
        """ stores the goal_dict for use with the rollout """
        self.cached_goals = cache_dict
