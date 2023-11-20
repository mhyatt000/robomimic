"""
Goal Conditioned Behavior Cloning
given a goal image, predict actions over time
especially suited for transformers
read: RT2, RoboCat
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.bc_config import BCConfig

class GCBCConfig(BCConfig, BaseConfig):
    """
    Goal Conditioned Behavior Cloning
    given a goal image, predict actions over time
    especially suited for transformers
    read: RT2, RoboCat
    """

    ALGO_NAME='gcbc'


    def train_config(self):
        """
        This function populates the `config.train` attribute of the config, which
        has several settings related to the training process, such as the dataset
        to use for training, and how the data loader should load the data. This
        class has a default implementation that usually doesn't need to be overriden.
        """
        super(GCBCConfig, self).train_config()

        # num workers for loading data - generally set to 0 for low-dim datasets, and 2 for image datasets
        self.train.num_data_workers = 2
        self.train.hdf5_load_next_obs = True # needed for goals rn

        # one of [None, "last"] - set to "last" to include goal observations in each batch
        # TODO add an option for random goals like in STEVE-1 paper
        self.train.goal_mode = "last"

        self.train.seq_length = 10
        self.train.pad_seq_length = True
        self.train.frame_stack = 10
        self.train.pad_frame_stack = True


    def algo_config(self):
        """
        populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """
        super(GCBCConfig, self).algo_config()

        # stochastic VAE policy settings
        self.algo.vae.enabled = False                   # whether to train a VAE policy
        self.algo.vae.latent_dim = 14                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 1.                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)

        # Transformer policy settings
        self.algo.transformer.enabled = True                       ## whether to train transformer policy
        self.algo.transformer.context_length = 10                   # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
        self.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
        self.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.transformer.block_output_dropout = 0.1            # dropout probability for final outputs for each transformer block
        self.algo.transformer.sinusoidal_embedding = False          # if True, use standard positional encodings (sin/cos)
        self.algo.transformer.supervise_all_steps = True           ## if true, supervise all intermediate actions, otherwise only final one
        self.algo.transformer.nn_parameter_for_timesteps = True     # if true, use nn.Parameter otherwise use nn.Embedding


    def observation_config(self):
        """docstring"""
        super(GCBCConfig, self).observation_config()

        # observation modalities
        self.observation.modalities.obs.low_dim = [  # specify low-dim observations for agent
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        # specify rgb image observations for agent
        self.observation.modalities.obs.rgb = [
            "sideview_image",
            "robot0_eye_in_hand_image"
        ]

        # specify rgb image goal observations to condition agent on
        self.observation.modalities.goal.rgb = ["sideview_image"]

        self.observation.encoder.rgb.core_kwargs.backbone_class = "R3MConv"


    def meta_config(self):
        """
        This function populates the `config.meta` attribute of the config. This portion of the config
        is used to specify job information primarily for hyperparameter sweeps.
        It contains hyperparameter keys and values, which are populated automatically
        by the hyperparameter config generator (see `utils/hyperparam_utils.py`).
        These values are read by the wandb logger (see `utils/log_utils.py`) to set job tags.
        """

        self.meta.hp_base_config_file = None  # base config file in hyperparam sweep
        self.meta.hp_keys = []  # relevant keys (swept) in hyperparam sweep
        self.meta.hp_values = []  # values corresponding to keys in hyperparam sweep



# class GLConfig(BaseConfig):

def algo_config(self):
    """
    This function populates the `config.algo` attribute of the config, and is given to the
    `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
    argument to the constructor. Any parameter that an algorithm needs to determine its
    training and test-time behavior should be populated here.
    """

    # subgoal definition: observation that is @subgoal_horizon number of timesteps in future from current observation
    self.algo.subgoal_horizon = 10

    # MLP size for deterministic goal network (unused if VAE is enabled)
    self.algo.ae.planner_layer_dims = (300, 400)

    # ================== VAE config ==================
    self.algo.vae.enabled = True                                        # set to true to use VAE network
    self.algo.vae.latent_dim = 16                                       # VAE latent dimension
    self.algo.vae.latent_clip = None                                    # clip latent space when decoding (set to None to disable)
    self.algo.vae.kl_weight = 1.                                        # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

    # VAE decoder settings
    self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
    self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

    # VAE prior settings
    self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
    self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
    self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
    self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
    self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights
    self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
    self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
    self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
    self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
    self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
    self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

    self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
    self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
    self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)


