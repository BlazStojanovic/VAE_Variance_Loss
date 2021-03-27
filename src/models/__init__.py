from .base import *
from .vanilla_vae import *
from .cont_reinforce_vae import *
from .cont_variance_vae import *
from .cont_variance_vae_conv import *

from .disc_reinforce_vae import *
from .disc_variance_vae import *
from .disc_variance_vae_2z import * 

# Aliases
VAE = VanillaVAE

vae_models = {
              'VanillaVAE':VanillaVAE
              }
