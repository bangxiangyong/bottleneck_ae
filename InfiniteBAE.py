from jax.config import config; 
config.update('jax_enable_x64', True)

from jax import jit
import neural_tangents as nt
from neural_tangents import stax
import numpy as np


# define maps
act_map = {"erf":stax.Erf(), 
           "leakyrelu":stax.LeakyRelu(0.01), 
           "relu": stax.Relu(),
           "sigmoid":stax.Sigmoid_like(),
           "gelu":stax.Gelu()
           }
norm_map = {"layer":stax.LayerNorm()}
base_map = {"linear":stax.Dense}

# define classes
class InfiniteBAE():
  def __init__(self,diag_reg=1e-4,layers=[], inf_type="nngp", batch_size=0):
    # default architecture
    if len(layers) == 0:
      init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1, W_std=0.5, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=0.5, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=0.5, b_std=0.05)
      )
    else:
      init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    if batch_size <= 0:
      self.kernel_fn = jit(kernel_fn, static_argnums=(2,))
    else:
      self.kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
    self.diag_reg = diag_reg
    self.inf_type = inf_type
  def fit(self,x):
    # train 
    self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.kernel_fn, x, x, diag_reg=self.diag_reg)

  def predict(self,x):
    # predict 
    nngp_mean = self.predict_fn(x_test=x, get=self.inf_type, compute_cov=False)

    return nngp_mean, ((nngp_mean-x)**2)

# get a block of norm-activation-linear
def get_new_block(base_type="linear", activation="erf", norm="none", parameterization="standard", **kwargs):
    # returns a stax serial block
    new_block = []
    if norm != "none":
        new_block.append(norm_map[norm])
    if activation !="none":
        new_block.append(act_map[activation])
    if base_type != "none":
        new_block.append(base_map[base_type](parameterization=parameterization,**kwargs))
    return stax.serial(*new_block)

# return list of blocks
def get_AE_layers(enc_params=["linear","linear"],
                  activation="erf", 
                  norm="none", 
                  last_activation="sigmoid", 
                  parameterization = "standard",
                  W_std = 1.0, 
                  b_std=None,):
    chain_params = enc_params+ enc_params[:-1][::-1]
    print(chain_params)
    all_layers = []
    for i,chain in enumerate(chain_params):
        if i == 0: 
           act_temp = "none"
           norm_temp = "none"
        else:
           act_temp = activation
           norm_temp = norm          
        if chain == "linear":
           new_block = get_new_block(base_type="linear", 
                                        activation=act_temp, 
                                        norm=norm_temp, 
                                        out_dim=5,
                                        parameterization=parameterization,
                                        W_std=W_std, 
                                        b_std=b_std,
                                        )
        all_layers.append(new_block)
    
    # append last act
    all_layers.append(act_map[last_activation])
    return all_layers

# add UNet-skip conn.
def add_unet_skip(ae_layers, last_activation="sigmoid"):
   # extract final act layer 
   if last_activation != "none":
      ae_layers_ = ae_layers[:-1]
      last_act = ae_layers_[-1]
   else:
      ae_layers_ = ae_layers[:]

   # number of layers  
   n_layers = len(ae_layers_)

   # last encoder layer  
   last_enc_layer = (n_layers//2)

   # start the loop 
   # add unet connections 
   skip_layers = []
   for first, i in enumerate(np.arange(0,last_enc_layer)[::-1]):
      if first == 0:
        layer = ae_layers_[last_enc_layer]
      else:
        layer = skip_enc_dec
      skip_enc_dec = stax.serial(ae_layers_[i],
                                  stax.FanOut(2), 
                                  stax.parallel(layer,stax.Identity()),
                                  stax.FanInSum(),
                                  ae_layers_[-i-1]
                                  )
   if last_activation != "none":
      return [stax.serial(skip_enc_dec, last_act)]
   else:
      return [skip_enc_dec]
