from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE

# fmt: off
full_likelihood = ["mse","homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli", "beta", "static-tgauss","std-mse"]
homoscedestic_mode_map = { "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every","hetero-gauss": "none", "homo-tgauss": "every", "hetero-tgauss": "none", "mse": "none","beta":"none", "static-tgauss":"single", "std-mse":"none"}
likelihood_map = {"bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian_v2", "hetero-gauss": "gaussian_v2", "homo-tgauss": "truncated_gaussian", "hetero-tgauss": "truncated_gaussian", "mse": "gaussian_v2","beta":"beta","static-tgauss":"truncated_gaussian", "std-mse":"gaussian_v2"}
twin_output_map = {"bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True, "homo-tgauss": False,"hetero-tgauss": True, "mse": False,"beta":True, "static-tgauss":False, "std-mse":False}
# fmt: on

n_bae_samples_map = {
    "ens": 5,
    "mcd": 50,
    "sghmc": 50,
    "vi": 50,
    "vae": 50,
    "ae": 1,
    "sae": 1,
}

bae_type_classes = {
    "ens": BAE_Ensemble,
    "mcd": BAE_MCDropout,
    "sghmc": BAE_SGHMC,
    "vi": BAE_VI,
    "vae": VAE,
    "ae": BAE_Ensemble,
    "sae": BAE_Ensemble,
}
