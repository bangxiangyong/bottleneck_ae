import itertools
import pickle

import numpy as np
from tqdm import tqdm

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from need_for_bottleneck.prepare_data_cifar import get_id_set, get_ood_set
from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from util.exp_manager import ExperimentManager

bae_set_seed(100)

# exp name and filenames
exp_name = "CIFAR_"
auroc_filename = exp_name + "AUROC.csv"
bce_se_filename = exp_name + "BCE_VS_SE.csv"
retained_perf_filename = exp_name + "retained_perf.csv"
misclas_perf_filename = exp_name + "misclas_perf.csv"


# Loop over all grid search combinations
# fmt: off
n_random_seeds = 1
random_seeds = np.random.randint(0, 1000, n_random_seeds)
full_likelihood = ["mse", "homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli",
                   "beta"]
homoscedestic_mode_map = {"bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every", "hetero-gauss": "none",
                          "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none", "beta": "none"}
likelihood_map = {"bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian",
                  "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian",
                  "hetero-tgauss": "truncated_gaussian", "mse": "gaussian", "beta": "beta"}
twin_output_map = {"bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True,
                   "homo-tgauss": False, "hetero-tgauss": True, "mse": False, "beta": True}
ood_dataset_map = {"FashionMNIST":"MNIST", "MNIST":"FashionMNIST", "CIFAR":"SVHN", "SVHN":"CIFAR"}
# fmt: on

bae_type_classes = {
    "ens": BAE_Ensemble,
    "mcd": BAE_MCDropout,
    "sghmc": BAE_SGHMC,
    "vi": BAE_VI,
    "vae": VAE,
    "ae": BAE_Ensemble,
}

n_bae_samples_map = {
    "ens": 5,
    "mcd": 100,
    "sghmc": 5,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

# SINGLE SAMPLE
grid = {
    "random_seed": random_seeds,
    # "id_dataset": ["CIFAR"],
    "id_dataset": ["FashionMNIST","MNIST","CIFAR","SVHN"],
    "skip": [True],
    "latent_factor": [2.0],
    "bae_type": ["vae"],
    "full_likelihood": ["bernoulli"],
    "eval_ood_unc": [False],
}

# # BOTTLENECK
# grid = {
#     "random_seed": random_seeds,
#     "id_dataset": ["FashionMNIST","CIFAR","SVHN","MNIST"],
#     "skip": [True,False],
#     "latent_factor": [0.01,0.1,0.5,1.0,2.0],
#     "bae_type": ["ae","ens","vae"],
#     "full_likelihood": ["mse"],
#     "eval_ood_unc":[False]
# }
#
# # LL
# grid = {
#     "random_seed": random_seeds,
#     "id_dataset": ["FashionMNIST","CIFAR","SVHN","MNIST"],
#     "skip": [True],
#     "latent_factor": [1.0],
#     "bae_type": ["ae","ens","vae","vi","mcd","sghmc"],
#     "full_likelihood": ["mse","hetero-gauss","bernoulli","cbernoulli"],
#     "eval_ood_unc":[False]
# }

# fmt: off
id_n_channels = {"CIFAR": 3, "SVHN": 3, "FashionMNIST": 1, "MNIST": 1}
flattened_dims = {"CIFAR": 3*32*32, "SVHN": 3*32*32, "FashionMNIST": 28*28, "MNIST": 28*28}
input_dims = {"CIFAR": 32, "SVHN": 32, "FashionMNIST": 28, "MNIST": 28}
# fmt: on

for values in tqdm(itertools.product(*grid.values())):
    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    id_dataset = exp_params["id_dataset"]

    # === PREPARE DATA ===
    train_loader, test_loader = get_id_set(
        id_dataset=id_dataset, n_channels=id_n_channels[id_dataset]
    )


    # iterate data
    for dt_loader,f_suffix in zip([train_loader,test_loader],["_train.p","_test.p"]):
        # iterate dataloader
        new_data = []
        for batch_idx, (data, target) in tqdm(enumerate(dt_loader)):
            new_data.append(data.cpu().detach().numpy())
        new_data = np.concatenate(new_data,axis=0)

        # save pickle
        pickle.dump(new_data, open(id_dataset+f_suffix,"wb"))









