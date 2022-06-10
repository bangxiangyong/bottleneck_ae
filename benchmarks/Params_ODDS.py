import os.path
import pickle as pickle

import numpy as np
import torch

from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.Params_GLOBAL import (
    twin_output_map,
    homoscedestic_mode_map,
    likelihood_map,
    n_bae_samples_map,
    bae_type_classes,
)

grid_ODDS = {
    "random_seed": [742],
    "dataset": [
        "cardio",
        "lympho",
        "optdigits",
        "pendigits",
        "thyroid",
        "ionosphere",
        "pima",
        "vowels",
    ],
    "skip": [False],
    # "layer_norm": ["layer"],  # {"layer","none"}
    "layer_norm": ["none"],
    "latent_factor": [0.1],
    "bae_type": ["ae"],
    # "full_likelihood": ["homo-tgauss"],
    # "full_likelihood": ["cbernoulli"],
    "full_likelihood": ["hetero-tgauss"],
    # "full_likelihood": ["hetero-tgauss"],
    # "full_likelihood": ["hetero-gauss"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [300],
}

# grid_ODDS = {
#     "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
#     "dataset": [
#         "cardio",
#         "lympho",
#         "optdigits",
#         "pendigits",
#         "thyroid",
#         "ionosphere",
#         "pima",
#         "vowels",
#     ],
#     "skip": [True],
#     "layer_norm": ["layer"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     # "full_likelihood": ["homo-tgauss"],
#     # "full_likelihood": ["bernoulli"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["hetero-tgauss"],
#     # "full_likelihood": ["hetero-gauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
# }

# LL sweep
# grid_ODDS = {
#     "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
#     "dataset": [
#         "cardio",
#         "lympho",
#         "optdigits",
#         "pendigits",
#         "thyroid",
#         "ionosphere",
#         "pima",
#         "vowels",
#     ],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "vi", "mcd", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
# }


# EXTRA LL
# grid_ODDS = {
#     "random_seed": [595],
#     "dataset": ["thyroid"],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
# }

# BOTTLENECK
# grid_ODDS = {
#     "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
#     "dataset": [
#         "cardio",
#         "lympho",
#         "optdigits",
#         "pendigits",
#         "thyroid",
#         "ionosphere",
#         "pima",
#         "vowels",
#     ],
#     "skip": [False, True],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1, 0.5, 1, 2, 10],
#     "bae_type": ["ae", "mcd", "vi", "vae", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1, 3, 5],
#     "n_enc_capacity": [4],
# }


# grid_ODDS = {
#     "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
#     "dataset": [
#         "cardio",
#         "lympho",
#         "optdigits",
#         "pendigits",
#         "thyroid",
#         "ionosphere",
#         "pima",
#         "vowels",
#     ],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "vi", "mcd", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli", "homo-gauss", "homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
# }

# RECON
# grid_ODDS = {
#     "random_seed": [510],
#     "dataset": [
#         "cardio",
#         "lympho",
#         "optdigits",
#         "pendigits",
#         "thyroid",
#         "ionosphere",
#         "pima",
#         "vowels",
#     ],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1, 0.25, 0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [300],
#     "activation": ["leakyrelu", "gelu", "elu", "selu", "none"],
#     "n_dense_layers": [1, 2, 3, 4],
# }

# BTNECK DEEP
grid_ODDS = {
    "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
    "dataset": [
        "cardio",
        "lympho",
        "optdigits",
        "pendigits",
        "thyroid",
        "ionosphere",
        "pima",
        "vowels",
    ],
    "skip": [False, True],
    "layer_norm": ["none"],
    "latent_factor": [0.1, 0.5, 1, 2, 10],
    "bae_type": ["ae", "mcd", "vi", "vae", "ens"],
    "full_likelihood": ["mse"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [300],
    "activation": ["leakyrelu"],
    "n_dense_layers": [2, 4, 6],
    "n_enc_capacity": [4],
}

#
grid_ODDS = {
    # "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
    "random_seed": [17],
    "dataset": [
        # "vowels",
        # "cardio",
        # "lympho",
        # "optdigits",
        # "pendigits",
        "thyroid",
        # "ionosphere",
        # "pima",
    ],
    "skip": [False],
    "layer_norm": ["none"],  # {"layer","none"}
    # "latent_factor": [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
    "latent_factor": [1],
    "bae_type": ["sae"],
    "full_likelihood": ["mse"],
    "activation": ["leakyrelu"],
    "weight_decay": [0, 1e-10, 1e-6, 1e-4, 1e-3, 1e-2],
    # "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [1],
    "noise_scale": [0],
    "noise_type": ["normal"],
    "n_dense_layers": [2],
    "n_enc_capacity": [4],
}


def prepare_data(pickle_path="pickles", suffix_name="_odds_benchmark_v2.p"):
    odds_data = {}
    for scaling in ["minmax", "standard", "quantile"]:
        filepath = pickle_path + "/" + scaling + suffix_name
        if os.path.exists(filepath):
            temp_data = pickle.load(
                open(pickle_path + "/" + scaling + suffix_name, "rb")
            )
            odds_data.update({scaling: temp_data})
    return odds_data


def get_x_splits(
    odds_data,
    exp_params,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # NOTE: DATA IS ALREADY PREPARED BEFOREHAND FOR ODDS IN ANOTHER SCRIPT
    # extract random_seed and odds sub-dataset
    random_seed = exp_params["random_seed"]
    dataset_ = exp_params["dataset"]
    dataset_split = dataset_ + "-" + str(random_seed)
    standardise = (
        "standard"
        if ("full_likelihood" in exp_params)
        and (exp_params["full_likelihood"] == "std-mse")
        else "minmax"
    )
    # load data
    x_id_train = odds_data[standardise][dataset_split]["x_id_train"]
    x_id_test = odds_data[standardise][dataset_split]["x_id_test"]
    x_ood_test = odds_data[standardise][dataset_split]["x_ood_test"]

    return x_id_train, x_id_test, x_ood_test


def get_bae_model(
    exp_params,
    x_id_train,
    activation="selu",
    bias=False,
    use_cuda=False,
    lr=0.001,
    dropout_rate=0.01,
    mean_prior_loss=False,
    collect_grads=False,
):
    # unpack exp_params variables
    random_seed = exp_params["random_seed"]
    bae_type = exp_params["bae_type"]
    latent_factor = exp_params["latent_factor"]
    layer_norm = exp_params["layer_norm"]
    full_likelihood_i = exp_params["full_likelihood"]
    skip = exp_params["skip"]
    weight_decay = exp_params["weight_decay"]
    n_bae_samples = exp_params["n_bae_samples"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"
    bae_set_seed(random_seed)

    # size of network
    n_dense_layers = (
        exp_params["n_dense_layers"] if "n_dense_layers" in exp_params.keys() else 1
    )
    n_enc_capacity = (
        exp_params["n_enc_capacity"] if "n_enc_capacity" in exp_params.keys() else 1
    )
    # n_dense_nodes = n_enc_capacity

    # continue unpacking based on chosen likelihood param
    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    if n_bae_samples == -1:  # resort to default
        n_bae_samples = n_bae_samples_map[bae_type]

    # unpack this function params
    dropout_params = {"dropout_rate": dropout_rate} if bae_type == "mcd" else {}

    # get input dimensions
    # required to scale the latent dim and specify architecture input nodes
    input_dim = x_id_train.shape[-1]
    if isinstance(x_id_train, torch.utils.data.dataloader.DataLoader):
        shapes = list(next(iter(x_id_train))[0].shape)
        latent_dim = int(np.product(shapes) * latent_factor)
    else:  # numpy
        latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)
    latent_dim = np.clip(latent_dim, 1, None)
    hidden_layers = [input_dim * n_enc_capacity] * n_dense_layers

    # specify architecture
    chain_params = [
        {
            "base": "linear",
            # "architecture": [input_dim, input_dim * 4, input_dim * 4, latent_dim],
            "architecture": [input_dim] + hidden_layers + [latent_dim],
            "activation": activation,
            "norm": layer_norm,
            "last_norm": layer_norm,
            "bias": bias,
        }
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid" if standardise == "minmax" else "none",
        last_norm=layer_norm,
        twin_output=twin_output,
        twin_params={"activation": "none", "norm": "none"},
        skip=skip,
        use_cuda=use_cuda,
        scaler_enabled=False,
        homoscedestic_mode=homoscedestic_mode,
        likelihood=likelihood,
        weight_decay=weight_decay,
        num_samples=n_bae_samples,
        anchored=True if bae_type == "ens" else False,
        l1_prior=True if bae_type == "sae" else False,
        learning_rate=lr,
        stochastic_seed=random_seed,
        mean_prior_loss=mean_prior_loss,
        collect_grads=collect_grads,
        **dropout_params,
    )

    return bae_model
