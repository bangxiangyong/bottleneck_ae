import pickle

from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.Params_GLOBAL import (
    twin_output_map,
    homoscedestic_mode_map,
    likelihood_map,
    n_bae_samples_map,
    bae_type_classes,
)
import numpy as np

from thesis_experiments.benchmarks.Images_util import (
    get_id_set,
    get_ood_set,
    get_standardise_vals,
)

# grid_Images = {
#     "random_seed": [510],
#     "id_dataset": ["IMAGES"],  # loop will change it later
#     "skip": [True],
#     "layer_norm": ["none"],
#     # "skip": [False],
#     # "layer_norm": ["none"],
#     "latent_factor": [0.01],
#     "bae_type": ["vi"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [10],
# }

# FULL-LL
# grid_Images = {
#     "random_seed": [510, 365, 382, 322, 988],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#         "MNIST",
#         "SVHN",
#     ],  # loop will change it later
#     "skip": [False],
#     "layer_norm": ["none", "layer"],
#     # "skip": [False],
#     # "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": [
#         "mse",
#         "bernoulli",
#         "cbernoulli",
#         "homo-tgauss",
#         "hetero-gauss",
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

# DEBUGGING
# grid_Images = {
#     "random_seed": [510],
#     "id_dataset": ["SVHN"],
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": [
#         "vae",
#     ],
#     "full_likelihood": ["mse"],
#     "layer_norm": ["none"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [5],
# }

# grid_Images = {
#     "random_seed": [510],
#     "id_dataset": ["CIFAR"],
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": [
#         "ae",
#     ],
#     "full_likelihood": ["mse"],
#     "layer_norm": ["none"],
#     "weight_decay": [1e-4],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [1],
# }

# grid_Images = {
#     "random_seed": [510],
#     "id_dataset": ["CIFAR"],
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": [
#         "ae",
#     ],
#     "full_likelihood": ["hetero-tgauss"],
#     "layer_norm": ["none"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [5],
# }

## FULL-LL 20220420
# grid_Images = {
#     "random_seed": [510, 365, 382, 322, 988],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#         "MNIST",
#         "SVHN",
#     ],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": [
#         "mse",
#         "bernoulli",
#         "cbernoulli",
#         "homo-gauss",
#         "homo-tgauss",
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

## BTNECK 20220420
# grid_Images = {
#     "random_seed": [510, 365, 382, 322, 988],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#     ],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1, 2],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

# BTNECK REBOOT 20220420
# grid_Images = {
#     "random_seed": [510, 100, 9, 69, 12],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#     ],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1, 2],
#     # "latent_factor": [0.1, 0.5, 2],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-11],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

# grid_Images = {
#     "random_seed": [510, 100, 9, 69, 12],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#     ],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1, 2],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

# REBOOT FULL-LL
# grid_Images = {
#     "random_seed": [510, 100, 9, 69, 12],
#     "id_dataset": [
#         "FashionMNIST",
#         "CIFAR",
#         "MNIST",
#         "SVHN",
#     ],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": [
#         "mse",
#         "bernoulli",
#         "cbernoulli",
#         "homo-gauss",
#         "homo-tgauss",
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
# }

# grid_Images = {
#     "random_seed": [510, 100, 9, 69, 12],
#     "id_dataset": ["FashionMNIST", "SVHN", "CIFAR", "MNIST"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ens"],
#     "full_likelihood": [
#         "homo-gauss",
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [5],  # default
#     "num_epochs": [20],
# }

# grid_Images = {
#     "random_seed":  [930, 717, 10, 5477, 510],
#     "id_dataset": [
#         "FashionMNIST", "CIFAR"
#     ],
#     "skip": [False,True],
#     "layer_norm": ["none","layer"],
#     "latent_factor": [0.1,0.5,1.0,2.0,10.],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": [
#         "mse"
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1,3,5],
#     "n_enc_capacity": [30],
# }

# =======OVERHAUL BTNECK========
# grid_Images = {
#     "random_seed":  [930, 717, 10, 5477, 510],
#     "id_dataset": [
#         "FashionMNIST", "CIFAR"
#     ],
#     "skip": [False,True],
#     "layer_norm": ["none","layer"],
#     "latent_factor": [0.1,0.5,1.0,2.0,10.],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": [
#         "mse"
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1,3,5],
#     "n_enc_capacity": [30],
# }

# grid_Images = {
#     "random_seed": [930],
#     "id_dataset": ["CIFAR"],
#     "skip": [True],
#     "layer_norm": ["layer"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [5],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [5],
#     "n_enc_capacity": [30],
# }

# BT-NECK OVERHAUL
# grid_Images = {
#     "random_seed": [930, 717, 10, 5477, 510],
#     "id_dataset": ["FashionMNIST"],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1.0, 2.0, 10.0],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1, 3, 5],
#     "n_enc_capacity": [30],
# }


# BT-NECK OVERHAUL
# grid_Images = {
#     "random_seed": [930, 717, 10, 5477, 510],
#     "id_dataset": ["FashionMNIST", "CIFAR"],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1.0, 2.0, 10.0],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [20],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1, 3, 5],
#     "n_enc_capacity": [30],
# }

grid_Images = {
    "random_seed": [717],
    "id_dataset": ["FashionMNIST"],
    "skip": [False],
    "layer_norm": ["none"],
    "latent_factor": [0.1],
    "bae_type": ["ae"],
    "full_likelihood": ["mse"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [1],
    "activation": ["leakyrelu"],
    "n_dense_layers": [1],
    "n_conv_layers": [3],
    "n_enc_capacity": [30],
    # "noise_scale": [0.5],
    # "noise_type": ["normal"],
}


# fmt: off
id_n_channels = {"CIFAR": 3, "SVHN": 3, "FashionMNIST": 1, "MNIST": 1}
flattened_dims = {"CIFAR": 3*32*32, "SVHN": 3*32*32, "FashionMNIST": 28*28, "MNIST": 28*28}
input_dims = {"CIFAR": 32, "SVHN": 32, "FashionMNIST": 28, "MNIST": 28}
# fmt: on


def get_x_splits(
    exp_params,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test
    ood_dataset_map = {
        "FashionMNIST": "MNIST",
        "MNIST": "FashionMNIST",
        "CIFAR": "SVHN",
        "SVHN": "CIFAR",
    }
    id_dataset = exp_params["id_dataset"]

    # apply z-standardise if needed
    standardise = (
        get_standardise_vals(id_dataset)
        if ("full_likelihood" in exp_params)
        and (exp_params["full_likelihood"] == "std-mse")
        else None
    )
    # === PREPARE DATA ===
    noise_scale = (
        exp_params["noise_scale"] if "noise_scale" in exp_params.keys() else None
    )
    noise_type = exp_params["noise_type"] if "noise_type" in exp_params.keys() else None

    train_loader, test_loader = get_id_set(
        id_dataset=id_dataset,
        n_channels=id_n_channels[id_dataset],
        standardise=standardise,
        noise_scale=noise_scale,
        noise_type=noise_type,
    )
    ood_loader = get_ood_set(
        ood_dataset=ood_dataset_map[id_dataset],
        n_channels=id_n_channels[id_dataset],
        resize=[input_dims[id_dataset]] * 2,
        standardise=standardise,
        noise_scale=noise_scale,
        noise_type=noise_type,
    )

    return train_loader, test_loader, ood_loader


def get_x_splits_v2(
    exp_params,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test
    ood_dataset_map = {
        "FashionMNIST": "MNIST",
        "MNIST": "FashionMNIST",
        "CIFAR": "SVHN",
        "SVHN": "CIFAR",
    }
    id_dataset = exp_params["id_dataset"]

    # apply z-standardise if needed
    standardise = (
        get_standardise_vals(id_dataset)
        if ("full_likelihood" in exp_params)
        and (exp_params["full_likelihood"] == "std-mse")
        else None
    )
    # === PREPARE DATA ===
    noise_scale = (
        exp_params["noise_scale"] if "noise_scale" in exp_params.keys() else None
    )
    noise_type = exp_params["noise_type"] if "noise_type" in exp_params.keys() else None

    # ===load pickle====
    id_train_loader, id_test_loader = get_id_set(
        id_dataset=id_dataset,
        n_channels=id_n_channels[id_dataset],
        standardise=standardise,
        noise_scale=noise_scale,
        noise_type=noise_type,
    )
    ood_test_loader = get_ood_set(
        ood_dataset=ood_dataset_map[id_dataset],
        n_channels=id_n_channels[id_dataset],
        # resize=[input_dims[id_dataset]] * 2,
        standardise=standardise,
        noise_scale=noise_scale,
        noise_type=noise_type,
    )

    # iterate data
    for dt_loader, x_id_label in zip(
        [id_train_loader, id_test_loader, ood_test_loader],
        ["x_id_train", "x_id_test", "x_ood_test"],
    ):
        # iterate dataloader
        new_data = []
        for batch_idx, (data, target) in enumerate(dt_loader):
            new_data.append(data.cpu().detach().numpy())
        new_data = np.concatenate(new_data, axis=0)

        if x_id_label == "x_id_train":
            x_id_train = np.copy(new_data)
        elif x_id_label == "x_id_test":
            x_id_test = np.copy(new_data)
        if x_id_label == "x_ood_test":
            x_ood_test = np.copy(new_data)

    train_loader = convert_dataloader(x_id_train.copy(), shuffle=True, drop_last=True)
    # train_loader = id_train_loader
    test_loader = convert_dataloader(x_id_test.copy(), drop_last=False)
    ood_loader = convert_dataloader(x_ood_test.copy(), drop_last=False)

    return train_loader, test_loader, ood_loader


def get_bae_model(
    exp_params,
    x_id_train,
    activation="leakyrelu",
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
    id_dataset = exp_params["id_dataset"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"
    bae_set_seed(random_seed)

    # size of network
    n_dense_layers = (
        exp_params["n_dense_layers"] if "n_dense_layers" in exp_params.keys() else 1
    )
    n_conv_layers = (
        exp_params["n_conv_layers"] if "n_conv_layers" in exp_params.keys() else 1
    )
    n_enc_capacity = (
        exp_params["n_enc_capacity"] if "n_enc_capacity" in exp_params.keys() else 1
    )
    n_conv_filters = n_enc_capacity * 1
    n_dense_nodes = n_enc_capacity * 100

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
    if id_dataset == "CIFAR" or id_dataset == "SVHN":
        input_dim = list([32, 32])
        input_channel = 3
    else:
        input_dim = list([28, 28])
        input_channel = 1

    latent_dim = int(
        (input_dims[id_dataset] ** 2) * id_n_channels[id_dataset] * latent_factor
    )
    latent_dim = np.clip(latent_dim, 1, None)

    # specify architecture
    chain_params = [
        # {
        #     "base": "conv2d",
        #     "input_dim": input_dim,
        #     "conv_channels": [input_channel, 10, 32],
        #     "conv_stride": [2, 1],
        #     "conv_kernel": [2, 2],
        #     "activation": activation,
        #     "norm": layer_norm,
        #     "order": ["base", "norm", "activation"],
        #     "bias": bias,
        #     "last_norm": layer_norm,
        # },
        # {
        #     "base": "linear",
        #     "architecture": [100] * n_dense_layers + [latent_dim],
        #     "activation": activation,
        #     "norm": "none",
        #     "last_norm": "none",
        #     "bias": bias,
        # },
        {
            "base": "conv2d",
            "input_dim": input_dim,
            "conv_channels": [input_channel] + [n_conv_filters] * (n_conv_layers + 1),
            "conv_stride": [2] + [1] * n_conv_layers,
            "conv_kernel": [2] + [2] * n_conv_layers,
            "activation": activation,
            "norm": layer_norm,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": layer_norm,
        },
        {
            "base": "linear",
            "architecture": [n_dense_nodes] * n_dense_layers + [latent_dim],
            # "architecture": [100] * n_dense_layers + [latent_dim],
            "activation": activation,
            "norm": "none",
            "last_norm": "none",
            "bias": bias,
        },
    ]
    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid" if standardise == "minmax" else "none",
        last_norm=layer_norm,
        twin_output=twin_output,
        twin_params={
            "activation": "sigmoid" if full_likelihood_i == "hetero-tgauss" else "selu",
            "norm": "none",
        },
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
