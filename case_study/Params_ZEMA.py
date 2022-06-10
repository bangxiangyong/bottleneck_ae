import pickle as pickle

import numpy as np
from sklearn.model_selection import train_test_split

from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.Params_GLOBAL import (
    twin_output_map,
    homoscedestic_mode_map,
    likelihood_map,
    n_bae_samples_map,
    bae_type_classes,
)
import torch
from uncertainty_ood_v2.util.sensor_preproc import (
    FFT_Sensor,
    MinMaxSensor,
    StandardiseSensor,
    FlattenStandardiseScaler,
)

# grid_ZEMA = {
#     "random_seed": np.random.randint(0, 1000, size=3),
#     "apply_fft": [False],
#     # "apply_fft": [True],
#     # "ss_id": [
#     #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
#     # ],  # before eliminate correlated sensors
#     "ss_id": [
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
#     ],  # after eliminating correlated sensors
#     # "ss_id": [[3, 0, 1, 9]],
#     # "ss_id": [[10, 3]],  # after eliminating correlated sensors
#     # "ss_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
#     # "ss_id": [[8, 5]],
#     # "ss_id": [[9, 0, 2, 1, 3, 11, 7, 10, 13, 8, 4, 5, 6, 12]],
#     # "ss_id": [[10, 13, 0, 11]],0.1
#     # "ss_id": [[10, 3]],
#     # "ss_id": [[5, 8]],
#     # "target_dim": [2],
#     "target_dim": [1],
#     # "resample_factor": [resample_factor],
#     "resample_factor": ["Hz_1"],
#     # "skip": [True],
#     "skip": [False],
#     # "layer_norm": ["layer"],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     # "bae_type": ["ens"],
#     # "full_likelihood": ["bernoulli"],
#     # "full_likelihood": ["cbernoulli"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["hetero-gauss"],
#     # "full_likelihood": ["homo-tgauss"],
#     # "full_likelihood": ["hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [100],
# }

# SENSOR RANKING BASED LL
# grid_ZEMA = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
#     # after eliminating correlated sensors
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [100],
# }

# EXTRA: SENSOR RANKING BASED LL
# grid_ZEMA = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
#     # after eliminating correlated sensors
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [100],
# }
#
#
# # LL SWEEP TOPK
# grid_ZEMA = {
#     "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [100],
# }

# EXTRA: LL SWEEP TOPK
# grid_ZEMA = {
#     "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [100],
# }


# GET Y RECON TEST FOR NORMALITY
# grid_ZEMA = {
#     "random_seed": [100, 5, 1, 3, 4],
#     "apply_fft": [False],
#     "ss_id": [[-1]],
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ens"],
#     "full_likelihood": ["homo-gauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [5],  # default
#     "num_epochs": [200],
# }

# # REBOOT: 200 EPOCH LL RANKING
# grid_ZEMA = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
#     # after eliminating correlated sensors
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": [
#         "none",
#     ],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli", "homo-gauss", "homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }
#
# # REBOOT: FULL LL+BAE
# grid_ZEMA = {
#     "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli", "homo-gauss", "homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }
#
# # BOTTLENECK
# grid_ZEMA = {
#     "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1.0, 10],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# grid_ZEMA = {
#     "full_likelihood": ["homo-gauss"],
#     # "full_likelihood": ["static-tgauss", "mse"],
#     # "standardise": ["minmax", "standard"],
#     "standardise": ["minmax"],
#     "random_seed": [333, 111, 222],
#     "apply_fft": [False],
#     "ss_id": [[-1]],
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [1],  # default
#     "num_epochs": [200],
#     # "standardise": ["minmax"],
# }

# grid_ZEMA = {
#     "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["std-mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }
#
# # ESTABLISH BASELINE
# grid_ZEMA = {
#     "random_seed": [53],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1, 0.25, 0.5],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu", "gelu", "elu", "selu", "none"],
#     "n_dense_layers": [1, 2, 3, 4],
#     "encoder_size": [1000],
# }

# EFFECTIVE CAPACITY
# grid_ZEMA = {
#     "random_seed": [9999],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "resample_factor": ["Hz_1"],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [2],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_enc_capacity": [10],
# }

# DEEP LAYERS + SKIP?
# grid_ZEMA = {
#     "random_seed": [11, 22],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [3],
#     "resample_factor": ["Hz_1"],
#     "skip": [False, True],
#     "layer_norm": ["none"],
#     "latent_factor": [0.01, 0.02, 0.1, 0.5, 1, 2, 10],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1, 5, 20, 40],
#     "n_enc_capacity": [3],
# }

# grid_ZEMA = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": ["Hz_1"],
#     "layer_norm": ["none"],
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1],
#     "n_enc_capacity": [20],
# }

grid_ZEMA = {
    "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
    "apply_fft": [False],
    # fmt: off
    "ss_id": [[-1]],
    # to be replaced via Best TOP-K Script
    # fmt: on
    "target_dim": [0, 1, 2, 3],
    "resample_factor": ["Hz_1"],
    "layer_norm": ["none"],
    "skip": [False],
    "latent_factor": [0.1],
    "bae_type": ["ae"],
    "full_likelihood": ["mse"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [200],
    "activation": ["leakyrelu"],
    "n_dense_layers": [1],
    "n_conv_layers": [1],
    "n_enc_capacity": [20],
    # "noise_scale": [0.0],
    # "noise_type": ["normal"],
}


def prepare_data(pickle_path="pickles"):
    zema_data = pickle.load(open(pickle_path + "/" + "zema_hyd_inputs_outputs.p", "rb"))
    return zema_data


def get_x_splits(zema_data, exp_params, min_max_clip=True, train_size=0.70):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # unpack exp_params variables
    apply_fft = exp_params["apply_fft"]
    random_seed = exp_params["random_seed"]
    target_dim = exp_params["target_dim"]
    resample_factor = exp_params["resample_factor"]
    sensor_i = exp_params["ss_id"]
    target_dim = [target_dim] if not isinstance(target_dim, list) else target_dim

    standardise = (
        "standard"
        if ("full_likelihood" in exp_params)
        and (exp_params["full_likelihood"] == "std-mse")
        else "minmax"
    )

    # ood and id
    id_args = zema_data["id_target"][target_dim[0]]
    ood_args = zema_data["ood_target_full"][target_dim[0]]

    # select sensors
    if isinstance(sensor_i, int):
        x_sensor_select = zema_data[resample_factor][:, :, [sensor_i]].copy()
    else:
        x_sensor_select = zema_data[resample_factor][:, :, sensor_i].copy()

    # ==========Train test split===========
    # prepare id and ood
    x_id_train = x_sensor_select[id_args]
    x_ood_test = x_sensor_select[ood_args]

    # move axis(required)
    x_id_train = np.moveaxis(x_id_train, 1, 2)
    x_ood_test = np.moveaxis(x_ood_test, 1, 2)

    # actually split
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=train_size, shuffle=True, random_state=random_seed
    )

    # option to apply fft
    if apply_fft:
        x_id_train = FFT_Sensor().transform(x_id_train)
        x_id_test = FFT_Sensor().transform(x_id_test)
        x_ood_test = FFT_Sensor().transform(x_ood_test)

    # min max scaling
    if standardise == "minmax":
        sensor_scaler = MinMaxSensor(
            num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
        )
    elif standardise == "standard":
        sensor_scaler = StandardiseSensor(num_sensors=x_id_train.shape[1], axis=1)

    # return splits
    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    return x_id_train, x_id_test, x_ood_test


def get_bae_model(
    exp_params,
    x_id_train,
    activation="leakyrelu",
    se_block=False,
    bias=False,
    use_cuda=False,
    lr=0.01,
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
    n_conv_layers = (
        exp_params["n_conv_layers"] if "n_conv_layers" in exp_params.keys() else 1
    )
    n_enc_capacity = (
        exp_params["n_enc_capacity"] if "n_enc_capacity" in exp_params.keys() else 1
    )
    n_conv_filters = n_enc_capacity * 1
    # n_dense_nodes = n_enc_capacity * 100
    n_dense_nodes = 1000

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

    # specify architecture
    chain_params = [
        # {
        #     "base": "conv1d",
        #     "input_dim": input_dim,
        #     "conv_channels": [x_id_train.shape[1], 10, 20],
        #     "conv_stride": [2, 2],
        #     "conv_kernel": [8, 2],
        #     "activation": activation,
        #     "norm": layer_norm,
        #     "se_block": se_block,
        #     "order": ["base", "norm", "activation"],
        #     "bias": bias,
        #     "last_norm": layer_norm,
        # },
        # {
        #     "base": "linear",
        #     "architecture": [1000] * n_dense_layers + [latent_dim],
        #     "activation": activation,
        #     "norm": "none",
        #     "last_norm": "none",
        # },
        {
            "base": "conv1d",
            "input_dim": input_dim,
            "conv_channels": [x_id_train.shape[1], 10]
            + [n_conv_filters] * (n_conv_layers),
            "conv_stride": [2] + [2] * n_conv_layers,
            "conv_kernel": [8] + [2] * n_conv_layers,
            "activation": activation,
            "norm": layer_norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": layer_norm,
        },
        {
            "base": "linear",
            "architecture": [n_dense_nodes] * n_dense_layers + [latent_dim],
            "activation": activation,
            "norm": layer_norm,
            "last_norm": layer_norm,
            "bias": bias,
        },
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
