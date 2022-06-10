# THIS FINDS THE MIN EPOCH FOR CONVERGENCE
# GENERATES PLOTS OF AUROC AND LOSS VS EPOCH

import pandas as pd
import os

import copy
import itertools
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager

bae_set_seed(10)

# data preproc. hyper params
resample_factor = 10
mode = "forging"
pickle_path = "pickles"

zema_data = pickle.load(open(pickle_path + "/" + "zema_hyd_inputs_outputs.p", "rb"))

n_random_seeds = 1
random_seeds = np.random.randint(0, 1000, n_random_seeds)

# fmt: off
full_likelihood = ["mse","homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli", "beta"]
homoscedestic_mode_map = { "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every","hetero-gauss": "none", "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none","beta":"none"}
likelihood_map = { "bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian", "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian", "hetero-tgauss": "truncated_gaussian", "mse": "gaussian","beta":"beta"}
twin_output_map = { "bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True, "homo-tgauss": False,"hetero-tgauss": True, "mse": False,"beta":True}
# fmt: on

min_max_clip = True
use_auto_lr = True
# use_auto_lr = False

# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.

resample_factor = "Hz_1"  # 1Hz

sensors_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]

# sensors_all = [8, 5]  #
sensors_all = [8]  #

# sensors_all = [9, 10, 11, 13, 14]
# sensors_all = [9, 10, 11]
# sensors_all = [11, 14, 13]
# sensors_all = [8, 10, 11]
# sensors_all = [7]

# sensors_all = [8, 5]

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "ss_id": [sensors_all],
#     "target_dim": [0, 1, 2, 3],
#     # "target_dim": [[15]],
#     "resample_factor": [resample_factor],
#     "skip": [False],
#     # "skip": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae", "ens"],
#     "full_likelihood": ["mse", "hetero-gauss", "bernoulli", "cbernoulli"],
# }

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "ss_id": [sensors_all],
#     "target_dim": [0, 1, 2, 3],
#     "resample_factor": [resample_factor],
#     # "skip": [True],
#     "skip": [False],
#     # "layer_norm": [False],
#     # "layer_norm": [True],
#     "layer_norm": [False, True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     # "full_likelihood": ["bernoulli"],
#     "full_likelihood": ["mse", "bernoulli"],
#     # "full_likelihood": ["hetero-gauss"],
# }

# grid = {
#     "random_seed": random_seeds,
#     "apply_fft": [False],
#     "ss_id": [sensors_all],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     # "skip": [True],
#     "skip": [False],
#     "layer_norm": [False],
#     # "layer_norm": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     # "full_likelihood": ["homo-tgauss"],
#     # "full_likelihood": ["bernoulli"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["hetero-tgauss"],
#     # "full_likelihood": ["hetero-gauss"],
# }

# grid = {
#     "random_seed": np.random.randint(0, 1000, size=10),
#     "apply_fft": [False],
#     "ss_id": [8],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     # "skip": [True],
#     "skip": [False],
#     "layer_norm": [True],
#     # "layer_norm": [True],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     # "full_likelihood": ["cbernoulli"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["homo-tgauss"],
#     # "full_likelihood": ["hetero-tgauss"],
#     # "full_likelihood": ["hetero-gauss"],
# }

grid = {
    "random_seed": [1],
    "apply_fft": [False],
    # "apply_fft": [True],
    # "ss_id": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]],
    # "ss_id": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]],
    # "ss_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
    # "ss_id": [[8, 5]],
    # "ss_id": [[9, 0, 2, 1, 3, 11, 7, 10, 13, 8, 4, 5, 6, 12]],
    # "ss_id": [[10, 13, 0, 11]],
    "ss_id": [[10, 5, 7]],
    # "ss_id": [[10]],
    "target_dim": [1],
    # "resample_factor": [resample_factor],
    "resample_factor": ["Hz_1"],
    # "skip": [True],
    "skip": [False],
    # "layer_norm": [True],
    "layer_norm": [False],
    "latent_factor": [0.1],
    "bae_type": ["ae"],
    # "full_likelihood": ["bernoulli"],
    # "full_likelihood": ["cbernoulli"],
    "full_likelihood": ["mse"],
    # "full_likelihood": ["homo-tgauss"],
    # "full_likelihood": ["hetero-tgauss"],
    # "full_likelihood": ["hetero-gauss"],
}

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
    "sghmc": 50,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

check_row_exists = False
exp_name = "ZEMA_HYD_NEW00"

exp_man = ExperimentManager(folder_name="experiments")

all_aurocs = []
all_ae_loss = []

# Loop over all grid search combinations
for values in tqdm(itertools.product(*grid.values())):

    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

    # check for continuity
    # a way to continue progress from before
    # if anything happened and interrupted the flow
    if check_row_exists:
        new_row = pd.DataFrame([exp_params])
        csv_path = os.path.join(exp_man.folder_name, exp_name + "AUROC.csv")
        if os.path.exists(csv_path):
            read_exp_csv = pd.read_csv(csv_path)
            num_columns = len(new_row.columns)
            read_exp_csv_ = read_exp_csv.iloc[:, 1 : num_columns + 1]
            common_row = new_row.merge(read_exp_csv_, "inner")
            common_row = read_exp_csv_.merge(new_row, "inner")
            if len(common_row) > 0:  # row already exist
                print("Row exists, skipping to next iteration...")
                continue

    # unpack exp params
    random_seed = exp_params["random_seed"]
    apply_fft = exp_params["apply_fft"]
    sensor_i = exp_params["ss_id"]
    target_dim = exp_params["target_dim"]
    resample_factor = exp_params["resample_factor"]
    skip = exp_params["skip"]
    latent_factor = exp_params["latent_factor"]
    bae_type = exp_params["bae_type"]
    full_likelihood_i = exp_params["full_likelihood"]
    layer_norm = exp_params["layer_norm"]

    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    n_bae_samples = n_bae_samples_map[bae_type]

    # continue unpacking
    bae_set_seed(random_seed)

    target_dim = [target_dim] if isinstance(target_dim, int) else target_dim

    resample_factor_map = {1: "Hz_1", 10: "Hz_10", 100: "Hz_100"}

    # ood and id
    id_args = zema_data["id_target"][target_dim[0]]
    # ood_args = zema_data["ood_target"][target_dim[0]]
    ood_args = zema_data["ood_target_full"][target_dim[0]]

    # select sensors
    if isinstance(sensor_i, int):
        x_sensor_select = zema_data[resample_factor][:, :, [sensor_i]]
    else:
        x_sensor_select = zema_data[resample_factor][:, :, sensor_i]

    x_id_train = x_sensor_select[id_args]
    x_ood_test = x_sensor_select[ood_args]

    # ==========Train test split===========
    x_id_train = np.moveaxis(x_id_train, 1, 2)
    x_ood_test = np.moveaxis(x_ood_test, 1, 2)

    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=0.70, shuffle=True, random_state=random_seed
    )

    # option to apply fft
    if apply_fft:
        x_id_train = FFT_Sensor().transform(x_id_train)
        x_id_test = FFT_Sensor().transform(x_id_test)
        x_ood_test = FFT_Sensor().transform(x_ood_test)

    # min max
    sensor_scaler = MinMaxSensor(
        num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
    )
    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    # ===============FIT BAE===============

    use_cuda = True
    # weight_decay = 0.0000000001
    weight_decay = 0.00000000001
    # weight_decay = 0.000001
    anchored = True if bae_type == "ens" else False
    bias = False
    se_block = False
    norm = "layer" if layer_norm else "none"
    self_att = False
    self_att_transpose_only = False
    # num_epochs = 500
    # num_epochs = 100
    num_epochs = 50
    activation = "leakyrelu"
    # activation = "elu"
    # activation = "selu"
    # activation = "none"
    lr = 0.01
    # lr = 500
    dropout = {"dropout_rate": 0.01} if bae_type == "mcd" else {}

    input_dim = x_id_train.shape[-1]
    latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)

    chain_params = [
        # {
        #     "base": "conv1d",
        #     "input_dim": input_dim,
        #     "conv_channels": [x_id_train.shape[1], 10, 20],
        #     "conv_stride": [2, 2],
        #     "conv_kernel": [8, 2],
        #     "activation": activation,
        #     "norm": norm,
        #     "se_block": se_block,
        #     "order": ["base", "norm", "activation"],
        #     "bias": bias,
        #     "last_norm": norm,
        # },
        # {
        #     "base": "linear",
        #     "architecture": [1000, latent_dim],
        #     "activation": activation,
        #     "norm": norm,
        #     "last_norm": norm,
        # },
        {
            "base": "conv1d",
            "input_dim": input_dim,
            "conv_channels": [x_id_train.shape[1], 10, 20],
            "conv_stride": [2, 2],
            "conv_kernel": [8, 2],
            # "conv_kernel": [20, 2],
            # "conv_channels": [x_id_train.shape[1], 10, 20],
            # "conv_channels": [x_id_train.shape[1], 10],
            # "conv_stride": [2],
            # "conv_kernel": [8],
            "activation": activation,
            "norm": norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": norm,
        },
        {
            "base": "linear",
            "architecture": [1000, latent_dim],
            # "architecture": [500, 500, latent_dim],
            "activation": activation,
            "norm": norm,
            "last_norm": norm,
        },
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        # last_activation="tanh",
        last_activation="sigmoid",
        # last_activation="none",
        last_norm=norm,
        twin_output=twin_output,
        # twin_params={"activation": "selu", "norm": "none"},
        twin_params={
            "activation": "sigmoid" if full_likelihood_i == "hetero-tgauss" else "selu",
            "norm": "none",
        },
        # twin_params={"activation": "none", "norm": "none"},
        skip=skip,
        use_cuda=use_cuda,
        scaler_enabled=False,
        homoscedestic_mode=homoscedestic_mode,
        likelihood=likelihood,
        weight_decay=weight_decay,
        num_samples=n_bae_samples,
        anchored=anchored,
        learning_rate=lr,
        stochastic_seed=random_seed,
        **dropout,
    )

    x_id_train_loader = convert_dataloader(
        x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
    )
    save_mecha = "copy"

    # In case error occurs
    # Wrap code around
    try:
        if use_auto_lr:
            min_lr, max_lr, half_iter = run_auto_lr_range_v4(
                x_id_train_loader,
                bae_model,
                window_size=1,
                num_epochs=10,
                run_full=False,
                plot=False,
                verbose=False,
                save_mecha=save_mecha,
            )

        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

        # check for num_nans
        num_nans = np.argwhere(np.isnan(bae_model.losses))
        while len(num_nans) > 0:
            bae_model.reset_parameters()
            valid_epoch = num_nans.ravel()[0] // len(x_id_train_loader)
            valid_epoch = np.clip(valid_epoch, a_min=1, a_max=None)
            time_method(bae_model.fit, x_id_train_loader, num_epochs=valid_epoch)
            num_nans = np.argwhere(np.isnan(bae_model.losses))

        (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
            eval_auroc,
            retained_res_all,
            misclas_res_all,
        ) = evaluate_ood_unc(
            bae_model=bae_model,
            x_id_train=x_id_train,
            x_id_test=x_id_test,
            x_ood_test=x_ood_test,
            exp_name=exp_name,
            exp_params=exp_params,
            eval_ood_unc=False,
            exp_man=exp_man,
            ret_flatten_nll=True,
            cdf_dists=["norm", "uniform", "ecdf", "expon"],
            norm_scalings=[True, False],
            eval_bce_se=False,
        )
        print(eval_auroc)

    except Exception as e:
        print(e)
        exp_man.update_csv(
            exp_params=exp_man.concat_params_res(exp_params, {"ERROR MSG": e}),
            csv_name=exp_name + "ERROR.csv",
        )

nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
nll_ood = bae_model.predict(x_ood_test, select_keys=["nll"])["nll"].mean(0).mean(-1)

aurocs_sensors = []
for sensor_i in range(nll_id.shape[-1]):
    aurocs_sensors.append(calc_auroc(nll_id[:, sensor_i], nll_ood[:, sensor_i]))
print(aurocs_sensors)

print("Best sensors:")
print(np.array(aurocs_sensors)[np.argsort(aurocs_sensors)[::-1]])
print(np.argsort(aurocs_sensors)[::-1])

plt.figure()
for trace in x_id_test:
    plt.plot(trace[0], color="tab:blue", alpha=0.1)


plt.figure()
for trace in x_ood_test:
    plt.plot(trace[0], color="tab:blue", alpha=0.1)

# # =================================================
# # ==== PLOT INCREASING LL FOR WORSE CONDITIONS ====
# # =================================================
#
# uniq_labels = np.unique(zema_data["raw_target"][ood_args][:, target_dim])
#
#
# auroc_labels = []
# nll_ood_levels = []
# for label in uniq_labels:
#     select_ood_label = np.argwhere(
#         zema_data["raw_target"][ood_args][:, target_dim] == label
#     )[:, 0]
#     nll_ood_level = nll_ood[select_ood_label].mean(-1)
#     nll_ood_levels.append(nll_ood_level)
#     nll_id_level = nll_id.mean(-1)
#
#     auroc_labels.append(calc_auroc(nll_id_level, nll_ood_level))
#
# print(auroc_labels)
#
# # add all nll
# all_nll = [nll_id_level] + nll_ood_levels
#
# all_nll = [nll[np.argwhere(flag_tukey_fence(nll) == 0)[:, 0]] for nll in all_nll]
#
# nll_scaler = MinMaxScaler().fit(np.concatenate([nll for nll in all_nll]).reshape(-1, 1))
# all_nll = [nll_scaler.transform(nll.reshape(-1, 1)).flatten() for nll in all_nll]
#
# plt.figure()
# plt.boxplot(all_nll, showfliers=False)
#
# plt.figure()
# plt.boxplot(all_nll, showfliers=True)
#
# calc_auroc(all_nll[0], all_nll[-1])
