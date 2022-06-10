# THIS FINDS THE MIN EPOCH FOR CONVERGENCE
# GENERATES PLOTS OF AUROC AND LOSS VS EPOCH


import copy
import itertools
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
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
import seaborn as sns

# bae_set_seed(67)
bae_set_seed(666)
# bae_set_seed(12548)
# bae_set_seed(7)
# bae_set_seed(100)
# bae_set_seed(97)

# data preproc. hyper params
resample_factor = 10
mode = "forging"
pickle_path = "pickles"

# tukey_threshold = 1.5
# tukey_threshold = 2.5
# tukey_threshold = 2
tukey_threshold = 1.5
# tukey_threshold = 2.0

target_dims_all = [1, 2, 7, 9, 12, 17]

# heating_traces = pickle.load(open(pickle_path + "/" + "heating_inputs.p", "rb"))
# forging_traces = pickle.load(open(pickle_path + "/" + "forging_inputs.p", "rb"))
# column_names = pickle.load(open(pickle_path + "/" + "column_names.p", "rb"))
# cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v2.p", "rb")).values
# cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v1.p", "rb")).values
# cmm_data = np.abs(cmm_data)


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
# tukey_adjusted = True
tukey_adjusted = False

# use_auto_lr = False

# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.

# 11, 13, 54
# chosen_forging_sensors = np.array([2, 25, 11, 13, 54, 71, 82, 72])
# resample_factor = 100
resample_factor = "Hz_1"  # 1Hz
# h_sensor = [62, 63, 69, 70, 78, 75, 76, 77]
# sensors_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
sensors_all = [8, 5]

grid = {
    "random_seed": random_seeds,
    "apply_fft": [False],
    # "apply_fft": [True],
    # "mode": ["forging"],
    # "mode": ["forging"],
    # FORGING
    # "ss_id": [[13]],
    # "ss_id": [25],
    # "ss_id": [[71]],
    # "ss_id": [[13, 25, 71]],
    # "ss_id": [2, 25, 11, 13, 54, 71, 82, 72],
    # "ss_id": [[2, 25, 11, 13, 54, 71, 82, 72]],
    # "ss_id": [[1]],
    # "ss_id": [sensors_all],
    "ss_id": [[8, 5]],
    # "ss_id": [[11, 13, 71]],
    # "ss_id": [[13]],
    # "ss_id": [2, 25, 11, 13, 54, 71, 82, 72],
    # "ss_id": [25],
    # HEATING
    # "ss_id": [76],
    # "ss_id": [[62, 63, 69, 70, 78, 76, 77]],
    # "ss_id": [62],
    # "ss_id": [77],
    # "target_dim": [2],
    "target_dim": [2],
    # "target_dim": [[15]],
    "resample_factor": [resample_factor],
    "skip": [False],
    # "skip": [True],
    "latent_factor": [0.5],
    "bae_type": ["ae"],
    # "full_likelihood": ["hetero-gauss"],
    "full_likelihood": ["bernoulli"],
    # "full_likelihood": ["mse"],
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

exp_name = "ZEMA_HYD_SENSORS_"

exp_man = ExperimentManager(folder_name="experiments")

all_aurocs = []
all_ae_loss = []

# Loop over all grid search combinations
for values in tqdm(itertools.product(*grid.values())):

    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    print(exp_params)

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
    anchored = True if bae_type == "ens" else False
    bias = False
    se_block = False
    # norm = "layer"
    norm = "none"
    self_att = False
    self_att_transpose_only = False
    num_epochs = 100
    # num_epochs = 100
    activation = "leakyrelu"
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
            # "conv_channels": [x_id_train.shape[1], 10, 20],
            # "conv_stride": [2, 2],
            # "conv_kernel": [8, 2],
            "conv_channels": [x_id_train.shape[1], 10, 20],
            "conv_stride": [2, 2],
            "conv_kernel": [8, 2],
            "activation": activation,
            "norm": norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": norm,
        },
        {
            "base": "linear",
            "architecture": [500, latent_dim],
            "activation": activation,
            "norm": norm,
            "last_norm": norm,
        },
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid",
        # last_activation="none",
        last_norm=norm,
        twin_output=twin_output,
        twin_params={"activation": "selu", "norm": "none"},
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
    save_mecha = "copy" if (bae_type == "vae" or bae_type == "mcd") else "file"

    time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

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

    # check for num_nans
    num_nans = np.argwhere(np.isnan(bae_model.losses))
    while len(num_nans) > 0:
        bae_model.reset_parameters()
        valid_epoch = num_nans.ravel()[0] // len(x_id_train_loader)
        valid_epoch = np.clip(valid_epoch, a_min=1, a_max=None)
        time_method(bae_model.fit, x_id_train_loader, num_epochs=valid_epoch)
        num_nans = np.argwhere(np.isnan(bae_model.losses))

    # (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
    #     eval_auroc,
    #     retained_res_all,
    #     misclas_res_all,
    # ) = evaluate_ood_unc(
    #     bae_model=bae_model,
    #     x_id_train=x_id_train,
    #     x_id_test=x_id_test,
    #     x_ood_test=x_ood_test,
    #     exp_name=exp_name,
    #     exp_params=exp_params,
    #     eval_ood_unc=False,
    #     exp_man=exp_man,
    #     ret_flatten_nll=True,
    #     cdf_dists=["norm", "uniform", "ecdf", "expon"],
    #     norm_scalings=[True, False],
    # )

    # ====================


def calc_perc_minmax(trace, upper_threshold=0.75, lower_threshold=0.25):
    trace_ = trace
    perc_minmax = len(
        np.argwhere((trace_ >= upper_threshold) | (trace_ <= lower_threshold))
    ) / len(trace_)
    return perc_minmax


sensor_i = 0
# target_dim = 1

percs_id = np.array([calc_perc_minmax(trace[sensor_i]) for trace in x_id_test])
percs_ood = np.array([calc_perc_minmax(trace[sensor_i]) for trace in x_ood_test])


nll_id = (
    bae_model.predict(x_id_test, select_keys=["nll"])["nll"]
    .mean(0)
    .mean(-1)[:, sensor_i]
)
nll_ood = (
    bae_model.predict(x_ood_test, select_keys=["nll"])["nll"]
    .mean(0)
    .mean(-1)[:, sensor_i]
)

plt.figure()
plt.scatter(percs_id, nll_id)
plt.scatter(percs_ood, nll_ood)
plt.xlabel("Proportion of zeros")
plt.ylabel("NLL")
plt.legend(["ID", "OOD"])

plt.figure()
sns.kdeplot(nll_id)
sns.kdeplot(nll_ood)
plt.legend(["Healthy", "Faulty"])
plt.xlabel("NLL")

plt.figure()
sns.kdeplot(percs_id)
sns.kdeplot(percs_ood)
plt.legend(["Healthy", "Faulty"])
plt.xlabel("Proportion of min and max values")

print(calc_auroc(nll_id, nll_ood))
pear_corr = pearsonr(
    np.concatenate((percs_id, percs_ood)), np.concatenate((nll_id, nll_ood))
)
print(pear_corr)

# ====

plt.figure()
sns.kdeplot(x_id_test.flatten())
sns.kdeplot(x_ood_test.flatten())
plt.xlabel("Values")

# # plt.figure()
# # plt.hist(x_id_test.flatten(), density=True)
# # plt.hist(x_ood_test.flatten(), density=True)
#
# plt.figure()
# plt.plot(x_id_test[1, 0])
# plt.plot(x_ood_test[-2, 0])
#
# plt.figure()
# plt.plot(x_id_test[1, 0])
# plt.plot(x_ood_test[-2, 0])
#
# # plt.figure()
# # plt.plot(nll_id[:, 0])
# # plt.plot(nll_ood[:, 0])
#
# nll_id_sensor = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0)[1, 0]
# nll_ood_sensor = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0)[-2, 0]
#
# plt.figure()
# plt.plot(nll_id_sensor)
# plt.plot(nll_ood_sensor)


# ===========

print("PERC ID:" + str(np.mean(percs_id)))
print("PERC OOD:" + str(np.mean(percs_ood)))


# ===========
sample_i = 1
sensor_i = 0
bae_pred_train = bae_model.predict(x_id_train, select_keys=["y_mu", "nll"])
bae_pred_id = bae_model.predict(x_id_test, select_keys=["y_mu", "nll"])
bae_pred_ood = bae_model.predict(x_ood_test, select_keys=["y_mu", "nll"])

plt.figure()
plt.plot(x_id_test[sample_i, sensor_i])
plt.plot(bae_pred_id["y_mu"].mean(0)[sample_i, sensor_i])
plt.legend(["Input", "Reconstructed"])

# plt.figure()
# plt.plot(x_id_train[sample_i, sensor_i])
# plt.plot(bae_pred_train["y_mu"].mean(0)[sample_i, sensor_i])
# plt.legend(["Input", "Reconstructed"])
# plt.title("ID")

plt.figure()
plt.plot(bae_pred_id["nll"].mean(0)[sample_i, sensor_i])
plt.title("ID-NLL")

plt.figure()
plt.plot(x_ood_test[sample_i, sensor_i])
plt.plot(bae_pred_ood["y_mu"].mean(0)[sample_i, sensor_i])
plt.legend(["Input", "Reconstructed"])
plt.title("OOD")

plt.figure()
plt.plot(bae_pred_ood["nll"].mean(0)[sample_i, sensor_i])
plt.title("OOD-NLL")


# plt.figure()
# plt.plot(nll_id_sensor)
# plt.plot(nll_ood_sensor)

# plot repeatedly
plt.figure()
for trace in x_id_train:
    plt.plot(trace[0], color="tab:blue")


# test
# perturbed_ood_test = np.random.uniform(0.1, 0.25, size=x_id_test.shape)
p = 0.1
random_mask = np.random.choice(a=[0, 1], size=x_id_test.shape, p=[p, 1 - p])
perturbed_ood_test = x_id_test * random_mask
nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0)
nll_ood = bae_model.predict(perturbed_ood_test, select_keys=["nll"])["nll"].mean(0)
plt.figure()
plt.scatter(x_id_test.flatten(), nll_id.flatten())
plt.scatter(perturbed_ood_test.flatten(), nll_ood.flatten())

print(calc_auroc(nll_id.mean(-1).mean(-1), nll_ood.mean(-1).mean(-1)))
