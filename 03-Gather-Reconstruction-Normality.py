# ============

# Need to collect the reconstructed signals and the original data
# To analyse the residuals behaviour, i.e. are they of Normal distribution?

# ============
# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
import probscale
from scipy import stats

from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.benchmarks import Params_ODDS, Params_Images
from statsmodels.distributions import ECDF

from thesis_experiments.util_analyse import grid_keyval_product

bae_set_seed(100)

import pandas as pd
import os
import copy
import itertools
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch.cuda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4, run_auto_lr_range_v5
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method

from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager
import torch

# =========IMPORT Params for each dataset=========
from case_study import Params_STRATH
from case_study import Params_ZEMA

# ================================================

# GLOBAL experiment parameters
set_scheduler = False  # use cyclic learning scheduler or not
use_auto_lr = False
show_auto_lr_plot = False
check_row_exists = False  # for continuation of grid, in case interrupted
eval_ood_unc = False
eval_test = True
autolr_window_size = 5  # 5 # 1
mean_prior_loss = False
use_cuda = torch.cuda.is_available()
bias = False

# total_mini_epochs = 1  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

# total_mini_epochs = 5  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

total_mini_epochs = 1  # every mini epoch will evaluate AUROC
min_epochs = 5  # minimum epochs to start evaluating mini epochs; mini epochs are counted only after min_epochs

load_custom_grid = False
# load_custom_grid = "grids/STRATH-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ZEMA-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ODDS_exll_errgrid_20220418.p"
# load_custom_grid = "grids/images_ll_incomp_20220420.p"

# Specify selected datasets and exp names
# dataset = "ZEMA"
dataset = "STRATH"
# dataset = "ODDS"

# dataset = "CIFAR"
# dataset = "FashionMNIST"
# dataset = "Images"
# pickle_suffix = "_MSE_XTEST_20220425.p"
# pickle_suffix = "_SVHN_20220426.p"
# pickle_suffix = "_STANDARDISE_20220426.p"
pickle_suffix = None  # not saving any pickle of recon

exp_names = {
    "ZEMA": "ZEMA_TEST11_",
    "STRATH": "STRATH_TEST11_",
    "ODDS": "ODDS_EXLL_ERREPAIR_",
    "CIFAR": "CIFAR_",
    "FashionMNIST": "FashionMNIST_",
    "SVHN": "SVHN_",
    "MNIST": "MNIST_",
    "Images": "Images_LL_INCOMP_",
}

# =================PREPARE DATASETS============

if dataset == "ZEMA":
    zema_data = Params_ZEMA.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "STRATH":
    strath_data = Params_STRATH.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "ODDS":
    odds_data = Params_ODDS.prepare_data(
        pickle_path=os.path.join("benchmarks", "pickles")
    )

# ============================================

tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False

# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
if not load_custom_grid:
    grids_datasets = {
        "ZEMA": Params_ZEMA.grid_ZEMA,
        "STRATH": Params_STRATH.grid_STRATH,
        "ODDS": Params_ODDS.grid_ODDS,
        "Images": Params_Images.grid_Images,
    }  # grids for each dataset
    grid = grids_datasets[dataset]  # select grid based on dataset
    grid_keys = grid.keys()
    grid_list = list(itertools.product(*grid.values()))
else:  # handles loading custom grid
    custom_grid = pickle.load(open(load_custom_grid, "rb"))
    grid_keys = custom_grid["grid_keys"]
    grid_list = custom_grid["grid_list"]
# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))
# ==================OPTIMISE SENSOR SELECTION?====================
grid_dicts = grid_keyval_product(grid)

# check if there is optim params file
if os.path.exists("analysis/" + dataset + "_optim_params.csv"):
    optim_params = pd.read_csv("analysis/" + dataset + "_optim_params.csv")
    for entry in grid_dicts:
        bae_type = entry["bae_type"]
        full_likelihood = entry["full_likelihood"]
        target_dim = entry[tasks_col_name]
        optim_ = optim_params[
            (optim_params["bae_type"] == bae_type)
            & (optim_params["full_likelihood"] == full_likelihood)
            & (optim_params[tasks_col_name] == target_dim)
        ]
        if len(optim_) > 0:
            entry.update({"num_epochs": optim_["current_epoch"].values[0]})

if dataset == "ZEMA" or dataset == "STRATH":
    zema_map_ssid = {
        0: [6, 14, 10, 0, 3, 7],
        1: [10],
        # 2: [8],
        2: [5, 8],
        3: [10, 3, 14],
    }
    strath_map_ssid = {2: [9]}
    ss_id_map = {"ZEMA": zema_map_ssid, "STRATH": strath_map_ssid}

    for entry in grid_dicts:
        target_dim = entry["target_dim"]
        entry.update({"ss_id": ss_id_map[dataset][target_dim]})

# ====================================================================

# instantiate experiment manager
# ready to loop over grid and save results
exp_man = ExperimentManager(folder_name="experiments")
exp_name = exp_names[dataset]

# Results have the following shape:
# target_dim : {"id": np.array of shape (random_seed,bae_samples,n_examples,features),
# "ood": similar}
y_pred_dicts = {}

# Loop over all grid search combinations
# for values in grid_list:
for exp_params_temp in grid_dicts:

    # setup the grid
    # exp_params = dict(zip(grid_keys, values))
    exp_params = exp_params_temp.copy()  # create a copy to be safe
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
            common_row = read_exp_csv_.astype(str).merge(new_row.astype(str), "inner")
            if len(common_row) > 0:  # row already exist
                print("Row exists, skipping to next iteration...")
                continue

    # unpack exp params
    random_seed = exp_params["random_seed"]
    num_epochs = exp_params["num_epochs"]

    # augment exp_params for ZEMA and STRATH to count k sensors being used in experiment results
    if (dataset == "ZEMA") or (dataset == "STRATH"):
        sensor_i = exp_params["ss_id"]
        sensor_i = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
        k_sensors = len(sensor_i)
        exp_params.update({"k_sens": k_sensors})
        print("K sensors: " + str(k_sensors))

    # set random seed
    bae_set_seed(random_seed)

    # ==============PREPARE X_ID AND X_OOD=============
    if dataset == "ZEMA":
        x_id_train, x_id_test, x_ood_test = Params_ZEMA.get_x_splits(
            zema_data, exp_params, min_max_clip=True, train_size=0.70
        )
        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
        )
    elif dataset == "STRATH":
        x_id_train, x_id_test, x_ood_test = Params_STRATH.get_x_splits(
            strath_data, exp_params, min_max_clip=True, train_size=0.70
        )
        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
        )
    elif dataset == "ODDS":
        x_id_train, x_id_test, x_ood_test = Params_ODDS.get_x_splits(
            odds_data, exp_params
        )
        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 3, shuffle=True, drop_last=True
        )
    elif (
        dataset == "CIFAR"
        or dataset == "FashionMNIST"
        or dataset == "MNIST"
        or dataset == "SVHN"
    ):
        exp_params.update({"id_dataset": dataset})
        x_id_train, x_id_test, x_ood_test = Params_Images.get_x_splits(exp_params)
        x_id_train_loader = x_id_train
    elif dataset == "Images":
        x_id_train, x_id_test, x_ood_test = Params_Images.get_x_splits(exp_params)
        x_id_train_loader = x_id_train

    # ===============INSTANTIATE BAE===================
    if dataset == "ZEMA":
        bae_model = Params_ZEMA.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            se_block=False,
            bias=bias,
            use_cuda=use_cuda,
            dropout_rate=0.05,
            mean_prior_loss=mean_prior_loss,
        )
    elif dataset == "STRATH":
        bae_model = Params_STRATH.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            se_block=False,
            bias=bias,
            use_cuda=use_cuda,
            dropout_rate=0.05,
            mean_prior_loss=mean_prior_loss,
        )
    elif dataset == "ODDS":
        bae_model = Params_ODDS.get_bae_model(
            exp_params,
            x_id_train,
            activation="selu",
            bias=bias,
            use_cuda=use_cuda,
            dropout_rate=0.05,
            mean_prior_loss=mean_prior_loss,
        )
    elif (
        dataset == "CIFAR"
        or dataset == "FashionMNIST"
        or dataset == "MNIST"
        or dataset == "SVHN"
    ):
        exp_params.update({"id_dataset": dataset})
        bae_model = Params_Images.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            bias=bias,
            use_cuda=use_cuda,
            dropout_rate=0.05,
            mean_prior_loss=mean_prior_loss,
        )
    elif dataset == "Images":
        bae_model = Params_Images.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            bias=bias,
            use_cuda=use_cuda,
            dropout_rate=0.05,
            mean_prior_loss=mean_prior_loss,
        )

    # ================FIT AND PREDICT BAE===========
    # In case error occurs
    # Wrap code around 'try' and catch exception
    # Error case: continues to next iteration and store the error msg in csv
    bae_set_seed(random_seed)
    save_mecha = "copy"

    try:
        if use_auto_lr:
            min_lr, max_lr, half_iter = run_auto_lr_range_v5(
                x_id_train_loader,
                bae_model,
                window_size=autolr_window_size,
                run_full=False,
                plot=show_auto_lr_plot,
                verbose=False,
                save_mecha=save_mecha,
                set_scheduler=set_scheduler,
                max_allowable_lr=0.001,
            )
        total_epochs = np.copy(exp_params["num_epochs"])

        # no intermediate evaluation as usual
        if total_mini_epochs <= 1:
            min_epochs = total_epochs
            per_epoch = 0
            total_mini_epochs = 1
        else:
            per_epoch = (total_epochs - min_epochs) // (total_mini_epochs - 1)

        # collect best stats
        aurocs_mini = []
        y_preds_mini = []

        for mini_epoch_i in range(total_mini_epochs):
            # to fulfill warm start minimum epochs
            if mini_epoch_i == 0:
                current_epoch = min_epochs
                time_method(
                    bae_model.fit,
                    x_id_train_loader,
                    num_epochs=min_epochs,
                    init_fit=True,
                )
            # per cycle
            else:
                current_epoch = min_epochs + (mini_epoch_i) * per_epoch
                time_method(
                    bae_model.fit,
                    x_id_train_loader,
                    num_epochs=per_epoch,
                    init_fit=False,
                )
            exp_params.update({"current_epoch": current_epoch})
            print("CURRENT EPOCH:" + str(current_epoch))
            if eval_test:
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
                    eval_ood_unc=eval_ood_unc,
                    exp_man=exp_man,
                    ret_flatten_nll=True,
                    cdf_dists=["norm", "uniform", "ecdf", "expon"],
                    norm_scalings=[True, False],
                    eval_bce_se=False,
                )
                print(eval_auroc)

            # start predicting
            # x_id_all = np.concatenate((x_id_train, x_id_test))
            x_id_all = x_id_test
            y_id_mu = bae_model.predict(x_id_all, select_keys=["y_mu"])["y_mu"]
            y_ood_mu = bae_model.predict(x_ood_test, select_keys=["y_mu"])["y_mu"]
            bae_gauss_noise = np.array(
                [
                    (torch.nn.functional.elu(bae_model.autoencoder[i].log_noise) + 1)
                    .detach()
                    .cpu()
                    .numpy()
                    for i in range(bae_model.num_samples)
                ]
            )
            y_preds_mini.append(
                {
                    "y_id_mu": y_id_mu,
                    "y_ood_mu": y_ood_mu,
                    "bae_gauss_noise": bae_gauss_noise,
                }
            )
            aurocs_mini.append(eval_auroc["E_AUROC"])

        print("=========")
    except Exception as e:
        err_msg = type(e).__name__ + ":" + str(e)
        print(err_msg)
        exp_man.update_csv(
            exp_params=exp_man.concat_params_res(exp_params, {"ERROR MSG": err_msg}),
            csv_name=exp_name + "ERROR.csv",
        )

    # Save only the best data
    target_dim = exp_params[tasks_col_name]  # get current target dim

    if target_dim not in y_pred_dicts.keys():  # create new entry if not existed
        y_pred_dicts.update(
            {
                target_dim: {
                    "y_id": [],
                    "y_ood": [],
                    "x_id": [],
                    "x_ood": [],
                    "bae_gauss_noise": [],
                }
            }
        )

    # Get reconstructed data based on the best AUROC from the mini epochs
    best_auroc_arg = np.argmax(np.array(aurocs_mini))
    print("BEST AUROC:" + str(aurocs_mini[best_auroc_arg]))
    best_y_pred = y_preds_mini[best_auroc_arg]
    y_id_mu = best_y_pred["y_id_mu"]
    y_ood_mu = best_y_pred["y_ood_mu"]
    bae_gauss_noise = best_y_pred["bae_gauss_noise"]

    # if images then extract the dataloader
    # iterate data
    if dataset == "Images":
        for dt_loader, x_id_label in zip(
            [x_id_train, x_id_test, x_ood_test],
            ["x_id_train", "x_id_test", "x_ood_test"],
        ):
            # iterate dataloader
            new_data = []
            for batch_idx, (data, target) in tqdm(enumerate(dt_loader)):
                new_data.append(data.cpu().detach().numpy())
            new_data = np.concatenate(new_data, axis=0)

            if x_id_label == "x_id_train":
                # randomly select training batch size 80% from data
                # to create randomness and reduce memory size
                total_examples = len(new_data)
                x_id_train = np.copy(new_data)[
                    np.random.choice(
                        np.arange(total_examples), size=int(total_examples * 0.8)
                    )
                ]
            elif x_id_label == "x_id_test":
                x_id_test = np.copy(new_data)
            elif x_id_label == "x_ood_test":
                x_ood_test = np.copy(new_data)
        x_id_all = x_id_test
    # append predictions
    y_pred_dicts[target_dim]["y_id"].append(y_id_mu.copy())
    y_pred_dicts[target_dim]["y_ood"].append(y_ood_mu.copy())
    y_pred_dicts[target_dim]["x_id"].append(x_id_test.copy())
    y_pred_dicts[target_dim]["x_ood"].append(x_ood_test.copy())
    y_pred_dicts[target_dim]["bae_gauss_noise"].append(bae_gauss_noise.copy())

# =======SAVE PICKLE=========
for target_dim in y_pred_dicts.keys():
    y_pred_dicts[target_dim]["y_id"] = np.array(y_pred_dicts[target_dim]["y_id"])
    y_pred_dicts[target_dim]["y_ood"] = np.array(y_pred_dicts[target_dim]["y_ood"])
    y_pred_dicts[target_dim]["x_id"] = np.array(y_pred_dicts[target_dim]["x_id"])
    y_pred_dicts[target_dim]["x_ood"] = np.array(y_pred_dicts[target_dim]["x_ood"])
    y_pred_dicts[target_dim]["bae_gauss_noise"] = np.array(
        y_pred_dicts[target_dim]["bae_gauss_noise"]
    )

if pickle_suffix is not None:
    pickle_filename = dataset + pickle_suffix
    pickle.dump(y_pred_dicts, open(pickle_filename, "wb"))
    print("Saving pickle as " + pickle_filename)
# ===========================


# ==========TEST=============
id_pred = bae_model.predict(x_id_test, select_keys=["nll", "se"])
ood_pred = bae_model.predict(x_ood_test, select_keys=["nll", "se"])

nll_id = flatten_np(id_pred["nll"].mean(0)).mean(-1)
nll_ood = flatten_np(ood_pred["nll"].mean(0)).mean(-1)

se_id = flatten_np(id_pred["se"].mean(0)).mean(-1)
se_ood = flatten_np(ood_pred["se"].mean(0)).mean(-1)
auroc_nll = calc_auroc(nll_id, nll_ood)
auroc_se = calc_auroc(se_id, se_ood)
print(auroc_nll)
print(auroc_se)

# =============================


# y_id_mu = bae_model.predict(x_id_train, select_keys=["y_mu"])["y_mu"].mean(0)
# y_ood_mu = bae_model.predict(x_ood_test, select_keys=["y_mu"])["y_mu"].mean(0)

#
# noise_level = (
#     (torch.nn.functional.elu(bae_model.autoencoder[0].log_noise) + 1)
#     .detach()
#     .cpu()
#     .numpy()
# )
# # noise_level = 1
# sample_i = 250
# feature_i = 10
#
# x_input = x_id_train[sample_i].reshape(-1)[feature_i]
# y_mu = y_id_mu[sample_i].reshape(-1)[feature_i]
# x_space = np.linspace(0, 1, num=1000)
# # y_pdf = stats.norm.pdf(x_space, y_mu, noise_level[feature_i])
# y_pdf = stats.norm.pdf(x_space, y_mu, 1)
#
# plt.figure()
# plt.plot(x_space, y_pdf, color="tab:blue")
# plt.scatter(x_input, 0, color="tab:blue")
# plt.axvline(x=y_mu, color="tab:blue")
#
# x_input = x_ood_test[sample_i].reshape(-1)[feature_i]
# y_mu = y_ood_mu[sample_i].reshape(-1)[feature_i]
# x_space = np.linspace(0, 1, num=1000)
# # y_pdf = stats.norm.pdf(x_space, y_mu, noise_level[feature_i])
# y_pdf = stats.norm.pdf(x_space, y_mu, 1)
# plt.plot(x_space, y_pdf, color="tab:orange")
# plt.scatter(x_input, 0, color="tab:orange")
# plt.axvline(x=y_mu, color="tab:orange")

#
# # =======GAUSSIAN NOISE?============
# from scipy.stats import truncnorm, pearsonr, shapiro, norm, anderson, kstest
#
# residual_id = flatten_np(y_id_mu) - flatten_np(x_id_train)
# residual_ood = flatten_np(y_ood_mu) - flatten_np(x_ood_test)
#
# plt.figure()
# plt.hist(residual_id.reshape(-1), density=True)
# plt.hist(residual_ood.reshape(-1), density=True)
#
# print(shapiro(residual_id.reshape(-1)))
# print(shapiro(residual_ood.reshape(-1)))
#
#
# # def normality_test(x):
# #     return np.var(x)
#
#
# def normality_test(x):
#     return shapiro(x)
#
#
# # def normality_test(x):
# #     return kstest(x, "norm")
#
#
# test_id = np.apply_along_axis(normality_test, axis=0, arr=residual_id)
# test_ood = np.apply_along_axis(normality_test, axis=0, arr=residual_ood)
#
# print(test_id[0].mean())
# print(test_ood[0].mean())
# print(test_id[0].mean() / test_ood[0].mean())
# print(test_id[1].mean() / test_ood[1].mean())
#
# # ==================================
# from scipy.stats import ks_2samp
#
# ks_res = []
# for feature_i in range(residual_id.shape[1]):
#     ks_res.append(ks_2samp(residual_id[:, feature_i], residual_ood[:, feature_i]))
# ks_res = np.array(ks_res)
#
#
# # ==================================
# import seaborn as sns
#
# plt.figure()
# sns.kdeplot(residual_id.reshape(-1))
# sns.kdeplot(residual_ood.reshape(-1))
#
# # ==================================
#
# import numpy as np
# import statsmodels.api as sm
#
# flattened_res_id = residual_id.reshape(-1)
# flattened_res_ood = residual_ood.reshape(-1)
# z_res_id = (flattened_res_id - flattened_res_id.mean()) / flattened_res_id.std()
# z_res_ood = (flattened_res_ood - flattened_res_ood.mean()) / flattened_res_ood.std()
#
# # plt.figure()
# # res = stats.probplot(z_res_id, plot=plt)
# #
# # plt.figure()
# # res_ = stats.probplot(z_res_ood, plot=plt)
#
# # sm.qqplot(
# #     (flattened_res_id - flattened_res_id.mean()) / flattened_res_id.std(), line="45"
# # )
# # sm.qqplot(
# #     (flattened_res_ood - flattened_res_ood.mean()) / flattened_res_ood.std(), line="45"
# # )
#
# # ===============PROBABILITY PLOT=======================
# flattened_res_id = residual_id.reshape(-1)
# flattened_res_ood = residual_ood.reshape(-1)
# z_res_id = (flattened_res_id - flattened_res_id.mean()) / flattened_res_id.std()
# z_res_ood = (flattened_res_ood - flattened_res_ood.mean()) / flattened_res_ood.std()
#
# x_data_id = z_res_id
# x_data_ood = z_res_ood
#
# empirical_cdf_id = ECDF(x_data_id)(x_data_id)
# theoretical_cdf_id = norm.cdf(x_data_id, loc=0, scale=1)
#
# empirical_cdf_ood = ECDF(x_data_ood)(x_data_ood)
# theoretical_cdf_ood = norm.cdf(x_data_ood, loc=0, scale=1)
#
# factor = 2
# random_args_ood = np.random.choice(
#     np.arange(len(empirical_cdf_ood)),
#     size=len(empirical_cdf_ood) // factor,
#     replace=False,
# )
# random_args_id = np.random.choice(
#     np.arange(len(empirical_cdf_id)),
#     size=len(empirical_cdf_id) // factor,
#     replace=False,
# )
#
# plt.figure()
# plt.plot(theoretical_cdf_id[random_args_id], theoretical_cdf_id[random_args_id])
# plt.scatter(theoretical_cdf_id[random_args_id], empirical_cdf_id[random_args_id])
# plt.scatter(theoretical_cdf_ood[random_args_ood], empirical_cdf_ood[random_args_ood])
#
# # MEAN ABS ERROR
# mae_id = (np.abs(theoretical_cdf_id - empirical_cdf_id)).mean() * 100
# mae_ood = (np.abs(theoretical_cdf_ood - empirical_cdf_ood)).mean() * 100
# print(mae_id)
# print(mae_ood)
#
# # ================PER FEATURE=========================
# feature_i = 20
#
# flattened_res_id = residual_id[:, feature_i]
# flattened_res_ood = residual_ood[:, feature_i]
# z_res_id = (flattened_res_id - flattened_res_id.mean()) / flattened_res_id.std()
# z_res_ood = (flattened_res_ood - flattened_res_ood.mean()) / flattened_res_ood.std()
#
# x_data_id = z_res_id
# x_data_ood = z_res_ood
#
# empirical_cdf_id = ECDF(x_data_id)(x_data_id)
# theoretical_cdf_id = norm.cdf(x_data_id, loc=0, scale=1)
#
# empirical_cdf_ood = ECDF(x_data_ood)(x_data_ood)
# theoretical_cdf_ood = norm.cdf(x_data_ood, loc=0, scale=1)
#
# factor = 1
# random_args_ood = np.random.choice(
#     np.arange(len(empirical_cdf_ood)),
#     size=len(empirical_cdf_ood) // factor,
#     replace=False,
# )
# random_args_id = np.random.choice(
#     np.arange(len(empirical_cdf_id)),
#     size=len(empirical_cdf_id) // factor,
#     replace=False,
# )
#
# plt.figure()
# plt.plot(theoretical_cdf_id[random_args_id], theoretical_cdf_id[random_args_id])
# plt.scatter(theoretical_cdf_id[random_args_id], empirical_cdf_id[random_args_id])
# plt.scatter(theoretical_cdf_ood[random_args_ood], empirical_cdf_ood[random_args_ood])
#
# # MEAN ABS ERROR
# mae_id = (np.abs(theoretical_cdf_id - empirical_cdf_id)).mean() * 100
# mae_ood = (np.abs(theoretical_cdf_ood - empirical_cdf_ood)).mean() * 100
# print(mae_id)
# print(mae_ood)
#
#
# # =======================================================

# import numpy as np
# import scipy.stats as stats
# from matplotlib import scale as mscale
# from matplotlib import transforms as mtransforms
# from matplotlib.ticker import Formatter, Locator
# from statsmodels.distributions import ECDF
#
#
# class PPFScale(mscale.ScaleBase):
#     name = "ppf"
#
#     def __init__(self, axis, **kwargs):
#         mscale.ScaleBase.__init__(self, axis)
#
#     def get_transform(self):
#         return self.PPFTransform()
#
#     def set_default_locators_and_formatters(self, axis):
#         class PercFormatter(Formatter):
#             def __call__(self, x, pos=None):
#                 # \u00b0 : degree symbol
#                 return "%d %%" % (x * 100)
#
#         class PPFLocator(Locator):
#             def __call__(self):
#                 return (
#                     np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]) / 100.0
#                 )
#
#         axis.set_major_locator(PPFLocator())
#         axis.set_major_formatter(PercFormatter())
#         axis.set_minor_formatter(PercFormatter())
#
#     def limit_range_for_scale(self, vmin, vmax, minpos):
#         return max(vmin, 1e-6), min(vmax, 1 - 1e-6)
#
#     class PPFTransform(mtransforms.Transform):
#         input_dims = 1
#         output_dims = 1
#         is_separable = True
#
#         def ___init__(self, thresh):
#             mtransforms.Transform.__init__(self)
#             self.thresh = thresh
#
#         def transform_non_affine(self, a):
#             out = stats.norm.ppf(a)
#             return out
#
#         def inverted(self):
#             return PPFScale.IPPFTransform()
#
#     class IPPFTransform(mtransforms.Transform):
#         input_dims = 1
#         output_dims = 1
#         is_separable = True
#
#         def transform_non_affine(self, a):
#
#             return stats.norm.cdf(a)
#
#         def inverted(self):
#             return PPFScale.PPFTransform()
#
#
# mscale.register_scale(PPFScale)
#
#
# values = z_res_id
# values.sort()
#
# values = z_res_ood
# values.sort()
#
# # calculate empirical CDF
# cumprob = ECDF(values)(values)
#
# # fit data
# loc, scale = stats.norm.fit(values)
# pffit = stats.norm(loc=loc, scale=scale)
#
# x = np.linspace(values.min(), values.max(), 3)
# fig, ax = plt.subplots(1, 1)
# ax.plot(values, cumprob, "go", alpha=0.7, markersize=10)
# ax.plot(x, pffit.cdf(x), "-", label="mean: {:.2f}".format(loc))
# ax.set_yscale("ppf")
# ax.set_ylim(0.01, 0.99)
# ax.grid(True)
# ax.legend(loc=0)
# plt.show()
