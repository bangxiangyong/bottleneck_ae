# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
from scipy.signal import find_peaks

from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.benchmarks import Params_ODDS, Params_Images
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
    apply_along_sensor,
    extract_wt_feats,
    wavedec_mean,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager
import torch
from pprint import pprint

# =========IMPORT Params for each dataset=========
from case_study import Params_STRATH
from case_study import Params_ZEMA

# ================================================

# GLOBAL experiment parameters
set_scheduler = False  # use cyclic learning scheduler or not
# set_scheduler = True  # use cyclic learning scheduler or not
use_auto_lr = True
# use_auto_lr = False
show_auto_lr_plot = False
check_row_exists = False  # for continuation of grid, in case interrupted
eval_ood_unc = False
eval_test = True
autolr_window_size = 5  # 5 # 1
mean_prior_loss = False
use_cuda = torch.cuda.is_available()
# use_cuda = False
bias = False
min_max_clip = True
optim_sensors = True
collect_grads = False

# total_mini_epochs = 0  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

total_mini_epochs = 1  # every mini epoch will evaluate AUROC
min_epochs = 50  # minimum epochs to start evaluating mini epochs; mini epochs are counted only after min_epochs
# train_bae = False
train_bae = True
load_custom_grid = False
# load_custom_grid = "grids/STRATH-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ZEMA-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ODDS_exll_errgrid_20220418.p"
# load_custom_grid = "grids/images_ll_incomp_20220420.p"

# Specify selected datasets and exp names
optim_sensors = False

# Specify selected datasets and exp names
# dataset = "ZEMA"
dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"
exp_theme = "dummyu_"

exp_names = {
    "ZEMA": "ZEMA_" + exp_theme,
    "STRATH": "STRATH_" + exp_theme,
    "ODDS": "ODDS_" + exp_theme,
    "Images": "IMAGES_" + exp_theme,
}
exp_name_prefix = exp_names[dataset]

# override optim sensors if not sensor datasets
if dataset == "ODDS" or dataset == "Images":
    optim_sensors = False
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

# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
if not load_custom_grid:
    grids_datasets = {
        "ZEMA": Params_ZEMA.grid_ZEMA,
        "STRATH": Params_STRATH.grid_STRATH,
        "ODDS": Params_ODDS.grid_ODDS,
        "CIFAR": Params_Images.grid_Images,
        "FashionMNIST": Params_Images.grid_Images,
        "SVHN": Params_Images.grid_Images,
        "MNIST": Params_Images.grid_Images,
        "Images": Params_Images.grid_Images,
    }  # grids for each dataset
    grid = grids_datasets[dataset]  # select grid based on dataset
    grid_keys = grid.keys()
    grid_list = list(itertools.product(*grid.values()))
else:  # handles loading custom grid
    custom_grid = pickle.load(open(load_custom_grid, "rb"))
    grid_keys = custom_grid["grid_keys"]
    grid_list = custom_grid["grid_list"]

# ==================OPTIMISE SENSOR SELECTION====================
if optim_sensors:
    grid_dicts = grid_keyval_product(grid)

    if dataset == "ZEMA" or dataset == "STRATH":
        zema_map_ssid = {
            0: {"bernoulli": [3], "mse": [6, 14]},
            1: {"bernoulli": [5, 8], "mse": [10]},
            2: {"bernoulli": [5], "mse": [5, 8]},
            3: {"bernoulli": [9], "mse": [10, 3]},
        }
        strath_map_ssid = {2: {"bernoulli": [9], "mse": [9, 13]}}
        # strath_map_ssid = {2: {"bernoulli": [9], "mse": [9, 13, 3]}}
        ss_id_map = {"ZEMA": zema_map_ssid, "STRATH": strath_map_ssid}

        # replace grid entry with optimised sensor selection
        for entry in grid_dicts:
            # select target dim
            target_dim = entry["target_dim"]
            likelihood = entry["full_likelihood"]

            # update selected sensors
            entry.update({"ss_id": ss_id_map[dataset][target_dim][likelihood]})
    grid_list = [list(grid_i.values()) for grid_i in grid_dicts]
# ====================================================================

# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))


# instantiate experiment manager
# ready to loop over grid and save results
exp_man = ExperimentManager(folder_name="experiments")
exp_name = exp_names[dataset]

# Loop over all grid search combinations
for values in grid_list:

    # setup the grid
    exp_params = dict(zip(grid_keys, values))
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
            zema_data, exp_params, min_max_clip=min_max_clip, train_size=0.70
        )
        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
        )

    elif dataset == "STRATH":
        # x_id_train, x_id_test, x_ood_test = Params_STRATH.get_x_splits(
        #     strath_data, exp_params, min_max_clip=min_max_clip, train_size=0.70
        # )
        (
            x_id_train,
            x_id_test,
            x_ood_test,
            id_train_args,
            id_test_args,
            ood_args,
            cmm_data_abs_err,
        ) = Params_STRATH.get_x_splits_ADHOC(
            strath_data, exp_params, min_max_clip=min_max_clip, train_size=0.70
        )

        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 1, shuffle=True, drop_last=True
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
    if train_bae:
        if dataset == "ZEMA":
            bae_model = Params_ZEMA.get_bae_model(
                exp_params,
                x_id_train,
                activation=exp_params["activation"]
                if "activation" in exp_params.keys()
                else "leakyrelu",
                se_block=False,
                bias=bias,
                use_cuda=use_cuda,
                dropout_rate=0.05,
                mean_prior_loss=mean_prior_loss,
                collect_grads=collect_grads,
            )
        elif dataset == "STRATH":
            bae_model = Params_STRATH.get_bae_model(
                exp_params,
                x_id_train,
                activation=exp_params["activation"]
                if "activation" in exp_params.keys()
                else "leakyrelu",
                se_block=False,
                bias=bias,
                use_cuda=use_cuda,
                dropout_rate=0.05,
                mean_prior_loss=mean_prior_loss,
                collect_grads=collect_grads,
            )
        elif dataset == "ODDS":
            bae_model = Params_ODDS.get_bae_model(
                exp_params,
                x_id_train,
                activation=exp_params["activation"]
                if "activation" in exp_params.keys()
                else "selu",
                bias=bias,
                use_cuda=use_cuda,
                dropout_rate=0.05,
                mean_prior_loss=mean_prior_loss,
                collect_grads=collect_grads,
            )
        elif dataset == "Images":
            bae_model = Params_Images.get_bae_model(
                exp_params,
                x_id_train,
                activation=exp_params["activation"]
                if "activation" in exp_params.keys()
                else "leakyrelu",
                bias=bias,
                use_cuda=use_cuda,
                dropout_rate=0.05,
                mean_prior_loss=mean_prior_loss,
                collect_grads=collect_grads,
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
                        eval_avgprc,
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
                    print(eval_avgprc)
                    print(
                        "BASELINE AVGPRC:"
                        + str(len(x_ood_test) / (len(x_ood_test) + len(x_id_test)))
                    )
            print("=========")
        except Exception as e:
            err_msg = type(e).__name__ + ":" + str(e)
            print(err_msg)
            exp_man.update_csv(
                exp_params=exp_man.concat_params_res(
                    exp_params, {"ERROR MSG": err_msg}
                ),
                csv_name=exp_name + "ERROR.csv",
            )

# ===== CHECK GRAD FLOW ===== #
all_args = np.concatenate((id_train_args, id_test_args, ood_args))
all_argsort = np.argsort(all_args)

if train_bae:
    if collect_grads:
        plt.figure()
        # layer_grads = bae_model.grads[len(bae_model.grads) // 2 :]
        layer_grads = bae_model.grads
        # layer_grads = np.array(layer_grads)
        # layer_grads = np.array(layer_grads).T

        layer_grads = layer_grads - np.min(layer_grads) / (
            np.max(layer_grads) - np.min(layer_grads)
        )
        # layer_grads = layer_grads.T
        for layer_grad in layer_grads:
            plt.plot(layer_grad, alpha=0.05, color="b")
            plt.hlines(0, 0, len(layer_grad) + 1, linewidth=1, color="k")
        plt.xticks(
            range(0, len(layer_grad), 1), bae_model.layer_names, rotation="vertical"
        )
        plt.xlim(xmin=0, xmax=len(layer_grad))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.tight_layout()

    # =====fill between====
    x_axis = np.arange(len(bae_model.layer_names))
    plt.figure()
    # layer_grads = np.array(bae_model.grads[len(bae_model.grads) // 2 :]).copy()
    layer_grads = np.array(bae_model.grads).copy()

    # mean_grads = layer_grads.mean(0)
    mean_grads = np.percentile(layer_grads, 50, axis=0)
    ub_grads = np.percentile(layer_grads, 75, axis=0)
    lb_grads = np.percentile(layer_grads, 25, axis=0)
    # ub_grads = np.percentile(layer_grads, 95, axis=0)
    # lb_grads = np.percentile(layer_grads, 5, axis=0)

    # scale by ub and lb
    mean_grads = (mean_grads - np.min(lb_grads)) / (np.max(ub_grads) - np.min(lb_grads))
    ub_grads = (ub_grads - np.min(lb_grads)) / (np.max(ub_grads) - np.min(lb_grads))
    lb_grads = (lb_grads - np.min(lb_grads)) / (np.max(ub_grads) - np.min(lb_grads))
    plt.plot(x_axis, mean_grads)
    plt.fill_between(x_axis, ub_grads, lb_grads, alpha=0.20)

    # ====================VIEW SAMPLES=============================
    target_dim = exp_params["target_dim"]
    dt_temp = cmm_data_abs_err[:, target_dim]
    std_threshold_factor = 1.0

    # drop a few?
    # drop_indices = [0, 1]
    # dt_temp = dt_temp[[i for i in range(len(dt_temp)) if i not in drop_indices]]

    Q1 = np.percentile(dt_temp, 25)
    Q3 = np.percentile(dt_temp, 75)
    IQR = Q3 - Q1
    ucl = Q3 + IQR * std_threshold_factor
    lcl = Q1 - IQR * std_threshold_factor
    cl = np.median(dt_temp)  # central line
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
    axes[0].plot(np.arange(len(dt_temp)), dt_temp)
    # axes[0].scatter(np.arange(len(dt_temp)), dt_temp)
    cl_pattern = "--"
    cl_color = "tab:red"
    axes[0].axhline(ucl, linestyle=cl_pattern, color=cl_color)
    # axes[0].axhline(lcl, linestyle=cl_pattern, color=cl_color)
    axes[0].axhline(cl, linestyle=cl_pattern, color="black")
    axes[0].set_title("TUKEY ABS")

    # show which are ids
    x_axis = np.arange(len(dt_temp))

    scat_id_train = axes[0].scatter(
        x_axis[id_train_args], dt_temp[id_train_args], color="tab:green"
    )
    scat_id_test = axes[0].scatter(
        x_axis[id_test_args], dt_temp[id_test_args], color="tab:blue"
    )
    scat_ood_test = axes[0].scatter(
        x_axis[ood_args], dt_temp[ood_args], color="tab:red"
    )
    axes[0].legend(
        [scat_id_train, scat_id_test, scat_ood_test], ["TRAIN", "ID-TEST", "OOD-TEST"]
    )

    # gather  NLL
    # nll_id_train = flatten_np(bae_model.predict(x_id_train)["nll"].mean(0)).mean(-1)
    # nll_id_test = flatten_np(bae_model.predict(x_id_test)["nll"].mean(0)).mean(-1)
    # nll_ood_test = flatten_np(bae_model.predict(x_ood_test)["nll"].mean(0)).mean(-1)
    #
    # nll_sort = np.concatenate((nll_id_train, nll_id_test, nll_ood_test))
    #
    # # plot nll
    # # axes[1].plot(x_axis[all_args][all_argsort], nll_sort)
    # axes[1].plot(all_args[all_argsort], nll_sort[all_argsort])
    #
    # scat_id_train = axes[1].scatter(id_train_args, nll_id_train, color="tab:green")
    # scat_id_test = axes[1].scatter(id_test_args, nll_id_test, color="tab:blue")
    # scat_ood_test = axes[1].scatter(ood_args, nll_ood_test, color="tab:red")
    # axes[1].legend(
    #     [scat_id_train, scat_id_test, scat_ood_test], ["TRAIN", "ID-TEST", "OOD-TEST"]
    # )
    # axes[0].grid()
    # axes[1].grid()
print("Num outliers:")
print(len(x_ood_test))

# ======================PLOT SAMPLES==========================
alpha = 0.35
shift = 0
total_sensors = x_id_train.shape[1]
fig, axes = plt.subplots(total_sensors, 1)
if total_sensors == 1:
    axes = [axes]
for sensor_i in range(total_sensors):
    for trace in x_id_train:
        axes[sensor_i].plot(trace[sensor_i][shift:], color="tab:green", alpha=alpha)
    for trace in x_id_test:
        axes[sensor_i].plot(trace[sensor_i][shift:], color="tab:blue", alpha=alpha)
    for trace in x_ood_test:
        axes[sensor_i].plot(trace[sensor_i], color="tab:red", alpha=alpha)

# ============================================================
id_all = np.concatenate((x_id_train, x_id_test))
ood_all = x_ood_test

x_all = np.concatenate((id_all, ood_all))

alpha = 0.35
total_sensors = x_id_train.shape[1]
fig, axes = plt.subplots(total_sensors, 1)
if total_sensors == 1:
    axes = [axes]
for sensor_i in range(total_sensors):
    axes[sensor_i].plot(all_args[all_argsort], x_all[all_argsort, sensor_i].mean(-1))
    axes[sensor_i].scatter(
        id_train_args, x_id_train[:, sensor_i].mean(-1), color="tab:green"
    )
    axes[sensor_i].scatter(
        id_test_args, x_id_test[:, sensor_i].mean(-1), color="tab:blue"
    )
    axes[sensor_i].scatter(ood_args, x_ood_test[:, sensor_i].mean(-1), color="tab:red")


# =======================CHECK FOR JITTER?===========================
shift = 0
plt.figure()
plt.plot(x_id_train[0][sensor_i][shift:], color="tab:green", alpha=0.5)
plt.plot(x_id_test[0][sensor_i][shift:], color="tab:blue", alpha=0.5)
plt.plot(x_ood_test[0][sensor_i], color="tab:red", alpha=0.5)

# ======================PYOD SIMPLE METHODS=========================
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.iforest import IsolationForest
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

res_clf = {}
for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
    clf_pyod = base_model().fit(flatten_np(x_id_train))
    ascore_id = clf_pyod.decision_function(flatten_np(x_id_test))
    ascore_ood = clf_pyod.decision_function(flatten_np(x_ood_test))

    auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
    res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})

pprint("AUROC PYOD:" + str(res_clf))
# heating_traces[:, :, 9] = heating_traces[:, :, 9] - heating_traces[:, :, 33]

# ======================ATTEMPT FEATURE EXTRACTION==================
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew

# cA,cD
# cA_id, cD_id = pywt.dwt(id_all[0][0], "db1")
# cA_ood, cD_ood = pywt.dwt(ood_all[0][0], "db1")

ood_all_temp = ood_all


# def wavedec_mean(trace, wt_type="haar", level=2, wt_summarise=False):
#     res = pywt.wavedec(trace, wt_type, level=level)
#     if wt_summarise:
#         res = np.array(
#             [[i.mean(), i.var(), kurtosis(i), skew(i), i.max(), i.min()] for i in res]
#         ).flatten()
#     else:
#         res = np.concatenate(res)
#     return np.array(res)
#
#
# def extract_wt_feats(sensor_data, wt_type="haar", wt_level=2, wt_summarise=False):
#     final_data = []
#     for trace in sensor_data:
#         if wt_level == 1:
#             feat = np.array([pywt.dwt(trace_ssid, wt_type) for trace_ssid in trace])
#         else:
#             feat = np.array(
#                 [
#                     wavedec_mean(
#                         trace_ssid,
#                         wt_type=wt_type,
#                         level=wt_level,
#                         wt_summarise=wt_summarise,
#                     )
#                     for trace_ssid in trace
#                 ]
#             )
#         final_data.append(feat)
#     return np.array(final_data)


main_family = "haar"
all_wavelist = pywt.wavelist(main_family)[:3]
wt_level = 3
wt_summarise = True
for wt_type in all_wavelist:
    wt_feats_id_train = extract_wt_feats(
        x_id_train, wt_type=wt_type, wt_level=wt_level, wt_summarise=wt_summarise
    )
    wt_feats_id_test = extract_wt_feats(
        x_id_test, wt_type=wt_type, wt_level=wt_level, wt_summarise=wt_summarise
    )
    wt_feats_ood = extract_wt_feats(
        ood_all_temp, wt_type=wt_type, wt_level=wt_level, wt_summarise=wt_summarise
    )

    res_clf = {}
    for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
        clf_pyod = base_model().fit(flatten_np(wt_feats_id_train))
        ascore_id = clf_pyod.decision_function(flatten_np(wt_feats_id_test))
        ascore_ood = clf_pyod.decision_function(flatten_np(wt_feats_ood))

        auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
        res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})

    pprint("AUROC PYOD DWT:" + str(res_clf))

# plot wt features
plt.figure()
for trace in flatten_np(wt_feats_id_train):
    plt.plot(trace, color="tab:blue")
for trace in flatten_np(wt_feats_ood):
    plt.plot(trace, color="tab:red")


# ===============TOY DWT==========================
# trace_ssid = x_id_train[0, 0]
# trace_ssid = x_ood_test[0, 0]
#
# dwt_res = pywt.wavedec(trace_ssid, wt_type, level=2)
#
# total_dim = 0
# for dwt in dwt_res:
#     total_dim += dwt.flatten().shape[0]

# pywt.downcoef("d", trace_ssid, wt_type, level=8)
#
# plt.figure()
# plt.plot(dwt_res[2])


# =======================SEGMENT FORGE===========================
RNompos_id = 30
ANompos_id = 33
# RNompos_id = -2
# ANompos_id = -1
x_ = strath_data["forging"]
RNompos_id_absdiff = np.abs(np.diff(x_[:, :, RNompos_id], axis=-1))
ANompos_id_absdiff = np.abs(np.diff(x_[:, :, ANompos_id], axis=-1))
# peaksR, _ = find_peaks(RNompos_id_absdiff, height=1)
# peaksA, _ = find_peaks(ANompos_id_absdiff[0], height=1)


def get_peaks(arr_, threshold=1):

    peaks, _ = find_peaks(arr_, height=threshold)
    return peaks


peaksR = np.apply_along_axis(
    func1d=get_peaks,
    axis=1,
    arr=RNompos_id_absdiff,
)
peaksA = np.apply_along_axis(
    func1d=get_peaks, axis=1, arr=ANompos_id_absdiff, threshold=10
)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(x_[0, :, 3], color="black")
ax2.plot(RNompos_id_absdiff[0], color="black")
ax3.plot(ANompos_id_absdiff[0], color="black")

for peak_r in peaksR[0]:
    ax1.axvline(peak_r, linestyle="--", color="red", alpha=0.5)
    ax2.axvline(peak_r, linestyle="--", color="red", alpha=0.5)
for peak_a in peaksA[0]:
    ax1.axvline(peak_a, linestyle="--", color="blue", alpha=0.5)
    ax3.axvline(peak_a, linestyle="--", color="blue", alpha=0.5)


# ====================VISUALISE DWT=========================

wt_type = "haar"
wt_level = 3

sensor_i = -1

trace_id = x_id_train[:, sensor_i]
trace_ood = x_ood_test[:, sensor_i]

# dwt_id = pywt.wavedec(trace_id, wt_type, level=wt_level)
# dwt_ood = pywt.wavedec(trace_ood, wt_type, level=wt_level)

alpha = 0.5
fig, axes = plt.subplots(wt_level + 2, 1)
axes = axes.flatten()
for trace in trace_id:
    dwt_id = pywt.wavedec(trace, wt_type, level=wt_level)
    axes[0].plot(trace, color="tab:blue")
    for level, dwt_ in enumerate(dwt_id):
        axes[level + 1].plot(dwt_, alpha=alpha, color="tab:blue")

for trace in trace_ood:
    dwt_ood = pywt.wavedec(trace, wt_type, level=wt_level)
    axes[0].plot(trace, color="tab:orange")
    for level, dwt_ in enumerate(dwt_ood):
        axes[level + 1].plot(dwt_, alpha=alpha, color="tab:orange")


# ====================BIG PROJECT==========================

# view
trace_id = x_id_test[-1]
trace_ood = x_ood_test[-3]

cwt_id = []
cwt_ood = []
fig, axes = plt.subplots(2, 4)
for row_i, traces in enumerate([trace_id, trace_ood]):
    for col_i, trace_ss in enumerate(traces):
        (cwt_res, _) = pywt.cwt(trace_ss, np.arange(1, len(trace_ss) // 2), "mexh")
        if row_i == 0:
            cwt_id.append(cwt_res)
        else:
            cwt_ood.append(cwt_res)
        axes[row_i][col_i].matshow(cwt_res)
fig.tight_layout()

# =====CONVERT TO CWT=======
cwt_id_train = []
cwt_id_test = []
cwt_ood_test = []

for (x_traces, cwt_res) in zip(
    [x_id_train, x_id_test, x_ood_test], [cwt_id_train, cwt_id_test, cwt_ood_test]
):
    for trace in x_traces:
        cwt_ = [
            pywt.cwt(trace_ss, np.arange(1, len(trace_ss) // 2), "mexh")[0]
            for trace_ss in trace
        ]
        cwt_res.append(cwt_)
cwt_id_train = np.array(cwt_id_train)
cwt_id_test = np.array(cwt_id_test)
cwt_ood_test = np.array(cwt_ood_test)

# ========================
sensor_scaler = MinMaxSensor(
    num_sensors=cwt_id_train.shape[1], axis=1, clip=min_max_clip
)
cwt_id_train = sensor_scaler.fit_transform(cwt_id_train)
cwt_id_test = sensor_scaler.transform(cwt_id_test)
cwt_ood_test = sensor_scaler.transform(cwt_ood_test)

# FIT CLF
res_clf = {}
for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
    clf_pyod = base_model().fit(flatten_np(cwt_id_train))
    ascore_id = clf_pyod.decision_function(flatten_np(cwt_id_test))
    ascore_ood = clf_pyod.decision_function(flatten_np(cwt_ood_test))

    auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
    res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})

pprint("AUROC PYOD CWT:" + str(res_clf))

# =====CONVERT TO DWT WITH MIN MAX SCALING=======
# trace_ss = x_id_train[0, 0]
# temp_dwt = pywt.wavedec(trace_ss, wavelet=wt_type, level=wt_level)
# len_dwt = np.cumsum([len(dwt_) for dwt_ in temp_dwt])

## for all sensors
# dwt_min_max_ss = []
# for sensor_i in range(x_id_train.shape[1]):
#     temp_min_max = []
#     for dwt_i, len_ in enumerate(len_dwt):
#         # fit
#         if dwt_i == 0:
#             trunc_traces = x_id_train[:, sensor_i, :len_]
#         else:
#             trunc_traces = x_id_train[:, sensor_i, len_dwt[dwt_i - 1] : len_]
#         dwt_max = np.max(trunc_traces)
#         dwt_min = np.min(trunc_traces)
#         temp_min_max.append([dwt_min, dwt_max])
#     dwt_min_max_ss.append(temp_min_max)
# dwt_min_max_ss = np.array(dwt_min_max_ss)  # (ss_id, dwt_level, min-max)
#
# x_id_train_temp = x_id_train.copy()
# for sensor_i in range(x_id_train_temp.shape[1]):
#     for dwt_i, len_ in enumerate(len_dwt):
#         # transform
#         dwt_min, dwt_max = dwt_min_max_ss[sensor_i, dwt_i]
#
#         if dwt_i == 0:
#             trunc_traces = x_id_train_temp[:, sensor_i, :len_]
#             x_id_train_temp[:, sensor_i, :len_] = (trunc_traces - dwt_min) / (
#                 dwt_max - dwt_min
#             )
#         else:
#             trunc_traces = x_id_train_temp[:, sensor_i, len_dwt[dwt_i - 1] : len_]
#             x_id_train_temp[:, sensor_i, len_dwt[dwt_i - 1] : len_] = (
#                 trunc_traces - dwt_min
#             ) / (dwt_max - dwt_min)


# x_id_train_ = MinMaxSeqSensor().fit_transform(x_id_train, len_dwt=len_dwt)

# =========================================================


# for trace_id in id_all:
#     cA_id, cD_id = pywt.dwt(trace_id[0], "db1")
#     # wt_feats_id.append(cA_id.mean())
#     wt_feats_id.append(cD_id.mean())
# for trace_ood in ood_all_temp:
#     cA_ood, cD_ood = pywt.dwt(trace_ood[0], "db1")
#     # wt_feats_ood.append(cA_ood.mean())
#     wt_feats_ood.append(cD_ood.mean())
#
# plt.figure()
# plt.boxplot([wt_feats_id, wt_feats_ood])
#
#
# sensor_i = -2
# trace_id = x_id_train[-1, sensor_i]
# # trace = x_ood_test[-1, -1]
# trace_ood = x_ood_test[-5, sensor_i]
#
# vmin = np.concatenate((trace_id, trace_ood)).min()
# vmax = np.concatenate((trace_id, trace_ood)).max()
# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
#
# x_len = x_id_train.shape[-1] // 2 + 1
# # x_len = x_id_train.shape[-1] // 1 + 1
# coef_id, freqs = pywt.cwt(trace_id, np.arange(1, x_len), "gaus1")
# coef_ood, freqs = pywt.cwt(trace_ood, np.arange(1, x_len), "gaus1")
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.matshow(coef_id, vmin=vmin, vmax=vmax)  # doctest: +SKIP
# ax2.matshow(coef_ood, vmin=vmin, vmax=vmax)  # doctest: +SKIP
# # ax1.matshow(coef_id)  # doctest: +SKIP
# # ax2.matshow(coef_ood)  # doctest: +SKIP
# # plt.show()  # doctest: +SKIP
# # plt.colorbar()
#
#
# def apply_cwt(x, *args, **kwargs):
#     res = []
#     for trace_ in x:
#         for sensor in trace_:
#             res.append(pywt.cwt(sensor, *args, **kwargs)[0])
#     return np.array(res)
#
#
# def cwt_single(x, *args, **kwargs):
#     return pywt.cwt(x, *args, **kwargs)[0]
#
#
# # cwt_id_train = apply_cwt(x_id_train, scales=np.arange(1, x_len), wavelet="gaus1")
# # cwt_id_test = apply_cwt(x_id_test, scales=np.arange(1, x_len), wavelet="gaus1")
# # cwt_ood_test = apply_cwt(x_ood_test, scales=np.arange(1, x_len), wavelet="gaus1")
#
# main_family = "gaus"
# all_wavelist = pywt.wavelist(main_family)[:3]
# for cwt_type in all_wavelist:
#     cwt_id_train = apply_along_sensor(
#         sensor_data=x_id_train,
#         func1d=cwt_single,
#         seq_axis=2,
#         scales=np.arange(1, x_len),
#         wavelet=cwt_type,
#     )
#     cwt_id_test = apply_along_sensor(
#         sensor_data=x_id_test,
#         func1d=cwt_single,
#         seq_axis=2,
#         scales=np.arange(1, x_len),
#         wavelet=cwt_type,
#     )
#     cwt_ood_test = apply_along_sensor(
#         sensor_data=x_ood_test,
#         func1d=cwt_single,
#         seq_axis=2,
#         scales=np.arange(1, x_len),
#         wavelet=cwt_type,
#     )
#     res_clf = {}
#     for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
#         clf_pyod = base_model().fit(flatten_np(cwt_id_train))
#         ascore_id = clf_pyod.decision_function(flatten_np(cwt_id_test))
#         ascore_ood = clf_pyod.decision_function(flatten_np(cwt_ood_test))
#
#         auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
#         res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})
#
#     pprint("AUROC PYOD CWT:" + str(res_clf))
#
#
# cwt_id_train_ = cwt_id_train.reshape(cwt_id_train.shape[0], cwt_id_train.shape[1], -1)
# cwt_id_test_ = cwt_id_test.reshape(cwt_id_test.shape[0], cwt_id_test.shape[1], -1)
# cwt_ood_test_ = cwt_ood_test.reshape(cwt_ood_test.shape[0], cwt_ood_test.shape[1], -1)
