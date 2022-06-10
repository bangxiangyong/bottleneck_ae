# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)

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
bias = False
min_max_clip = True
optim_sensors = True
collect_grads = True

# total_mini_epochs = 0  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

total_mini_epochs = 2  # every mini epoch will evaluate AUROC
min_epochs = 1  # minimum epochs to start evaluating mini epochs; mini epochs are counted only after min_epochs
train_bae = False
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
exp_theme = "EFFECTIVE_CAP_OVERPARAM2JJ"

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
    print("Num outliers:")
    print(len(x_ood_test))

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
    nll_id_train = flatten_np(bae_model.predict(x_id_train)["nll"].mean(0)).mean(-1)
    nll_id_test = flatten_np(bae_model.predict(x_id_test)["nll"].mean(0)).mean(-1)
    nll_ood_test = flatten_np(bae_model.predict(x_ood_test)["nll"].mean(0)).mean(-1)

    nll_sort = np.concatenate((nll_id_train, nll_id_test, nll_ood_test))

    # plot nll
    # axes[1].plot(x_axis[all_args][all_argsort], nll_sort)
    axes[1].plot(all_args[all_argsort], nll_sort[all_argsort])

    scat_id_train = axes[1].scatter(id_train_args, nll_id_train, color="tab:green")
    scat_id_test = axes[1].scatter(id_test_args, nll_id_test, color="tab:blue")
    scat_ood_test = axes[1].scatter(ood_args, nll_ood_test, color="tab:red")
    axes[1].legend(
        [scat_id_train, scat_id_test, scat_ood_test], ["TRAIN", "ID-TEST", "OOD-TEST"]
    )
    axes[0].grid()
    axes[1].grid()
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

# ood_all_temp = np.concatenate((id_all, ood_all))[all_argsort][[60, 61, 62]]
ood_all_temp = ood_all


def wavedec_mean(trace, wt_type="db1", level=2):
    res = pywt.wavedec(trace, wt_type, level=level)
    res = [
        [
            i.mean(),
            i.var(),
            kurtosis(i),
            skew(i),
        ]
        for i in res
    ]
    return np.array(res)


def extract_wt_feats(sensor_data, wt_type="db1", level=1):
    final_data = []
    for trace in sensor_data:
        if level == 1:
            feat = np.array([pywt.dwt(trace_ssid, wt_type) for trace_ssid in trace])
        else:
            feat = np.array(
                [
                    wavedec_mean(trace_ssid, wt_type=wt_type, level=level)
                    for trace_ssid in trace
                ]
            )
        final_data.append(feat)
    return np.array(final_data)


main_family = "haar"
all_wavelist = pywt.wavelist(main_family)[:3]
level = 2
for wt_type in all_wavelist:
    wt_feats_id_train = extract_wt_feats(x_id_train, wt_type=wt_type, level=level)
    wt_feats_id_test = extract_wt_feats(x_id_test, wt_type=wt_type, level=level)
    wt_feats_ood = extract_wt_feats(ood_all_temp, wt_type=wt_type, level=level)

    res_clf = {}
    for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
        clf_pyod = base_model().fit(flatten_np(wt_feats_id_train))
        ascore_id = clf_pyod.decision_function(flatten_np(wt_feats_id_test))
        ascore_ood = clf_pyod.decision_function(flatten_np(wt_feats_ood))

        auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
        res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})

    pprint("AUROC PYOD WT:" + str(res_clf))

# ==================================================================
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
