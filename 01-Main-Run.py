# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.seed import bae_set_seed
from benchmarks import Params_ODDS, Params_Images
from util_analyse import grid_keyval_product
from util.add_noise import add_noise

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
from util.evaluate_ood_unc import evaluate_ood_unc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager
import torch

# plt.rcParams.update({"font.size": 15})

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
# show_auto_lr_plot = True
check_row_exists = False  # for continuation of grid, in case interrupted
# check_row_exists = True
eval_ood_unc = False
eval_test = True
autolr_window_size = 5  # 5 # 1
# mean_prior_loss = True
mean_prior_loss = False
use_cuda = torch.cuda.is_available()
# use_cuda = False
bias = False
min_max_clip = True
collect_grads = False

eval_noise = True
# eval_noise = False

# total_mini_epochs = 0  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

total_mini_epochs = 1  # every mini epoch will evaluate AUROC
min_epochs = 100  # minimum epochs to start evaluating mini epochs; mini epochs are counted only after min_epochs

load_custom_grid = False
# load_custom_grid = "grids/STRATH-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ZEMA-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ODDS_exll_errgrid_20220418.p"
# load_custom_grid = "grids/images_ll_incomp_20220420.p"
# load_custom_grid = "grids/STRATH-noisy.p"
# load_custom_grid = "grids/Images-noisy.p"
# load_custom_grid = "grids/ZEMA-noisy-v2.p"
# load_custom_grid = "grids/ODDS-246-nonoise+bae.p"

# Specify selected datasets and exp names
# optim_sensors = False
optim_sensors = True

# Specify selected datasets and exp names
# dataset = "ZEMA"
dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"

exp_theme = "_NOISY_V3"

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
    # if load_custom_grid:
    grid_dicts = [dict(zip(grid_keys, grid_i)) for grid_i in grid_list]
    # else:
    #     grid_dicts = grid_keyval_product(grid)

    if dataset == "ZEMA" or dataset == "STRATH":
        zema_map_ssid = {
            0: {"bernoulli": [3], "mse": [6, 14]},
            1: {"bernoulli": [5, 8], "mse": [10]},
            2: {"bernoulli": [5], "mse": [5, 8]},
            3: {"bernoulli": [9], "mse": [10, 3]},
        }
        strath_map_ssid = {2: {"bernoulli": [9], "mse": [9]}}
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

        # test for robustness to noise
        if "noise_type" in exp_params.keys() or "noise_scale" in exp_params.keys():
            x_id_train, x_id_test, x_ood_test = add_noise(
                x_id_train,
                x_id_test,
                x_ood_test,
                noise_type=exp_params["noise_type"],
                noise_scale=exp_params["noise_scale"],
            )

        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
        )

    elif dataset == "STRATH":
        x_id_train, x_id_test, x_ood_test = Params_STRATH.get_x_splits(
            strath_data,
            exp_params,
            min_max_clip=min_max_clip,
            train_size=0.70,
            tukey_threshold=1.5,
            tukey_adjusted=False,
            apply_dwt=False,
        )

        # test for robustness to noise
        if "noise_type" in exp_params.keys() or "noise_scale" in exp_params.keys():
            x_id_train, x_id_test, x_ood_test = add_noise(
                x_id_train,
                x_id_test,
                x_ood_test,
                noise_type=exp_params["noise_type"],
                noise_scale=exp_params["noise_scale"],
            )

        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 5, shuffle=True, drop_last=True
        )
    elif dataset == "ODDS":
        x_id_train, x_id_test, x_ood_test = Params_ODDS.get_x_splits(
            odds_data, exp_params
        )

        # test for robustness to noise
        if "noise_type" in exp_params.keys() or "noise_scale" in exp_params.keys():
            x_id_train, x_id_test, x_ood_test = add_noise(
                x_id_train,
                x_id_test,
                x_ood_test,
                noise_type=exp_params["noise_type"],
                noise_scale=exp_params["noise_scale"],
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

        # x_id_train, x_id_test, x_ood_test = Params_Images.get_x_splits_v2(exp_params)
        # x_id_train_loader = x_id_train
    # ===============INSTANTIATE BAE===================
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
            else "leakyrelu",
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

    if use_auto_lr:
        try:
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
                # savefile="lr-ex.png",
            )
        except Exception as e_lr:
            bae_model.set_learning_rate(0.001)
            print(e_lr)
    total_epochs = np.copy(exp_params["num_epochs"])
    try:
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
        print("=========")
    except Exception as e:
        err_msg = type(e).__name__ + ":" + str(e)
        print(err_msg)
        exp_man.update_csv(
            exp_params=exp_man.concat_params_res(exp_params, {"ERROR MSG": err_msg}),
            csv_name=exp_name + "ERROR.csv",
        )
    # set noise
    if eval_noise:
        bae_set_seed(exp_params["random_seed"])
        noise_scales = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        noise_res = {}
        for noise_type in ["normal", "uniform"]:
            noise_aurocs_ = []
            for noise_scale in noise_scales:
                noise_aurocs_rep = []
                exp_params.update(
                    {"noise_scale": noise_scale, "noise_type": noise_type}
                )
                for rep in range(3):
                    if dataset == "Images":
                        (
                            _,
                            x_id_test_noisy,
                            x_ood_test_noisy,
                        ) = Params_Images.get_x_splits(exp_params)
                    else:
                        x_id_test_noisy, x_ood_test_noisy = add_noise(
                            x_id_test.copy(),
                            x_ood_test.copy(),
                            noise_scale=noise_scale,
                            noise_type=noise_type,
                        )
                    nll_id_noisy = flatten_np(
                        bae_model.predict(x_id_test_noisy, select_keys=["nll"])[
                            "nll"
                        ].mean(0)
                    ).mean(-1)
                    nll_ood_noisy = flatten_np(
                        bae_model.predict(x_ood_test_noisy, select_keys=["nll"])[
                            "nll"
                        ].mean(0)
                    ).mean(-1)
                    auroc_noisy = calc_auroc(nll_id_noisy, nll_ood_noisy)
                    noise_aurocs_rep.append(auroc_noisy)
                noise_aurocs_.append(np.mean(noise_aurocs_rep))
                print("NOISE:" + str(noise_scale) + " AUROC:" + str(auroc_noisy))
                exp_man.update_csv(
                    exp_params=exp_man.concat_params_res(
                        exp_params,
                        {
                            "E_AUROC": noise_aurocs_[-1],
                        },
                    ),
                    csv_name=exp_name + "NOISE_TEST.csv",
                )
        noise_res.update({noise_type: noise_aurocs_})


# ======PLOT NOISE=========
plt.figure()
plt.plot(noise_res["uniform"])

# ===== CHECK GRAD FLOW ===== #
print("NUM OUTLIERS:" + str(len(x_ood_test)))
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
    plt.xticks(range(0, len(layer_grad), 1), bae_model.layer_names, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(layer_grad))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()

# =====fill between====
if collect_grads:
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

    # scale by mean only
    # mean_grads_ = (mean_grads - np.min(mean_grads)) / (
    #     np.max(mean_grads) - np.min(mean_grads)
    # )
    # ub_grads_ = (ub_grads - np.min(mean_grads)) / (np.max(mean_grads) - np.min(mean_grads))
    # lb_grads_ = (lb_grads - np.min(mean_grads)) / (np.max(mean_grads) - np.min(mean_grads))
    # ub_grads_ = np.clip(ub_grads_, 0, 1)
    # lb_grads_ = np.clip(lb_grads_, 0, 1)
    # plt.plot(x_axis, mean_grads_)
    # plt.fill_between(x_axis, ub_grads_, lb_grads_, alpha=0.1)


# =============================================
# nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# nll_ood = bae_model.predict(x_ood_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# nll_ood = bae_model.predict(next(iter(x_ood_test))[0], select_keys=["nll"])["nll"]
# nll_id = bae_model.predict(next(iter(x_id_test))[0], select_keys=["nll"])["nll"]
#
# y_mu_id = bae_model.predict(next(iter(x_id_test))[0], select_keys=["y_mu"])[
#     "y_mu"
# ].mean(0)
# y_mu_ood = bae_model.predict(next(iter(x_ood_test))[0], select_keys=["y_mu"])[
#     "y_mu"
# ].mean(0)
# plt.figure()
# # plt.imshow(np.moveaxis(y_mu_ood[-1], 0, 2))
# plt.imshow(np.moveaxis(y_mu_id[0], 0, 2))
#
# # bae_model.fit(next(iter(x_id_train_loader))[0])
# #
# # for data, target in x_id_train_loader:
# #     print(data.shape)
#
# # bae_model.fit(x_id_train_loader, num_epochs=1)
# # bae_model.fit(next(iter(x_id_train_loader))[0][:5])
# nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# nll_ood = bae_model.predict(x_ood_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# aurocs_sensors = []
# for sensor_i in range(nll_id.shape[-1]):
#     aurocs_sensors.append(calc_auroc(nll_id[:, sensor_i], nll_ood[:, sensor_i]))
# print(aurocs_sensors)
#
# print("Best sensors:")
# print(np.array(aurocs_sensors)[np.argsort(aurocs_sensors)[::-1]])
# print(np.array(exp_params["ss_id"])[np.argsort(aurocs_sensors)[::-1]])
#

# all_aurocs_ranking = [scipy.stats.rankdata(aurocs) for aurocs in all_aurocs]
# all_aurocs_ranking = np.mean(all_aurocs_ranking, 0)
# all_aurocs_ranking_ = scipy.stats.rankdata(all_aurocs_ranking)
#
# sensor_importances = np.argsort(all_aurocs_ranking_)[
#     ::-1
# ]  # sensor importances by AUROC ranking
#
# # sensor_importances = [10, 3, 4, 11, 13, 7, 0, 1, 2, 6, 8, 9, 5, 12]  # coalition
# # sensor_importances = [5, 8, 10, 11, 12, 9, 3, 4, 2, 1, 0, 6, 13, 7]

# # ============APPLY SENSOR SELECTION ON NLL=======
# print(sensor_importances)
# aurocs_sensor_selected = []
# for rep_i in range(len(all_nll_ids)):
#     topk_sensor_aurocs = []
#     for i in range(len(sensor_importances)):
#         topk_sensors = np.array(sensor_importances)[np.arange(0, i + 1)]
#         topk_sensor_auroc = calc_auroc(
#             all_nll_ids[rep_i][:, topk_sensors].mean(-1),
#             all_nll_oods[rep_i][:, topk_sensors].mean(-1),
#         )
#         topk_sensor_aurocs.append(topk_sensor_auroc)
#     aurocs_sensor_selected.append(topk_sensor_aurocs)
#
#     # aurocs_sensor_selected.append(
#     #     calc_auroc(all_nll_ids[rep_i].mean(-1), all_nll_oods[rep_i])
#     # )
#

# ============PLOTTING TOPK AUROCS================
# topk_aurocs_mean = np.mean(topk_sensor_aurocs_rep, 0)
# topk_aurocs_std = np.std(topk_sensor_aurocs_rep, 0)

# topk_aurocs_mean = np.mean(aurocs_sensor_selected, 0)
# topk_aurocs_std = scipy.stats.sem(aurocs_sensor_selected, 0)
#
# print("BASELINE AUROC:" + str(topk_aurocs_mean[-1]))
# print("BEST AUROC:" + str(np.max(topk_aurocs_mean)))
# print("BEST TOPK:" + str(np.argmax(topk_aurocs_mean) + 1))
#
# plt.figure()
# plt.plot(topk_aurocs_mean)
# plt.fill_between(
#     np.arange(len(topk_aurocs_mean)),
#     topk_aurocs_mean + topk_aurocs_std,
#     topk_aurocs_mean - topk_aurocs_std,
#     alpha=0.5,
# )
#
#
# # all_aurocs_ranking = [np.argsort(np.argsort(aurocs)) for aurocs in all_aurocs]
# # plt.figure()
# # plt.plot(topk_sensor_aurocs)
# # ===============================================
#
# # order = array.argsort()
# # ranks = order.argsort()
#
# # consider from top N sensors only
# # avg_rank for each sensors
#
# # nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# # nll_ood = bae_model.predict(x_ood_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
# #
# # aurocs_sensors = []
# # for sensor_i in range(nll_id.shape[-1]):
# #     aurocs_sensors.append(calc_auroc(nll_id[:, sensor_i], nll_ood[:, sensor_i]))
# # print(aurocs_sensors)
# #
# # print("Best sensors:")
# # print(np.array(aurocs_sensors)[np.argsort(aurocs_sensors)[::-1]])
# # print(np.argsort(aurocs_sensors)[::-1])
#
# # ====================================================
# # ====================================================
# fig, axes = plt.subplots(2, 1)
# sensor_i = 1
# for trace in x_id_test:
#     axes[0].plot(trace[sensor_i], color="tab:blue", alpha=0.1)
# for trace in x_ood_test:
#     axes[1].plot(trace[sensor_i], color="tab:orange", alpha=0.1)

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

# plt.figure()
# plt.plot(bae_model.losses)
