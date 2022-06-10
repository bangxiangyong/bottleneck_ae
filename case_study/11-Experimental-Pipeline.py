# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
from baetorch.baetorch.util.seed import bae_set_seed

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
set_scheduler = False
use_auto_lr = True
check_row_exists = False  # for continuation of grid, in case interrupted

use_cuda = torch.cuda.is_available()

# Specify selected datasets and exp names
# dataset = "ZEMA"
dataset = "STRATH"
exp_names = {"ZEMA": "ZEMA_HYD_NEW00", "STRATH": "STRATH_NEW"}

# =================PREPARE DATASETS============
pickle_folder = "pickles"
if dataset == "ZEMA":
    zema_data = Params_ZEMA.prepare_data(pickle_path=pickle_folder)
elif dataset == "STRATH":
    strath_data = Params_STRATH.prepare_data(pickle_path=pickle_folder)
# ============================================

# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
grids_datasets = {
    "ZEMA": Params_ZEMA.grid_ZEMA,
    "STRATH": Params_STRATH.grid_STRATH,
}  # grids for each dataset
grid = grids_datasets[dataset]  # select grid based on dataset
# ============================================

# all_aurocs = []
# all_ae_loss = []
# topk_sensor_aurocs_rep = []
# all_nll_ids = []
# all_nll_oods = []

# instantiate experiment manager
# ready to loop over grid and save results
exp_man = ExperimentManager(folder_name="experiments")
exp_name = exp_names[dataset]

# Loop over all grid search combinations
for values in itertools.product(*grid.values()):

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
    num_epochs = exp_params["num_epochs"]

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
    # ===============INSTANTIATE BAE===================
    if dataset == "ZEMA":
        bae_model = Params_ZEMA.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            se_block=False,
            bias=False,
            use_cuda=use_cuda,
            lr=0.01,
            dropout_rate=0.05,
        )
    elif dataset == "STRATH":
        bae_model = Params_STRATH.get_bae_model(
            exp_params,
            x_id_train,
            activation="leakyrelu",
            se_block=False,
            bias=False,
            use_cuda=use_cuda,
            lr=0.01,
            dropout_rate=0.05,
        )
    save_mecha = "copy"

    # ================FIT AND PREDICT BAE===========
    # In case error occurs
    # Wrap code around 'try' and catch exception
    # Error case: continues to next iteration and store the error msg in csv
    try:
        if use_auto_lr:
            min_lr, max_lr, half_iter = run_auto_lr_range_v5(
                x_id_train_loader,
                bae_model,
                window_size=15,
                num_epochs=10,
                run_full=False,
                plot=True,
                verbose=False,
                save_mecha=save_mecha,
                set_scheduler=set_scheduler,
            )
        time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

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
        err_msg = type(e).__name__ + ":" + str(e)
        print(err_msg)
        exp_man.update_csv(
            exp_params=exp_man.concat_params_res(exp_params, {"ERROR MSG": err_msg}),
            csv_name=exp_name + "ERROR.csv",
        )
    # =============================================
    nll_id = bae_model.predict(x_id_test, select_keys=["nll"])["nll"].mean(0).mean(-1)
    nll_ood = bae_model.predict(x_ood_test, select_keys=["nll"])["nll"].mean(0).mean(-1)

    aurocs_sensors = []
    for sensor_i in range(nll_id.shape[-1]):
        aurocs_sensors.append(calc_auroc(nll_id[:, sensor_i], nll_ood[:, sensor_i]))
    print(aurocs_sensors)

    print("Best sensors:")
    print(np.array(aurocs_sensors)[np.argsort(aurocs_sensors)[::-1]])
    print(np.argsort(aurocs_sensors)[::-1])


# all_aurocs_ranking = [scipy.stats.rankdata(aurocs) for aurocs in all_aurocs]
# all_aurocs_ranking = np.mean(all_aurocs_ranking, 0)
# all_aurocs_ranking_ = scipy.stats.rankdata(all_aurocs_ranking)
#
# sensor_importances = np.argsort(all_aurocs_ranking_)[
#     ::-1
# ]  # sensor importances by AUROC ranking
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
