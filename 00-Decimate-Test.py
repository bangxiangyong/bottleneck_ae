# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
import matplotlib.pyplot as plt
from scipy.signal import decimate

from baetorch.baetorch.util.seed import bae_set_seed
from benchmarks import Params_ODDS, Params_Images
from util.evaluate_ood_unc import evaluate_ood_unc
from util_analyse import grid_keyval_product

bae_set_seed(100)

import pandas as pd
import os
import itertools
import pickle as pickle
import numpy as np
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v5
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method

from util.exp_manager import ExperimentManager
import torch

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
# train_bae = True
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

# ==================================
def downsample_data(data_series, n=10):
    temp_df = pd.DataFrame(data_series)
    resampled_data = (
        temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
    )
    return resampled_data


order = 4
resample_factor = 50
trace = x_id_train[1, -1]
decimated_signal = decimate(trace, q=resample_factor, ftype="iir", n=order)
# decimated_signal = decimate(
#     decimate(trace, q=10, ftype="iir", n=order), q=5, n=order, ftype="iir"
# )
pd_agg_signal = downsample_data(trace, n=resample_factor)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(trace)
ax2.plot(decimated_signal)
ax3.plot(pd_agg_signal)
