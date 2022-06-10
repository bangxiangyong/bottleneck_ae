# =========HPC CODES===============
import os

from util.add_noise import add_noise

os.chdir("/home/bxy20/rds/hpc-work/understanding-bae")
try:
    total_slurms = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    slurm_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1  # SLURM
    USING_SLURM = True
except:
    slurm_id = 0
    total_slurms = 1
    USING_SLURM = False
import torch

use_cuda = torch.cuda.is_available()
if use_cuda:
    print(torch.cuda.get_device_name(0))
import sys

sys.path.append("/home/bxy20/rds/hpc-work/understanding-bae")
from operator import itemgetter

# ==================================


import pickle
import os
import itertools
import InfiniteBAE
import pandas as pd
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.evaluation import (
    calc_auroc,
    calc_avgprc,
)
from util.exp_manager import ExperimentManager
import time
import numpy as np
import torch
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# ======SPECIFY DATASET=======
# dataset = "ZEMA"
# dataset = "STRATH"
dataset = "ODDS"
# dataset = "Images"


check_row_exists = True
# check_row_exists = False

exp_names = {
    "ZEMA": "ZEMA_INF_FULL_",
    "STRATH": "STRATH_INF_FULL_",
    "ODDS": "ODDS_INF_FULL_",
    "Images": "IMAGES_INF_FULL_",
}
exp_name_prefix = exp_names[dataset]

# ========LOAD DATASET===========

filenames = {
    "ZEMA": "ZEMA_np_data.p",
    "STRATH": "STRATH_np_data.p",
    "Images": "Images_np_data.p",
    "ODDS": "ODDS_np_data.p",
}
dataset_folder = "thesis_experiments/np_datasets"
pickled_dataset = pickle.load(
    open(os.path.join(dataset_folder, filenames[dataset]), "rb")
)
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]
# ================================


# =========SPECIFY GRID===========

## FULL GRID
grid = {
    "W_std": [1.4, 1.2, 1.0, 0.8],
    "diag_reg": [1e-5, 1e-4, 1e-3],
    "norm": ["layer", "none"],
    "skip": [False],
    "num_layers": [2, 3, 4, 5],
    "activation": ["leakyrelu", "gelu", "erf"],
}

## STANDARD SINGLE TRY
# grid = {
#   "W_std":[1.2],
#   "diag_reg": [1e-5],
#   "norm":["layer"],
#   "skip":[False],
#   "num_layers":[4],
#   "activation": ["leakyrelu"]
# }

grid_keys = grid.keys()
grid_list = list(itertools.product(*grid.values()))
# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))
# ==========DIVIDE TASKS FOR SLURMS ====================
# NOTE: NUMPY CONVERTS INTO STRING FOR SOME REASON
# NOTE: USE SS_ID: [[8],[9]] IN GRID INSTEAD OF [8,9], OTHERWISE NUMPY CONVERTS INTO STRING?
split_list = np.array_split(np.arange(len(grid_list)), total_slurms)
EXP_CUDA = "GPU_" if use_cuda else "CPU_"
exp_name = exp_name_prefix + EXP_CUDA + str(slurm_id)
this_slurm_tasks = itemgetter(*split_list[slurm_id])(
    grid_list
)  # avoid applying np.array here which converts everything into string
this_slurm_tasks_shape = np.array(this_slurm_tasks).shape
print("SHAPE SLURM LIST:")
print(this_slurm_tasks_shape)
if len(this_slurm_tasks_shape) == 1:
    this_slurm_tasks = [this_slurm_tasks]
print("TOTAL TASKS:" + str(len(grid_list)))
print("TOTAL SLURMS:" + str(total_slurms))
total_slurm_tasks = 0
for slurm_i in range(total_slurms):
    total_slurm_tasks += len(split_list[slurm_i])
print("TOTAL SLURM TASKS:" + str(total_slurm_tasks))
for slurm_i in range(total_slurms):
    print("SLURM# " + str(slurm_i) + ": " + str(len(split_list[slurm_i])) + " TASKS")
if total_slurm_tasks != len(grid_list):
    raise Exception("ERROR: TOTAL SLURM TASKS NOT EQUAL TO TOTAL AVAILABLE TASKS!")
# ======================================================

exp_man = ExperimentManager(folder_name="thesis_experiments/inf_experiments")
start_exp_time = time.time()

final_res = []
for rep, values in enumerate(this_slurm_tasks):

    # setup the grid
    exp_params = dict(zip(grid.keys(), values))
    W_std = exp_params["W_std"]
    diag_reg = exp_params["diag_reg"]
    norm = exp_params["norm"]
    skip = exp_params["skip"]
    num_layers = exp_params["num_layers"]
    activation = exp_params["activation"]

    for target_key, data_list in pickled_dataset.items():
        ## handle images nested structure differently
        if dataset == "Images":
            iterate_list = data_list["train"]
        else:
            iterate_list = data_list

        # N data lists , 1 for each random seed split
        for data_dict in iterate_list:
            random_seed = data_dict["random_seed"]
            x_id_train = data_dict["x_id_train"]

            # unpack for images differently
            if dataset == "Images":
                x_id_train = x_id_train[: len(x_id_train) // 3]
                x_id_test = data_list["x_id_test"]
                x_ood_test = data_list["x_ood_test"]
            else:
                x_id_test = data_dict["x_id_test"]
                x_ood_test = data_dict["x_ood_test"]

            exp_params.update({tasks_col_name: target_key, "random_seed": random_seed})

            # check for continuity
            # a way to continue progress from before
            # if anything happened and interrupted the flow
            if check_row_exists:
                new_row = pd.DataFrame([exp_params])
                csv_path = os.path.join(exp_man.folder_name, exp_name + "_AUROC.csv")
                if os.path.exists(csv_path):
                    read_exp_csv = pd.read_csv(csv_path)
                    num_columns = len(new_row.columns)
                    read_exp_csv_ = read_exp_csv.iloc[:, 1 : num_columns + 1]
                    common_row = read_exp_csv_.astype(str).merge(
                        new_row.astype(str), "inner"
                    )
                    if len(common_row) > 0:  # row already exist
                        print("Row exists, skipping to next iteration...")
                        continue

            # instantiate BAE
            ae_layers = InfiniteBAE.get_AE_layers(
                enc_params=["linear"] * num_layers,
                activation=activation,
                norm=norm,
                last_activation="sigmoid",
                W_std=W_std,
                b_std=None,
                parameterization="standard",
            )

            if skip:
                BAE = InfiniteBAE.InfiniteBAE(
                    diag_reg=diag_reg,
                    layers=InfiniteBAE.add_unet_skip(
                        ae_layers, last_activation="sigmoid"
                    ),
                    inf_type="nngp",
                )
            else:
                BAE = InfiniteBAE.InfiniteBAE(
                    diag_reg=diag_reg, layers=ae_layers, inf_type="nngp"
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

            try:
                # fit-predict
                BAE.fit(flatten_np(x_id_train))
                y_mu_id_test, nll_grid_id_test = BAE.predict(flatten_np(x_id_test))
                y_mu_ood_test, nll_grid_ood_test = BAE.predict(flatten_np(x_ood_test))

                # evaluate
                auroc_score = calc_auroc(
                    flatten_np(nll_grid_id_test).mean(-1),
                    flatten_np(nll_grid_ood_test).mean(-1),
                )
                avgprc_score = calc_avgprc(
                    flatten_np(nll_grid_id_test).mean(-1),
                    flatten_np(nll_grid_ood_test).mean(-1),
                )

                # new results
                new_res_auroc = {"E_AUROC": auroc_score}
                new_res_avgprc = {"E_AVGPRC": avgprc_score}

                # Exp params
                print(exp_params)
                print(new_res_auroc)

                exp_man.update_csv(
                    exp_params=exp_man.concat_params_res(exp_params, new_res_auroc),
                    csv_name=exp_name + "_AUROC.csv",
                )
                exp_man.update_csv(
                    exp_params=exp_man.concat_params_res(exp_params, new_res_avgprc),
                    csv_name=exp_name + "_AVGPRC.csv",
                )

            except Exception as e:
                err_msg = type(e).__name__ + ":" + str(e)
                print(err_msg)
                exp_man.update_csv(
                    exp_params=exp_man.concat_params_res(
                        exp_params, {"ERROR MSG": err_msg}
                    ),
                    csv_name=exp_name + "_ERROR.csv",
                )
end_exp_time = time.time()
taken_exp_time = end_exp_time - start_exp_time
print(
    "EXP TIME TAKEN: {:.5f} for ".format(taken_exp_time)
    + str(len(split_list[slurm_id]))
    + " tasks."
)
