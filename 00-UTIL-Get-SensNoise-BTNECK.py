# Finds the best btnecked and non-btnecked architecture for each dataset
# And create grids that inject noise into them to be evaluated
import itertools
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    append_mean_ate_rows,
    rearrange_df,
    replace_df_label_maps,
    save_pickle_grid,
)

# from case_study import Params_ZEMA, Params_STRATH
# from thesis_experiments.benchmarks import Params_ODDS, Params_Images

grid_ODDS = {
    "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
    "dataset": [
        "cardio",
        "lympho",
        "optdigits",
        "pendigits",
        "thyroid",
        "ionosphere",
        "pima",
        "vowels",
    ],
    # ========to be optimised========
    "skip": [False],
    "latent_factor": [0.1],
    "layer_norm": ["none"],
    "n_dense_layers": [2],
    "bae_type": ["ae"],
    "num_epochs": [300],
    # ========to be optimised========
    "n_enc_capacity": [4],
    "full_likelihood": ["mse"],
    "n_bae_samples": [-1],  # default
    "activation": ["leakyrelu"],
    "weight_decay": [1e-10],
}

grid_STRATH = {
    "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
    "apply_fft": [False],
    # fmt: off
    "ss_id": [[9]],
    # to be replaced via Best TOP-K Script
    # fmt: on
    "target_dim": [2],
    "mode": ["forging"],
    "resample_factor": [50],
    # ========to be optimised========
    "bae_type": ["ae"],
    "latent_factor": [5],
    "skip": [False],
    "layer_norm": ["none"],
    "n_conv_layers": [1],
    "num_epochs": [200],
    # ========to be optimised========
    "n_dense_layers": [1],
    "n_enc_capacity": [20],
    "full_likelihood": ["mse"],
    "n_bae_samples": [-1],  # default
    "activation": ["leakyrelu"],
    # "weight_decay": [1e-10],
    "weight_decay": [0, 1e-10, 1e-6, 1e-4, 1e-2],
}

grid_ZEMA = {
    "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
    "apply_fft": [False],
    # fmt: off
    "ss_id": [[-1]],
    # to be replaced via Best TOP-K Script
    # fmt: on
    "target_dim": [0, 1, 2, 3],
    "resample_factor": ["Hz_1"],
    # ========to be optimised========
    "skip": [False],
    "layer_norm": ["none"],
    "latent_factor": [0.1],
    "n_conv_layers": [1],
    "bae_type": ["ae"],
    "num_epochs": [200],
    # ========to be optimised========
    "n_dense_layers": [1],
    "n_enc_capacity": [20],
    "full_likelihood": ["mse"],
    "n_bae_samples": [-1],  # default
    "activation": ["leakyrelu"],
    # "weight_decay": [1e-10],
    "weight_decay": [0, 1e-10, 1e-6, 1e-4, 1e-2],
}

grid_Images = {
    "random_seed": [930, 717, 10, 5477, 510],
    "id_dataset": ["FashionMNIST", "CIFAR"],
    "full_likelihood": ["mse"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "activation": ["leakyrelu"],
    "n_dense_layers": [1],
    "n_enc_capacity": [30],
    # ========to be optimised========
    "bae_type": ["ae"],
    "n_conv_layers": [1],
    "latent_factor": [0.1],
    "skip": [False],
    "layer_norm": ["none"],
    "num_epochs": [20],
    # ========to be optimised========
}


# dataset = "STRATH"
dataset = "ZEMA"
# dataset = "Images"
# dataset = "ODDS"

grids = {
    "Images": grid_Images,
    "ZEMA": grid_ZEMA,
    "STRATH": grid_STRATH,
    "ODDS": grid_ODDS,
}
full_grid = grids[dataset]

# add noise grid
noise_grid = {
    "noise_scale": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "noise_type": ["normal", "uniform"],
}

## ====Options: Add noise or not to grid=============
## TRUE: ADD noise grid to the final best BTNECK grid (TRAIN+TEST)
## FALSE: Or select only the best BTNECK grid and add noise later on (TEST).
add_noise_grid = False
# add_noise_grid = True

## ====Options: Select best AE or BAE=========
# select_bae_type = ["ae_"]
# select_bae_type = ["bae_"]
# select_bae_type = ["sae_"]
select_bae_type = ["ae_", "sae_"]

## ====Options: Select btneck or nonbtneck====
# select_btneck_type = ["hasbtneck", "nobtneck"]
select_btneck_type = ["nobtneck"]

##=====Option: Save to pickle grid============
save_grid = True
# save_grid = False

# noisy_grid_filename = "grids/" + dataset + "-246-nonoise.p"  # save grid file
# noisy_grid_filename = (
#     "grids/" + dataset + "-noise+" + select_bae_type[0] + "-zero.p"
# )  # save grid file
# noisy_grid_filename = "grids/" + dataset + "-ae-noise-test.p"  # save grid file
noisy_grid_filename = "grids/" + dataset + "-sae-wdecay-test.p"  # save grid file

metric_key_word = "AUROC"

## define path to load the csv and mappings related to datasets
paths = {
    # "ZEMA": "results/zema-btneck-overhaul-repair-v2-20220517",
    # # "STRATH": "../results/STRATH-BOTTLENECKV3",
    # # "ODDS": "../results/odds-btneck-20220422",
    # # "Images": "../results/images-btneck-reboot-20220421",
    # # "ZEMA": "../results/zema-btneck-20220515",
    # # "STRATH": "../results/strath-btneck-20220515",
    # # "STRATH": "../results/strath-btneck-bxy20-20220519",
    # # "STRATH": "../results/strath-btneck-overhaul-20220516",
    # "STRATH": "results/strath-btneck-incomp-v4",
    # # "ODDS": "results/odds-btneck-ovhaul-20220525",
    # "ODDS": "results/odds-btneck-246-20220527",
    # # "Images": "../results/images-btneck-20220515",
    # # "Images": "results/cifar-btneck-overhaul-full-20220522",
    # # "Images": "../results/fmnist-btneck-overhaul-v2-20220524",
    # "Images": "results/images-btneck-overhaul-20220525",
    "ZEMA": "results/zema-btneck-overhaul-repair-v2-20220517",
    "STRATH": "results/strath-btneck-incomp-v4",
    "ODDS": "results/odds-btneck-246-20220527",
    "Images": "results/images-btneck-overhaul-20220525",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
path = paths[dataset]

## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word)
if dataset == "STRATH":
    raw_df = raw_df[raw_df["resample_factor"] == 50]

##======CREATE LABELS OF BTNECK TYPE=====
# list of conditions
btneck_A = (raw_df["latent_factor"] < 1.0) & (raw_df["skip"] == False)
btneck_B = (raw_df["latent_factor"] >= 1.0) & (raw_df["skip"] == False)
btneck_C = (raw_df["latent_factor"] < 1.0) & (raw_df["skip"] == True)
btneck_D = (raw_df["latent_factor"] >= 1.0) & (raw_df["skip"] == True)
conditions = [
    btneck_A,
    btneck_B,
    btneck_C,
    btneck_D,
]
btneck_labels = ["A", "B", "C", "D"]  # label for each condition
raw_df["BTNECK_TYPE"] = np.select(
    conditions, btneck_labels
)  # apply condition and label
raw_df["BTNECK_TYPE+INF"] = np.select(
    conditions, btneck_labels
)  # apply condition and label

# whether the model is bottlenecked type or not
raw_df["HAS_BTNECK"] = np.select(
    [raw_df["BTNECK_TYPE"] == "A", raw_df["BTNECK_TYPE"] != "A"], ["YES", "NO"]
)

# ================================================================
print("===========================================")
print("==========BEST BTNECK VS NON-BTNECK=========")

groupby_cols = ["bae_type", "HAS_BTNECK"]

collect_gridlist = []

# for bae_type_ in ["ae_", "bae_"]:
for bae_type_ in select_bae_type:
    if bae_type_ == "ae_":
        optim_df_hasbtneck = apply_optim_df(
            raw_df[raw_df["bae_type"] == "ae"],
            fixed_params=["bae_type", "HAS_BTNECK"],
            optim_params=[
                "current_epoch",
                "latent_factor",
                "skip",
                "layer_norm",
                "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
                "resample_factor",
            ],
            perf_key="E_AUROC",
            target_dim_col=tasks_col_map[dataset],
        )
    elif bae_type_ == "sae_":  # override AE with SAE
        optim_df_hasbtneck = apply_optim_df(
            raw_df[raw_df["bae_type"] == "ae"],
            fixed_params=["bae_type", "HAS_BTNECK"],
            optim_params=[
                "current_epoch",
                "latent_factor",
                "skip",
                "layer_norm",
                "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
                "resample_factor",
            ],
            perf_key="E_AUROC",
            target_dim_col=tasks_col_map[dataset],
        )
        optim_df_hasbtneck["bae_type"] = "sae"
    elif bae_type_ == "bae_":
        # select the best BAE
        optim_df_hasbtneck = apply_optim_df(
            raw_df[(raw_df["bae_type"] != "ae") & (raw_df["bae_type"] != "vae")],
            fixed_params=["HAS_BTNECK"],
            optim_params=[
                "bae_type",
                "current_epoch",
                "latent_factor",
                "skip",
                "layer_norm",
                "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
                "resample_factor",
            ],
            perf_key="E_AUROC",
            target_dim_col=tasks_col_map[dataset],
        )

    for target_dim in optim_df_hasbtneck[tasks_col_name].unique():
        # for target_dim in [0]:
        res_groupby = (
            optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim]
            .groupby(groupby_cols)
            .mean()["E_" + metric_key_word]
            .reset_index()
        )

        res_groupby = (
            optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim]
            # .groupby(["latent_factor", "skip", "layer_norm", "n_conv_layers", "resample_factor"])
            # .mean()["E_" + metric_key_word]
            # .reset_index()
        )
        # bae_type = "ae"
        col_params = [
            "bae_type",
            "latent_factor",
            "skip",
            "layer_norm",
            "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
            "current_epoch",
            tasks_col_name,
        ]
        params_hasbtneck = (
            optim_df_hasbtneck[
                (optim_df_hasbtneck[tasks_col_name] == target_dim)
                # & (optim_df_hasbtneck["bae_type"] == bae_type)
                & (optim_df_hasbtneck["HAS_BTNECK"] == "YES")
            ][col_params]
            .reset_index(drop=True)
            .drop_duplicates()
        )

        params_nobtneck = (
            optim_df_hasbtneck[
                (optim_df_hasbtneck[tasks_col_name] == target_dim)
                # & (optim_df_hasbtneck["bae_type"] == bae_type)
                & (optim_df_hasbtneck["HAS_BTNECK"] == "NO")
            ][col_params]
            .reset_index(drop=True)
            .drop_duplicates()
        )

        # convert into grid
        hasbtneck_grid = full_grid.copy()
        for col in params_hasbtneck.columns:
            hasbtneck_grid.update({col: [params_hasbtneck[col].item()]})

        nobtneck_grid = full_grid.copy()
        for col in params_nobtneck.columns:
            nobtneck_grid.update({col: [params_nobtneck[col].item()]})

        # get best num epoch
        hasbtneck_grid.update(
            {"num_epochs": [params_hasbtneck["current_epoch"].item()]}
        )
        nobtneck_grid.update({"num_epochs": [params_nobtneck["current_epoch"].item()]})

        # add noise grids
        if add_noise_grid:
            for key in noise_grid.keys():
                hasbtneck_grid.update({key: noise_grid[key]})
                nobtneck_grid.update({key: noise_grid[key]})

        # unpack noise grid list
        hasbtneck_grid_keys = hasbtneck_grid.keys()
        hasbtneck_grid_list = list(itertools.product(*hasbtneck_grid.values()))
        nobtneck_grid_keys = nobtneck_grid.keys()
        nobtneck_grid_list = list(itertools.product(*nobtneck_grid.values()))

        # collect grid list
        if "hasbtneck" in select_btneck_type:
            collect_gridlist += hasbtneck_grid_list
        if "nobtneck" in select_btneck_type:
            collect_gridlist += nobtneck_grid_list

# select_btneck_type = ["hasbtneck", "nobtneck"]
select_btneck_type = ["nobtneck"]
if "hasbtneck" in select_btneck_type:
    print(hasbtneck_grid_list[-5:])
if "nobtneck" in select_btneck_type:
    print(nobtneck_grid_list[-5:])
print("GRID LENGTH:" + str(len(collect_gridlist)))
if save_grid:
    grid_keys = list(hasbtneck_grid_keys)
    save_pickle_grid(grid_keys, collect_gridlist, noisy_grid_filename)
    print("SAVED GRID: " + noisy_grid_filename)
