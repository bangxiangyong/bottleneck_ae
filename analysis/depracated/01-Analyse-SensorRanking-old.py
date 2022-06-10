# This script converts the individually trained BAE experiments into sensor rankings importance
# OUTPUT 1: Bar plots of sensor importance
# OUTPUT 2: TOP-K sensor grids for each likelihood choice (to be executed on main scripts)

import itertools
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import scipy.stats
from thesis_experiments.util_analyse import get_topk_grid, save_pickle_grid
from pprint import pprint

# Concat all results from HPC slurms
# dataset = "ZEMA"
dataset = "STRATH"

# OPTION: Run second half of the script to create grids
# save_topk_grid = True
save_topk_grid = False

save_topk_pickle_map = {
    # "ZEMA": "../grids/ZEMA-ranking-grid-nolayer-20220414.p",
    # "STRATH": "../grids/STRATH-ranking-grid-nolayer-20220414.p",
    "ZEMA": "../grids/ZEMA-ranking-grid-ex-20220416.p",
    "STRATH": "../grids/STRATH-ranking-grid-ex-20220416.p",
}
results_folders = {
    # "ZEMA": "../results/zema-sensors-rank-20220414/",
    # "STRATH": "../results/strath-sensors-rank-20220414/",
    "ZEMA": "../results/zema-ex-sensors-rank-20220416/",
    "STRATH": "../results/strath-ex-sensors-rank-20220416/",
}
results_folder = results_folders[dataset]


all_files = [file for file in os.listdir(results_folder) if "AUROC.csv" in file]
res_df_full = pd.concat([pd.read_csv(results_folder + file) for file in all_files])
ranks_args_all = {}  # structure: target_dim -> ae+ll -> sensor_ranks

# turn off layer norm
res_df_full = res_df_full[res_df_full["layer_norm"] == "none"]

all_targets = res_df_full["target_dim"].unique()
for target_dim in all_targets:
    res_df_target_dim = res_df_full[res_df_full["target_dim"] == target_dim]
    res_df_target_dim["ae+ll"] = (
        res_df_target_dim["bae_type"] + res_df_target_dim["full_likelihood"]
    )
    mode = "ss"
    figsize = (6, 3)
    mean_auroc = (
        res_df_target_dim.groupby(
            ["bae_type", mode + "_id", "full_likelihood", "layer_norm"]
        )
        .mean()["E_AUROC"]
        .reset_index()
    )

    # =====================================================
    # ====find layer_norm params that maximises E_AUROC====
    # =====================================================
    # params are fixed at this grid
    # we find the rows that maximises AUROC with values fixed to this grid
    grid = {
        "bae_type": mean_auroc["bae_type"].unique(),
        "ss_id": mean_auroc["ss_id"].unique(),
        "full_likelihood": mean_auroc["full_likelihood"].unique(),
    }
    filtered_res = []
    for values in itertools.product(*grid.values()):
        # setup the grid
        exp_params = dict(zip(grid.keys(), values))
        print(exp_params)
        new_row = pd.DataFrame([exp_params])
        merged_rows = new_row.merge(
            mean_auroc
        )  # intersection of fixed params with the final list
        argmax_ = pd.DataFrame(
            merged_rows.iloc[merged_rows["E_AUROC"].argmax()]
        ).T  # get rows in the intersected that maximises AUROC
        argmax_ = argmax_.drop(["E_AUROC"], axis=1)
        filtered_res.append(res_df_target_dim.merge(argmax_))  # add into final df
    filtered_res_ = pd.concat(filtered_res)
    # ===============PLOT BAR AUROCS=========================

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bplot = sns.barplot(
        x="ss_id",
        hue="ae+ll",
        y="E_AUROC",
        data=filtered_res_,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        hatch="///",
    )
    ax.axhline(y=0.7, color="black", linestyle="--")
    ax.set_title(str(target_dim))

    # ==============CONVERT AUROC TO RANKINGS================
    # group by random seed and model+ll,
    # then compute rank based on the sensor
    # setup the grid combination
    random_seeds = filtered_res_["random_seed"].unique()
    ae_ll = filtered_res_["ae+ll"].unique()
    rank_grid = {"random_seed": random_seeds, "ae+ll": ae_ll}
    sensor_rankings_res = []  # collect results
    for values in itertools.product(*rank_grid.values()):
        # get grid param
        exp_params = dict(zip(rank_grid.keys(), values))

        # groupby res
        groupby_res = (
            filtered_res_[
                (filtered_res_["random_seed"] == exp_params["random_seed"])
                & (filtered_res_["ae+ll"] == exp_params["ae+ll"])
            ]
            .copy()
            .sort_values("ss_id")
        )

        # convert the aurocs into rankings
        aurocs = groupby_res["E_AUROC"]  # get the aurocs
        sensor_aurocs_rankings = scipy.stats.rankdata(aurocs)  # do ranking
        groupby_res["ranks"] = sensor_aurocs_rankings  # add new column
        sensor_rankings_res.append(groupby_res)  # add new res into final list
    sensor_rankings_res = pd.concat(sensor_rankings_res).reset_index(drop=True)

    # normalise ranks
    # normalised_ss_ranks = []
    # for values in itertools.product(*rank_grid.values()):
    sensor_rankings_res["normalised_ranks"] = sensor_rankings_res["ranks"] / np.max(
        sensor_rankings_res["ranks"]
    )
    # ===============PLOT RANKS =======================

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bplot = sns.barplot(
        x="ss_id",
        hue="ae+ll",
        y="normalised_ranks",
        data=sensor_rankings_res,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        hatch="///",
    )
    ax.axhline(y=0.7, color="black", linestyle="--")
    ax.axhline(y=1.0, color="black", linestyle="--")
    ax.set_title("TARGET DIM:" + str(target_dim))
    ax.set_ylabel("SENSOR IMPORTANCE RANKS (NORMALISED)")

    # ===============EXTRACT SENSOR RANKINGS=====================
    mean_ranks = (
        sensor_rankings_res.groupby(["ss_id", "ae+ll"]).mean()["ranks"].reset_index()
    )

    # need to collect results for each target_dim
    ranks_args_target = {}
    for ae_ll_i in ae_ll:
        ranks_df = mean_ranks[mean_ranks["ae+ll"] == ae_ll_i].sort_values("ss_id")
        ss_ids = ranks_df["ss_id"].values
        ranks_args = list(np.argsort(ranks_df["ranks"]))[::-1]
        ranks_args_target.update({ae_ll_i: [ss_ids[arg] for arg in ranks_args]})
        ranks_args_target.update({ae_ll_i: [ss_ids[arg] for arg in ranks_args]})
    ranks_args_all.update({int(target_dim): ranks_args_target.copy()})
print("SENSOR RANKS:")
pprint(ranks_args_all)
# ===============MAKE CUSTOM GRID FOR TOPK SENSOR RANK=======================
if save_topk_grid:
    conditional_grid = {
        "target_dim": list(all_targets),
        "full_likelihood": list(res_df_full["full_likelihood"].unique()),
    }
    conditional_grid_keys = conditional_grid.keys()
    conditional_grid_params = list(itertools.product(*conditional_grid.values()))
    bae_type = "ae"

    # define a base grid
    grid_base_ZEMA = {
        "random_seed": list(res_df_full["random_seed"].unique()),
        "apply_fft": [False],
        "resample_factor": ["Hz_1"],
        "skip": [False],
        "layer_norm": ["none"],
        "latent_factor": [0.1],
        "bae_type": ["ae"],
        "weight_decay": [1e-10],
        "n_bae_samples": [-1],  # default
        "num_epochs": [100],
    }

    grid_base_STRATH = {
        "random_seed": list(res_df_full["random_seed"].unique()),
        "apply_fft": [False],
        "resample_factor": [50],
        "skip": [False],
        "layer_norm": ["none"],
        "mode": ["forging"],
        "latent_factor": [0.1],
        "bae_type": ["ae"],
        "weight_decay": [1e-10],
        "n_bae_samples": [-1],  # default
        "num_epochs": [50],
    }

    # =====SELECT HERE======
    # grid_base = grid_base_ZEMA  # change here to STRATH/ZEMA
    grid_base_map = {"ZEMA": grid_base_ZEMA, "STRATH": grid_base_STRATH}
    grid_base = grid_base_map[dataset]  # change here to STRATH/ZEMA
    save_topk_pickle_file = save_topk_pickle_map[dataset]

    # ======================

    topk_grid_list = []
    for param_i, values in enumerate(conditional_grid_params):
        # create conditional params
        params = dict(zip(conditional_grid_keys, values))

        # unpack values
        target_dim = params["target_dim"]
        likelihood_i = params["full_likelihood"]
        ae_ll = bae_type + params["full_likelihood"]

        # get sensor ranks per ae+ll and target_dim from the main ranks list
        sensor_ranks = ranks_args_all[target_dim][ae_ll]
        topk_sensors = get_topk_grid(sensor_ranks)

        # create custom new grid with the topk sensors and likelihood
        topk_grid = {
            "ss_id": topk_sensors,
            "full_likelihood": [likelihood_i],
            "target_dim": [target_dim],
        }
        topk_grid.update(grid_base)

        # add the combination of grids to the final grid
        topk_grid_list += list(itertools.product(*topk_grid.values()))

    final_grid_keys = list(topk_grid.keys())

    # VALIDATION CHECK
    same_columns_grid = len(final_grid_keys) == len(
        [col for col in res_df_full.columns[1:-4] if col != "k_sens"]
    )
    print("VALIDATION CHECK FOR COLUMNS OF TOPK GRIDS:" + str(same_columns_grid))
    same_length_grid = len(topk_grid_list) == len(res_df_full)
    print("VALIDATION CHECK FOR ROWS OF TOPK GRIDS:" + str(same_length_grid))
    print("GRID LENGTH:" + str(len(topk_grid_list)))
    if not same_length_grid:
        raise Exception("Created grids not the same number of rows as expected.")
    if not same_columns_grid:
        raise Exception("Created grids not the same number of columns as expected.")

    # save into pickle
    save_pickle_grid(final_grid_keys, topk_grid_list, save_topk_pickle_file)
