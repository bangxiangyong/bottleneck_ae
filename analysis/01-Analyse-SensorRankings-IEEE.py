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
from thesis_experiments.util_analyse import (
    get_topk_grid,
    save_pickle_grid,
    rearrange_df,
    replace_df_label_maps,
)
from pprint import pprint

plt.rcParams.update({"font.size": 20})

# Concat all results from HPC slurms
# dataset = "ZEMA"
dataset = "STRATH"
#
# OPTION: Run second half of the script to create grids
sort_sensor_order = True
# save_topk_grid = True
save_topk_grid = False
save_fig = True
# save_fig = False
save_fig_folder = "plots/"
save_topk_pickle_map = {
    # "ZEMA": "../grids/ZEMA-ranking-grid-nolayer-20220414.p",
    # "STRATH": "../grids/STRATH-ranking-grid-nolayer-20220414.p",
    # "ZEMA": "../grids/ZEMA-reboot-rk-grid-20220418.p",
    # "STRATH": "../grids/STRATH-reboot-rk-grid-20220418.p",
    "ZEMA": "../grids/ZEMA-stdmse-rk-grid-20220429.p",
    "STRATH": "../grids/STRATH-stdmse-rk-grid-20220429.p",
}
results_folders = {
    # "ZEMA": "../results/zema-reboot-ssr-20220418/",
    # "STRATH": "../results/strath-reboot-ssr-20220418/",
    # "ZEMA": "../results/zema-stgauss-ssr-20220426/",
    # "STRATH": "../results/strath-stgauss-ssr-20220426/",
    # "ZEMA": "../results/zema-stgauss-temp-ssr-20220426/",
    # "STRATH": "../results/strath-stgauss-temp-ssr-20220426/",
    # "ZEMA": "../results/zema-stdmse-ssr-20220429/",
    # "STRATH": "../results/strath-stdmse-ssr-20220429/",
    "ZEMA": "../results/zema-ssr-20220514/",
    "STRATH": "../results/strath-ssr-20220514/",
}
sensor_metadata_maps = {
    "ZEMA": "../case_study/pickles/zema_hyd_inputs_outputs.p",
    "STRATH": "../case_study/pickles/strath_inputs_outputs.p",
}
sensor_metadata = pickle.load(open(sensor_metadata_maps[dataset], "rb"))[
    "sensor_metadata"
]
results_folder = results_folders[dataset]

all_files = [file for file in os.listdir(results_folder) if "AUROC.csv" in file]
res_df_full = pd.concat([pd.read_csv(results_folder + file) for file in all_files])

# select LL
res_df_full = res_df_full[res_df_full["full_likelihood"] == "mse"]

ranks_args_all = {}  # structure: target_dim -> ae+ll -> sensor_ranks

# SELECT OPTIMISED HYPERPARAMETERS
fixed_params = ["bae_type", "full_likelihood"]
optim_params = ["current_epoch"]
perf_key = "E_AUROC"  # to maximise this

# CREATE A COLUMN BY ADDING TWO COLUMNS
def add_df_cols(df, cols):
    # get model+ll column identifiers
    # add multiple columns of a given df into a single col
    new_df_col = None
    for i, fixed_param in enumerate(cols):
        if i == 0 and new_df_col is None:
            new_df_col = df.loc[:, fixed_param].astype(str).copy()
        else:
            new_df_col += df.loc[:, fixed_param].astype(str).copy()
    return new_df_col


# iterate through target dim
all_targets = res_df_full["target_dim"].unique()
all_max_params = []
for target_dim in all_targets:
    res_df_target_dim = res_df_full[res_df_full["target_dim"] == target_dim].copy()
    all_params = fixed_params + optim_params

    # mean over random seeds, while fixing these params
    res_df_target_dim_groupby = (
        res_df_target_dim.groupby(all_params).mean()[perf_key].reset_index()
    )
    res_df_target_dim_groupby["fixed_params"] = add_df_cols(
        res_df_target_dim_groupby, fixed_params
    )

    # for each combination of fixed params
    for fixed_param in res_df_target_dim_groupby["fixed_params"].unique():
        temp_df = res_df_target_dim_groupby[
            res_df_target_dim_groupby["fixed_params"] == fixed_param
        ]
        argmax_params = temp_df[perf_key].argmax()

        # get the optimised params
        max_params = temp_df.iloc[argmax_params].loc[all_params]
        max_params.loc["target_dim"] = target_dim
        all_max_params.append(max_params)

# result of optimised params to be merged with main raw df (for filtering)
all_max_params_ = pd.concat(all_max_params, axis=1).T  # concat results from the loop
optim_df = res_df_full.merge(all_max_params_).copy()  # filter raw df by max params
optim_df["ae+ll"] = add_df_cols(optim_df, fixed_params)

# ===============CONVERT AUROCS INTO SENSOR RANKINGS=========================
filtered_res_ = optim_df.copy()
for tgt_i, target_dim in enumerate(all_targets):

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
                & (filtered_res_["target_dim"] == target_dim)
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
    sensor_rankings_res["normalised_ranks"] = sensor_rankings_res["ranks"] / np.max(
        sensor_rankings_res["ranks"]
    )

    # ===============PLOT RANKS =======================
    ## Cosmetic modifications for plotting
    sensor_rankings_res_temp_ = sensor_rankings_res.copy()
    sensor_rankings_res_temp_ = rearrange_df(
        sensor_rankings_res_temp_,
        "ae+ll",
        labels=["aebernoulli", "aecbernoulli", "aemse", "aestd-mse", "aestatic-tgauss"],
    )
    ll_map = {
        "aebernoulli": "Bernoulli",
        "aecbernoulli": "C-Bernoulli",
        "aemse": "Gaussian",
        "aestd-mse": "Gaussian (Z-std)",
        "aestatic-tgauss": "Trunc. Gaussian",
    }

    sensor_maps = {
        ss_id: sensor_metadata.iloc[ss_id]["sensor_id"]
        for ss_id in sensor_rankings_res_temp_["ss_id"].unique()
    }

    sensor_rankings_res_temp_ = replace_df_label_maps(
        sensor_rankings_res_temp_, ll_map, col="ae+ll"
    )
    sensor_rankings_res_temp_ = replace_df_label_maps(
        sensor_rankings_res_temp_, sensor_maps, col="ss_id"
    )

    # sort by sensors
    if sort_sensor_order:
        sensor_order = (
            sensor_rankings_res_temp_.groupby("ss_id")["normalised_ranks"]
            .mean()
            .sort_values(ascending=False)
            .index.values
        )
    if dataset == "ZEMA":
        figsize = (9, 2.5)
    if dataset == "STRATH":
        figsize = (9, 2.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bplot = sns.barplot(
        x="ss_id",
        hue="ae+ll",
        y="normalised_ranks",
        data=sensor_rankings_res_temp_,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        palette="tab10",
        order=sensor_order if sort_sensor_order else None,
    )
    # plt.xticks(rotation=25)
    plt.xticks(rotation=25, fontsize="small")
    ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.5)
    if not save_fig:
        ax.set_title("TARGET DIM:" + str(target_dim))
    ax.set_ylabel("Ranks")
    ax.set_xlabel("Sensors")
    ax.set_xlabel("")
    # ax.axhline(y=0.8, alpha=0.35, linestyle="--", color="black")
    ax.set_yticks([0, 0.5, 1.0])
    # legends
    l = ax.legend(fontsize="x-small")
    # l = ax.legend()
    l.set_title("")
    # if tgt_i != (len(all_targets) - 1):
    # if tgt_i != 0:
    l.remove()
    fig.tight_layout(pad=0)
    if save_fig:
        fig.savefig(
            save_fig_folder + dataset + "-SSR-IEEE-" + str(target_dim) + ".png", dpi=600
        )
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
print("DATASET:" + str(dataset))
print("SENSOR RANKS:")
pprint(ranks_args_all)

# ===================HORIZONTAL BAR==============================
filtered_res_ = optim_df.copy()
for tgt_i, target_dim in enumerate(all_targets):

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
                & (filtered_res_["target_dim"] == target_dim)
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
    sensor_rankings_res["normalised_ranks"] = sensor_rankings_res["ranks"] / np.max(
        sensor_rankings_res["ranks"]
    )

    # ===============PLOT RANKS =======================
    ## Cosmetic modifications for plotting
    sensor_rankings_res_temp_ = sensor_rankings_res.copy()
    sensor_rankings_res_temp_ = rearrange_df(
        sensor_rankings_res_temp_,
        "ae+ll",
        labels=["aebernoulli", "aecbernoulli", "aemse", "aestd-mse", "aestatic-tgauss"],
    )
    ll_map = {
        "aebernoulli": "Bernoulli",
        "aecbernoulli": "C-Bernoulli",
        "aemse": "Gaussian",
        "aestd-mse": "Gaussian (Z-std)",
        "aestatic-tgauss": "Trunc. Gaussian",
    }

    sensor_maps = {
        ss_id: sensor_metadata.iloc[ss_id]["sensor_id"]
        for ss_id in sensor_rankings_res_temp_["ss_id"].unique()
    }

    sensor_rankings_res_temp_ = replace_df_label_maps(
        sensor_rankings_res_temp_, ll_map, col="ae+ll"
    )
    sensor_rankings_res_temp_ = replace_df_label_maps(
        sensor_rankings_res_temp_, sensor_maps, col="ss_id"
    )

    # sort by sensors
    if sort_sensor_order:
        sensor_order = (
            sensor_rankings_res_temp_.groupby("ss_id")["normalised_ranks"]
            .mean()
            .sort_values(ascending=False)
            .index.values
        )
    if dataset == "ZEMA":
        figsize = (2.4, 4.5)
    if dataset == "STRATH":
        figsize = (9, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bplot = sns.barplot(
        y="ss_id",
        hue="ae+ll",
        x="normalised_ranks",
        data=sensor_rankings_res_temp_,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        palette="tab10",
        order=sensor_order if sort_sensor_order else None,
    )
    # plt.xticks(rotation=25)
    # plt.xticks(rotation=25, fontsize="small")
    ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.5)
    ax.set_xlabel("Ranks", fontsize="small")
    # ax.set_xlabel("Sensors")
    ax.set_ylabel("")

    zema_title_map = {0: "Cooler", 1: "Valve", 2: "Pump", 3: "Accumulator"}

    ax.set_title(zema_title_map[target_dim], fontsize="small")
    # ax.axhline(y=0.8, alpha=0.35, linestyle="--", color="black")
    ax.set_xticks([0, 0.5, 1.0])
    # legends
    l = ax.legend(fontsize="x-small")
    # l = ax.legend()
    l.set_title("")
    # if tgt_i != (len(all_targets) - 1):
    # if tgt_i != 0:
    l.remove()
    fig.tight_layout(pad=0)
    # fig.subplots_adjust(
    #     top=0.935, bottom=0.149, left=0.269, right=0.92, hspace=0.2, wspace=0.2
    # )
    if save_fig:
        fig.savefig(
            save_fig_folder
            + dataset
            + "-SSR-IEEE-HORIZONTAL"
            + str(target_dim)
            + ".png",
            dpi=600,
        )
