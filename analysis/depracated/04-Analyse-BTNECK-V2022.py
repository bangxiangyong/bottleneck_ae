import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from pprint import pprint

# dataset = "FashionMNIST"
# dataset = "CIFAR"
from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    append_mean_ate_rows,
)

# dataset = "STRATH"
# dataset = "ZEMA"
# dataset = "Images"
dataset = "ODDS"

metric_key_word = "AUROC"
round_deci = 3

## define path to load the csv and mappings related to datasets
paths = {
    "ZEMA": "../results/zema-btneck-20220419",
    "STRATH": "../results/strath-btneck-20220419",
    "ODDS": "../results/odds-btneck-20220422",
    # "Images": "../results/images-btneck-repaired-20220420",
    "Images": "../results/images-btneck-reboot-20220421",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}

tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
path = paths[dataset]

all_pivots = []


## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word)

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

# whether the model is bottlenecked type or not
raw_df["HAS_BTNECK"] = np.select(
    [raw_df["BTNECK_TYPE"] == "A", raw_df["BTNECK_TYPE"] != "A"], ["YES", "NO"]
)

# raw_df = raw_df[raw_df["layer_norm"] == "none"]

##======OPTIMISE PARAMS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "BTNECK_TYPE"],
    optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)

# optim_df = apply_optim_df(
#     raw_df,
#     fixed_params=["bae_type", "BTNECK"],
#     optim_params=["layer_norm"],
#     perf_key="E_AUROC",
#     target_dim_col=tasks_col_map[dataset],
# )

# optim_df = apply_optim_df(
#     raw_df,
#     fixed_params=["bae_type", "BTNECK"],
#     # optim_params=["current_epoch", "latent_factor", "skip", "layer_norm", "bae_type"],
#     optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
#     perf_key="E_AUROC",
#     target_dim_col=tasks_col_map[dataset],
# )

##======PLOT BOXPLOT===========
showfliers = False
for target_dim in optim_df[tasks_col_name].unique():
    temp_df = optim_df[optim_df[tasks_col_name] == target_dim]
    bae_type_order = ["ae", "vae", "mcd", "vi", "ens"]
    x = "bae_type"
    hue = "BTNECK_TYPE"
    x_order = bae_type_order
    hue_order = ["A", "B", "C", "D"]
    y = "E_" + metric_key_word

    # BOX PLOT
    # fig, ax = plt.subplots(1, 1)
    # sns.boxplot(
    #     x=x,
    #     y=y,
    #     hue=hue,
    #     ax=ax,
    #     data=temp_df,
    #     order=x_order,
    #     hue_order=hue_order,
    #     showfliers=showfliers,
    # )
    # fig.canvas.manager.set_window_title(str(target_dim))

    #
    fig, ax = plt.subplots(1, 1)
    bplot = sns.barplot(
        x="bae_type",
        hue="BTNECK_TYPE",
        y="E_" + metric_key_word,
        data=optim_df[optim_df[tasks_col_name] == target_dim],
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        hatch="///",
        hue_order=["A", "B", "C", "D"],
    )
    fig.canvas.manager.set_window_title(str(target_dim))

# =========TABLE OF RESULTS PER TARGET DIM==========
groupby_cols = ["bae_type", "BTNECK_TYPE"]
collect_pivot_abcd_dfs = []
for target_dim in optim_df[tasks_col_name].unique():
    res_groupby = (
        optim_df[optim_df[tasks_col_name] == target_dim]
        .groupby(groupby_cols)
        .mean()["E_" + metric_key_word]
        .reset_index()
    )

    # convert to pivot table per task
    pivot_df = res_groupby.pivot(
        index="bae_type", columns="BTNECK_TYPE", values="E_" + metric_key_word
    ).reset_index()

    # SUMMARY ROW: Mean and ATE of tasks
    pivot_df_mean_ate = append_mean_ate_rows(
        pivot_df, label_col="bae_type", baseline_col="A"
    )

    # Collect to make final table
    collect_pivot_abcd_dfs.append(pivot_df)

    # print results
    print("RESULT PER TASK:" + str(target_dim))
    pprint(pivot_df_mean_ate.round(round_deci))

# Aggregated tables for all tasks
collect_pivot_abcd_dfs = pd.concat(collect_pivot_abcd_dfs)
collect_pivot_abcd_dfs = (
    collect_pivot_abcd_dfs.groupby(["bae_type"]).mean().reset_index()
)
collect_pivot_abcd_dfs = append_mean_ate_rows(
    collect_pivot_abcd_dfs, label_col="bae_type", baseline_col="A"
)

# print results
print("AGGREGATED RESULTS:")
pprint(collect_pivot_abcd_dfs.round(round_deci))

print("===========================================")
print("==========BEST BTNECK VS NON-BTNECK=========")

# ==========BEST BTNECK VS NONBTNECK==========
groupby_cols = ["bae_type", "HAS_BTNECK"]
optim_df_hasbtneck = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "HAS_BTNECK"],
    optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)
collect_pivot_dfs_btneck = []
for target_dim in optim_df_hasbtneck[tasks_col_name].unique():
    res_groupby = (
        optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim]
        .groupby(groupby_cols)
        .mean()["E_" + metric_key_word]
        .reset_index()
    )

    # convert to pivot table
    pivot_df = res_groupby.pivot(
        index="bae_type", columns="HAS_BTNECK", values="E_" + metric_key_word
    ).reset_index()

    # SUMMARY ROW: Mean and ATE of tasks
    pivot_df_mean_ate = append_mean_ate_rows(
        pivot_df, label_col="bae_type", baseline_col="YES"
    )

    # Collect to make final table
    collect_pivot_dfs_btneck.append(pivot_df)

    # print results
    print("RESULT PER TASK:" + str(target_dim))
    pprint(pivot_df_mean_ate.round(round_deci))

    # ========PLOT BEST BTNECK========
    fig, ax = plt.subplots(1, 1)
    bplot = sns.barplot(
        x="bae_type",
        hue="HAS_BTNECK",
        y="E_" + metric_key_word,
        data=optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim],
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        hatch="///",
        hue_order=["YES", "NO"],
    )

    # bplot = sns.boxplot(
    #     x="bae_type",
    #     hue="HAS_BTNECK",
    #     y="E_" + metric_key_word,
    #     data=optim_df_all[optim_df_all[tasks_col_name] == target_dim],
    #     ax=ax,
    #     hue_order=[True, False],
    # )
    fig.canvas.manager.set_window_title(str(target_dim))

# Aggregated tables for all tasks
collect_pivot_dfs_btneck = pd.concat(collect_pivot_dfs_btneck)
collect_pivot_dfs_btneck = (
    collect_pivot_dfs_btneck.groupby(["bae_type"]).mean().reset_index()
)
collect_pivot_dfs_btneck = append_mean_ate_rows(
    collect_pivot_dfs_btneck, label_col="bae_type", baseline_col="YES"
)

# print results
print("AGGREGATED RESULTS:")
pprint(collect_pivot_dfs_btneck.round(round_deci))
