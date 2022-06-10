# Basically to illustrate the difference from baseline deterministic AE
# In a more obvious way!

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
)

# dataset = "STRATH"
# dataset = "ZEMA"
# dataset = "Images"
dataset = "ODDS"

metric_key_word = "AUROC"
round_deci = 3
save_fig = False
save_csv = False
# save_csv = True

## define path to load the csv and mappings related to datasets
# paths = {
#     # "ZEMA": "../results/zema-btneck-20220419",
#     # "STRATH": "../results/STRATH-BOTTLENECKV3",
#     # "ODDS": "../results/odds-btneck-20220422",
#     # "Images": "../results/images-btneck-reboot-20220421",
#     "ZEMA": "../results/zema-btneck-20220515",
#     "STRATH": "../results/strath-btneck-20220515",
#     "ODDS": "../results/odds-btneck-20220515",
#     "Images": "../results/images-btneck-20220515",
# }

paths = {
    "ZEMA": "../results/zema-btneck-overhaul-repair-v2-20220517",
    "STRATH": "../results/strath-btneck-incomp-v4",
    "ODDS": "../results/odds-btneck-246-20220527",
    "Images": "../results/images-btneck-overhaul-20220525",
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

if dataset == "STRATH":
    raw_df = raw_df[raw_df["resample_factor"] == 50]

bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
bae_type_map = {
    "ae": "Deterministic AE",
    "ens": "BAE-Ensemble",
    "mcd": "BAE-MCD",
    "vi": "BAE-BBB",
    "sghmc": "BAE-SGHMC",
    "vae": "VAE",
    "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
}

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

# raw_df = raw_df[raw_df["layer_norm"] == "none"]
# raw_df = raw_df[raw_df["current_epoch"] == 300]

##======OPTIMISE PARAMS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "BTNECK_TYPE"],
    optim_params=[
        "current_epoch",
        "latent_factor",
        "skip",
        "layer_norm",
        "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
    ],
    # optim_params=["latent_factor", "skip"],
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

# ================INF BAE=================
# handle inf
# inf_paths = {
#     "ZEMA": "../results/zema-inf-20220515",
#     "STRATH": "../results/strath-inf-20220515",
#     "ODDS": "../results/odds-inf-20220515",
#     "Images": "../results/images-inf-20220515",
# }

## handle inf
inf_paths = {
    "ZEMA": "../results/zema-inf-noise-revamp-20220528",
    "STRATH": "../results/strath-inf-noise-revamp-20220528",
    "ODDS": "../results/odds-inf-noise-repair-20220527",
    "Images": "../results/images-inf-noise-20220527",
}
inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC.csv")

if "noise_scale" in inf_df.columns:
    inf_df = inf_df[(inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")]
    # inf_df = inf_df.drop(["noise_scale", "noise_type"])

inf_df["bae_type"] = "bae_inf"
inf_df["BTNECK_TYPE"] = "B"
inf_df["BTNECK_TYPE+INF"] = "INF"
# inf_df["HAS_BTNECK"] = "INF"
inf_df["HAS_BTNECK"] = "NO"

inf_df["current_epoch"] = 1
inf_df["latent_factor"] = 1
inf_df["layer_norm"] = inf_df["norm"]

# inf_df = inf_df[inf_df["num_layers"] == 2]  # limit num layers?
# inf_df["dataset"] = inf_df["dataset"] + ".mat"  # for odds?
fixed_inf_params = ["bae_type", "BTNECK_TYPE", "HAS_BTNECK"]
optim_inf_params = [
    "W_std",
    "diag_reg",
    "norm",
    "num_layers",
    "activation",
    "skip",
    "current_epoch",
    "latent_factor",
    "layer_norm",
]
optim_inf_df = apply_optim_df(
    inf_df,
    fixed_params=fixed_inf_params,
    optim_params=optim_inf_params,
    perf_key="E_AUROC",
    target_dim_col=tasks_col_name,
)
optim_df_combined = optim_df.append(optim_inf_df)

# report mean
optim_inf_df_mean = (
    optim_inf_df.groupby([tasks_col_name] + fixed_inf_params).mean().reset_index()
)

##======PLOT BOXPLOT===========
figsize = (6, 3)
showfliers = False
for target_dim in optim_df[tasks_col_name].unique():
    temp_df = optim_df[optim_df[tasks_col_name] == target_dim]
    bae_type_order = ["ae", "vae", "mcd", "vi", "ens"]
    x = "bae_type"
    hue = "BTNECK_TYPE"
    x_order = bae_type_order
    hue_order = ["A", "B", "C", "D"]
    y = "E_" + metric_key_word

    # ==== BARPLOT ====

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bplot = sns.barplot(
        x="bae_type",
        hue="BTNECK_TYPE",
        y="E_" + metric_key_word,
        data=temp_df,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        hue_order=["A", "B", "C", "D"],
        order=x_order,
    )

    # bplot = sns.barplot(
    #     x="bae_type",
    #     hue="HAS_BTNECK",
    #     y="E_" + metric_key_word,
    #     data=temp_df,
    #     capsize=0.1,
    #     ax=ax,
    #     errwidth=1.5,
    #     ci=95,
    #     hue_order=["YES", "NO"],
    #     order=x_order,
    # )

    # ====Labels====
    ax.set_xlabel("")
    ax.set_ylabel(metric_key_word)
    # ymin = temp_df["E_" + metric_key_word].min() - 0.05
    temp_group_mean_ = temp_df.groupby(["bae_type", "BTNECK_TYPE"]).mean()[
        "E_" + metric_key_word
    ]
    ymin = temp_group_mean_.min() - 0.05
    ymax = np.clip(temp_group_mean_.max() + 0.05, 0, 1.0)
    print(ymin)
    # ax.set_ylim(ymin, 1.0)
    ax.set_ylim(ymin, ymax)
    fig.canvas.manager.set_window_title(str(target_dim))

# =========TABLE OF RESULTS PER TARGET DIM==========
# The cosmetic operations are as follows:
# 1. Rearrange sequences
# 2. Replace BAE Types and Likelihood labels
def convert_latex_csv(df_mean, df_sem, csv_name, save_csv=False):
    ## CONVERT TO CSV TABLE WITH Mean $\pm$ SEM
    rearrranged_df_mean = rearrange_df(
        df_mean,
        "bae_type",
        labels=bae_type_order,
    )
    rearrranged_df_sem = rearrange_df(
        df_sem,
        "bae_type",
        labels=bae_type_order,
    )
    rearrranged_df_mean = replace_df_label_maps(
        rearrranged_df_mean, bae_type_map, col="bae_type"
    )
    rearrranged_df_sem = replace_df_label_maps(
        rearrranged_df_sem, bae_type_map, col="bae_type"
    )

    csv_table = rearrranged_df_mean.copy()
    display_format = "{:.0" + str(round_deci) + "f}"
    csv_table.iloc[:, 1:] = (
        rearrranged_df_mean.iloc[:, 1:].applymap(display_format.format)
        + "$\pm$"
        + rearrranged_df_sem.iloc[:, 1:].applymap(display_format.format)
    )
    if save_csv:
        csv_table.round(round_deci).to_csv(
            csv_name,
            index=False,
        )
    return csv_table


groupby_cols = ["bae_type", "BTNECK_TYPE"]
collect_pivot_abcd_dfs = []

for target_dim in optim_df_combined[tasks_col_name].unique():
    res_groupby = optim_df_combined[
        optim_df_combined[tasks_col_name] == target_dim
    ].groupby(groupby_cols)
    res_groupby_mean = res_groupby.mean()["E_" + metric_key_word].reset_index()
    res_groupby_sem = res_groupby.sem()["E_" + metric_key_word].reset_index()

    # convert to pivot table per task
    pivot_df_mean = res_groupby_mean.pivot(
        index="bae_type", columns="BTNECK_TYPE", values="E_" + metric_key_word
    ).reset_index()
    pivot_df_sem = res_groupby_sem.pivot(
        index="bae_type", columns="BTNECK_TYPE", values="E_" + metric_key_word
    ).reset_index()

    # SUMMARY ROW: Mean and ATE of tasks
    pivot_df_mean_ate = append_mean_ate_rows(
        pivot_df_mean, label_col="bae_type", baseline_col="A"
    )

    if save_csv:
        csv_name = (
            "tables/"
            + dataset
            + "-"
            + str(target_dim)
            + "-"
            + metric_key_word
            + "-BTNECK-LATEX.csv"
        )
        convert_latex_csv(
            pivot_df_mean,
            pivot_df_sem,
            csv_name=csv_name,
            save_csv=save_csv,
        )
        pivot_df_mean_ate.round(round_deci).to_csv(
            "tables/"
            + dataset
            + "-"
            + str(target_dim)
            + "-"
            + metric_key_word
            + "-BTNECK-RAW.csv",
            index=False,
        )
    # Collect to make final table
    collect_pivot_abcd_dfs.append(pivot_df_mean)

    # print results
    print("RESULT PER TASK:" + str(target_dim))
    pprint(pivot_df_mean_ate.round(round_deci))

    # =====barplot vs heatmap=====
    test1 = pivot_df_mean.iloc[:, 1:]
    heatmap_ = (test1.values - test1.values[0, 0]) * 100
    # heatmap_ = test1.values - np.expand_dims(test1.values[:, 0], 1)
    # heatmap_ = test1

    # make df for barplot
    heatmap_df = pivot_df_mean.copy()
    heatmap_df.iloc[:, 1:] = heatmap_
    # heatmap_df.iloc[:, 1:] = np.exp(heatmap_)
    heatmap_df = pd.melt(
        heatmap_df,
        id_vars="bae_type",
        value_vars=list(heatmap_df.columns[1:]),  # list of days of the week
        var_name="has_btneck",
        value_name="E_AUROC",
    )

    # plt.figure()
    # sns.heatmap(heatmap_, annot=True, cmap="magma")

    fig, ax = plt.subplots(1, 1)
    bplot = sns.barplot(
        y="bae_type", hue="has_btneck", x="E_AUROC", data=heatmap_df, ax=ax, zorder=3
    )
    ax.xaxis.grid(alpha=0.25, zorder=0)
    bplot.legend_.remove()

    #
    ae_baselines = heatmap_[0]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, baseline in enumerate(ae_baselines):
        plt.axvline(
            x=baseline, linestyle="--", linewidth=1.2, alpha=0.7, color=colors[i]
        )
