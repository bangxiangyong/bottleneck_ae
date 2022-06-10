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
    inf_df = inf_df[(inf_df["noise_scale"] == 0)]
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

# Aggregated tables for all tasks
collect_pivot_abcd_dfs = pd.concat(collect_pivot_abcd_dfs).groupby(["bae_type"])
collect_pivot_abcd_dfs_mean = collect_pivot_abcd_dfs.mean().reset_index()
collect_pivot_abcd_dfs_sem = collect_pivot_abcd_dfs.sem().reset_index()
collect_pivot_abcd_dfs_mean = append_mean_ate_rows(
    collect_pivot_abcd_dfs_mean, label_col="bae_type", baseline_col="A"
)

if save_csv:
    csv_name = "tables/" + dataset + "-AGG-" + metric_key_word + "-BTNECK-LATEX.csv"
    convert_latex_csv(
        collect_pivot_abcd_dfs_mean,
        collect_pivot_abcd_dfs_sem,
        csv_name=csv_name,
        save_csv=save_csv,
    )
    collect_pivot_abcd_dfs_mean.round(round_deci).to_csv(
        "tables/" + dataset + "-AGG-" + metric_key_word + "-BTNECK-RAW.csv", index=False
    )

# print results
print("AGGREGATED RESULTS:")
pprint(collect_pivot_abcd_dfs_mean.round(round_deci))

print("===========================================")
print("==========BEST BTNECK VS NON-BTNECK=========")

# ==========BEST BTNECK VS NONBTNECK==========
groupby_cols = ["bae_type", "HAS_BTNECK"]
optim_df_hasbtneck = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "HAS_BTNECK"],
    optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
    # optim_params=["latent_factor", "skip"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)
# append inf bae
optim_df_hasbtneck = optim_df_hasbtneck.append(optim_inf_df)

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


# ==============OPTIM BAR PLOTS-20220501====================

compare_col = "HAS_BTNECK"
# compare_col = "BTNECK_TYPE+INF"

# raw_df.append(optim_inf_df)
# Bar plot optimising over bae types, leaving only likelihood
# groupby_cols = ["bae_type", compare_col]
optim_df_hasbtneck = apply_optim_df(
    raw_df.append(optim_inf_df),
    fixed_params=[compare_col],
    optim_params=["current_epoch", "latent_factor", "skip", "layer_norm", "bae_type"],
    # optim_params=["latent_factor", "skip"],
    perf_key="E_" + metric_key_word,
    target_dim_col=tasks_col_map[dataset],
)
## Cosmetic modifications for plotting
# optim_df_hasbtneck = rearrange_df(
#     optim_df_hasbtneck,
#     "full_likelihood",
#     labels=["bernoulli", "cbernoulli", "mse", "std-mse", "static-tgauss"],
# )
# ll_map = {
#     "bernoulli": "Bernoulli",
#     "cbernoulli": "C-Bernoulli",
#     "mse": "Gaussian",
#     "std-mse": "Gaussian (Z-std)",
#     "static-tgauss": "Trunc. Gaussian",
# }
# target_map = {
#     "ZEMA": {0: "Cooler", 1: "Pump", 2: "Valve", 3: "Accumulator"},
#     "STRATH": {2: "Diameter"},
#     "Images": {label: label for label in optim_df_v2[tasks_col_map[dataset]].unique()},
#     "ODDS": {
#         label: str(label).title()
#         for label in optim_df_v2[tasks_col_map[dataset]].unique()
#     },
# }
# optim_df_v2 = replace_df_label_maps(optim_df_v2, ll_map, col="full_likelihood")
# optim_df_v2 = replace_df_label_maps(
#     optim_df_v2, target_map[dataset], col=tasks_col_map[dataset]
# )
figsize = (9, 5)
fig, ax = plt.subplots(1, 1, figsize=figsize)
bplot = sns.barplot(
    x=tasks_col_map[dataset],
    hue=compare_col,
    y="E_" + metric_key_word,
    data=optim_df_hasbtneck,
    capsize=0.1,
    ax=ax,
    errwidth=1.5,
    ci=95,
    palette="tab10",
)
# bplot = sns.boxplot(
#     x=tasks_col_map[dataset],
#     hue=compare_col,
#     y="E_" + metric_key_word,
#     data=optim_df_hasbtneck,
#     ax=ax,
#     palette="tab10",
# )
ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.5)
ax.set_ylim(ymin=0.0 if dataset == "Images" else 0.45, ymax=1.05)
ax.set_xlabel("Task")
ax.set_ylabel(metric_key_word)

# legends
l = ax.legend(fontsize="small")
l.set_title("")
fig.tight_layout()
if save_fig:
    fig.savefig("plots/" + dataset + "-BTNECK.png", dpi=500)
