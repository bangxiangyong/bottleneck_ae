# 1. Aggregates the results on LL and posterior sampling methods
# 2. Outputs the table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    append_mean_ate_rows,
    replace_df_label_maps,
    rearrange_df,
)
from pprint import pprint
import numpy as np

plt.rcParams.update({"font.size": 15})

# ===========OPTIONS============
## select dataset
# dataset = "ZEMA"
# dataset = "STRATH"
dataset = "ODDS"
# dataset = "Images"

metric_key_word = "AUROC"
# metric_key_word = "AVGPRC"
round_deci = 3


# save csv
# save_csv = True
save_csv = False
save_fig = True
# save_fig = False
csv_folder = "tables/"
# =================================
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]

## define path to load the csv
paths = {
    # "ZEMA": "../results/zema-ll-20220415",
    # "STRATH": "../results/strath-ll-20220415",
    # "ODDS": "../results/odds-ll-20220414",
    # "ZEMA": "../results/zema-reboot-ll-20220418",
    # "ZEMA": "../results/zema-stgauss-ll-20220428",
    # "STRATH": "../results/strath-reboot-ll-20220418",
    # "STRATH": "../results/strath-stgauss-ll-20220428",
    # "STRATH": "../results/strath-topk-reboot-20220418",
    # "ZEMA": "../results/zema-full-ll-20220417",
    # "STRATH": "../results/strath-full-ll-20220417",
    # "ODDS": "../results/odds-reboot-ll-20220420",
    # "ODDS": "../results/odds-vowel-fix-v2-20220426",
    # "ODDS": "../results/odds-stgauss-ll-20220429",
    # "Images": "../results/images-ll-20220420",
    # "Images": "../results/images-ll-reboot-20220422-incomp",
    # "Images": "../results/images-svhn-ll-fix-20220426",
    # "Images": "../results/images-stgauss-ll-20220429",
    "ZEMA": "../results/zema-full-ll-20220514",
    "STRATH": "../results/strath-full-ll-20220514",
    "ODDS": "../results/odds-full-ll-20220514",
    "Images": "../results/images-full-ll-20220514",
}


path = paths[dataset]

## ==========COSMETICS==============
## LABEL MAPS AND ORDER
ll_order = [
    "bernoulli",
    "cbernoulli",
    "mse",
    # "homo-gauss",
    # "homo-tgauss",
    "std-mse",
    "static-tgauss",
]
bae_type_order = ["ae", "vae", "mcd", "vi", "ens"]
bae_type_map = {
    "ae": "Deterministic AE",
    "ens": "BAE-Ensemble",
    "mcd": "BAE-MCD",
    "vi": "BAE-BBB",
    "sghmc": "BAE-SGHMC",
    "vae": "VAE",
}
# ll_type_map = {
#     "bernoulli": "Ber($\hat{{x}}^*$)",
#     "cbernoulli": "C-Ber($\hat{{x}}^*$)",
#     "mse": "N($\hat{{x}}^*$,1)",
# }
ll_type_map = {
    "bernoulli": "Bernoulli",
    "cbernoulli": "C-Bernoulli",
    "mse": "Gaussian",
    "std-mse": "Gaussian (Z-std)",
    "static-tgauss": "Trunc. Gaussian",
}

## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word)

##======OPTIMISE BY TRAIN EPOCHS=========
groupby_cols = ["bae_type", "full_likelihood"]
optim_df = apply_optim_df(
    raw_df,
    fixed_params=groupby_cols,
    optim_params=["current_epoch"],
    perf_key="E_" + metric_key_word,
    target_dim_col=tasks_col_map[dataset],
)

# =========SAVE OPTIM PARAMS FOR OTHER PURPOSES============
optimised_params = (
    optim_df[["bae_type", "full_likelihood", "current_epoch", tasks_col_map[dataset]]]
    .drop_duplicates()
    .reset_index(drop=True)
).to_csv(dataset + "_optim_params.csv")
# =========TABLE OF RESULTS PER TARGET DIM==========
perf_cols = [
    "E_" + metric_key_word,
    "V_" + metric_key_word,
    "WAIC_" + metric_key_word,
    "VX_" + metric_key_word,
]
selected_cols = groupby_cols + perf_cols

collect_dfs_mean = {}
collect_dfs_sem = {}
all_tasks = optim_df[tasks_col_name].unique()  ## name of task per dataset
num_tasks = len(all_tasks)  ## total number of tasks
for target_dim in all_tasks:
    res_groupby_mean = (
        optim_df[optim_df[tasks_col_name] == target_dim]
        .groupby(groupby_cols)
        .mean()
        .reset_index()
    )[selected_cols]
    res_groupby_sem = (
        optim_df[optim_df[tasks_col_name] == target_dim]
        .groupby(groupby_cols)
        .sem()
        .reset_index()
    )[selected_cols]

    # collect results
    collect_dfs_mean.update({target_dim: res_groupby_mean.copy()})
    collect_dfs_sem.update({target_dim: res_groupby_sem.copy()})

    # print results
    print("RESULT PER TASK:" + str(target_dim))
    pprint(res_groupby_mean.round(round_deci))

# Aggregate mean and sem across all tasks
if num_tasks > 1:
    collect_dfs_mean_ = pd.concat([df for df in collect_dfs_mean.values()])
    aggregate_dfs_mean = collect_dfs_mean_.groupby(groupby_cols).mean().reset_index()
    aggregate_dfs_sem = collect_dfs_mean_.groupby(groupby_cols).sem().reset_index()
    pprint(aggregate_dfs_mean.round(round_deci))

# ==============OPTIM BAR PLOTS-20220430====================
# Bar plot optimising over bae types, leaving only likelihood
groupby_cols = ["full_likelihood"]
optim_df_v2 = apply_optim_df(
    raw_df,
    fixed_params=groupby_cols,
    optim_params=["current_epoch", "bae_type"],
    perf_key="E_" + metric_key_word,
    target_dim_col=tasks_col_map[dataset],
)
## Cosmetic modifications for plotting
optim_df_v2 = rearrange_df(
    optim_df_v2,
    "full_likelihood",
    labels=["bernoulli", "cbernoulli", "mse", "std-mse", "static-tgauss"],
)
ll_map = {
    "bernoulli": "Bernoulli",
    "cbernoulli": "C-Bernoulli",
    "mse": "Gaussian",
    "std-mse": "Gaussian (Z-std)",
    "static-tgauss": "Trunc. Gaussian",
}
target_map = {
    "ZEMA": {0: "Cooler", 1: "Pump", 2: "Valve", 3: "Accumulator"},
    "STRATH": {2: "Radial forge"},
    "Images": {label: label for label in optim_df_v2[tasks_col_map[dataset]].unique()},
    "ODDS": {
        label: str(label).title()
        for label in optim_df_v2[tasks_col_map[dataset]].unique()
    },
}
optim_df_v2 = replace_df_label_maps(optim_df_v2, ll_map, col="full_likelihood")
optim_df_v2 = replace_df_label_maps(
    optim_df_v2, target_map[dataset], col=tasks_col_map[dataset]
)
figsize = (16, 3.5)
fig, ax = plt.subplots(1, 1, figsize=figsize)
bplot = sns.barplot(
    x=tasks_col_map[dataset],
    hue="full_likelihood",
    y="E_" + metric_key_word,
    data=optim_df_v2,
    capsize=0.1,
    ax=ax,
    errwidth=1.5,
    ci=95,
    palette="tab10",
)
ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.5)
ax.set_ylim(ymin=0.0 if dataset == "Images" else 0.45, ymax=1.05)
ax.set_xlabel("Task")
ax.set_ylabel(metric_key_word)

# legends
# l = ax.legend(fontsize="x-small")
l = ax.legend(fontsize="x-small", loc="lower center")
l.set_title("")
fig.tight_layout()
if save_fig:
    fig.savefig("plots/" + dataset + "-LL.png", dpi=500)
# ==========APPLY COSMETICS TO LATEX AND COMBINE MEAN+-SEM===========
# The cosmetic operations are as follows:
# 1. Rearrange sequences
# 2. Replace BAE Types and Likelihood labels
def convert_ll_latex_csv(df_mean, df_sem, csv_name):
    ## COSMETIC 1: rearrange columns and rows
    aggregated_df_mean = df_mean[selected_cols]
    aggregated_df_sem = df_sem[selected_cols]
    rearrranged_df_mean = []
    rearrranged_df_sem = []
    for bae_type in bae_type_order:
        for ll in ll_order:
            rearrranged_df_mean.append(
                aggregated_df_mean[
                    (aggregated_df_mean["bae_type"] == bae_type)
                    & (aggregated_df_mean["full_likelihood"] == ll)
                ]
            )
            rearrranged_df_sem.append(
                aggregated_df_sem[
                    (aggregated_df_sem["bae_type"] == bae_type)
                    & (aggregated_df_sem["full_likelihood"] == ll)
                ]
            )
    rearrranged_df_mean = pd.concat(rearrranged_df_mean).reset_index(drop=True)
    rearrranged_df_sem = pd.concat(rearrranged_df_sem).reset_index(drop=True)

    ## COSMETIC 2: Rename variables
    rearrranged_df_mean = replace_df_label_maps(rearrranged_df_mean, bae_type_map)
    rearrranged_df_mean = replace_df_label_maps(rearrranged_df_mean, ll_type_map)
    rearrranged_df_sem = replace_df_label_maps(rearrranged_df_sem, bae_type_map)
    rearrranged_df_sem = replace_df_label_maps(rearrranged_df_sem, ll_type_map)

    ## CONVERT TO CSV TABLE WITH Mean $\pm$ SEM
    csv_table = rearrranged_df_mean.copy()
    display_format = "{:.0" + str(round_deci) + "f}"
    csv_table.loc[:, perf_cols] = (
        rearrranged_df_mean[perf_cols].applymap(display_format.format)
        + "$\pm$"
        + rearrranged_df_sem[perf_cols].applymap(display_format.format)
    )
    if save_csv:
        csv_table.to_csv(csv_name)


if save_csv:
    # Table for each task
    for task in collect_dfs_mean.keys():
        # collect results
        df_mean = collect_dfs_mean[task]
        df_sem = collect_dfs_sem[task]
        convert_ll_latex_csv(
            df_mean,
            df_sem,
            csv_name=csv_folder
            + dataset
            + "-"
            + str(task)
            + "-"
            + metric_key_word
            + "-table.csv",
        )

    # For aggregated results
    if num_tasks > 1:
        convert_ll_latex_csv(
            aggregate_dfs_mean,
            aggregate_dfs_sem,
            csv_name=csv_folder + dataset + "-AGG-" + metric_key_word + "-table.csv",
        )

# ==========================
#  Get all optimised current epoch
# optim_params = (
#     optim_df.groupby(groupby_cols + ["dataset"] + ["current_epoch"])
#     .mean()
#     .reset_index()
# )

# ==========================

## COSMETIC: Rename variables
# for key, val in bae_type_map.items():
#     rearrranged_df = rearrranged_df.replace(key, val)
# for key, val in ll_type_map.items():
#     rearrranged_df = rearrranged_df.replace(key, val)
# rearrranged_df = rearrranged_df.round(3)


# ==========BEST BTNECK VS NONBTNECK==========


#
# ## results will be group by these columns
# ## if aggregate_all_tasks, then further aggregate by the task column
# groupby_cols = ["bae_type", "full_likelihood"]
# if not aggregate_all_tasks:
#     groupby_cols += [tasks_col_name]
#
# ## aggregate results by bae_type and ll
# ## apply aggregate function mean or sem (on random seeds)
# perf_cols = [
#     "E_" + metric_key_word,
#     "V_" + metric_key_word,
#     "WAIC_" + metric_key_word,
#     "VX_" + metric_key_word,
# ]
# aggregated_df = optim_df.groupby(groupby_cols).mean()[perf_cols].reset_index()
#
# ## Postprocessing starts here
# ## COSMETIC: rearrange columns and rows
# aggregated_df = aggregated_df[groupby_cols + perf_cols]
# rearrranged_df = []
# for bae_type in bae_type_order:
#     for ll in ll_order:
#         rearrranged_df.append(
#             aggregated_df[
#                 (aggregated_df["bae_type"] == bae_type)
#                 & (aggregated_df["full_likelihood"] == ll)
#             ]
#         )
# rearrranged_df = pd.concat(rearrranged_df).reset_index(drop=True)
#
# ## COSMETIC: Rename variables
# for key, val in bae_type_map.items():
#     rearrranged_df = rearrranged_df.replace(key, val)
# for key, val in ll_type_map.items():
#     rearrranged_df = rearrranged_df.replace(key, val)
# rearrranged_df = rearrranged_df.round(3)
#
# ## Save to csv
# if aggregate_all_tasks:
#     rearrranged_df.to_csv(dataset + "AGG" + csv_suffix, index=False)
#     print("Aggregated DF results")
#     print(rearrranged_df)
# else:
#     for task in rearrranged_df[tasks_col_name].unique():
#         rearrranged_df_bytask = rearrranged_df[rearrranged_df[tasks_col_name] == task]
#         rearrranged_df_bytask.to_csv(
#             dataset + "_" + str(task) + csv_suffix, index=False
#         )
#         print("TASK: " + str(task))
#         print(rearrranged_df_bytask)
#
# ## ========BOX PLOTS=========
# # x = "full_likelihood"
# # hue = "bae_type"
# # x_order = ll_order
#
# x = "bae_type"
# hue = "full_likelihood"
# x_order = bae_type_order
#
# y = "E_" + metric_key_word
# if aggregate_all_tasks:
#     fig, ax = plt.subplots(1, 1)
#     sns.boxplot(x=x, y=y, hue=hue, ax=ax, data=optim_df, order=x_order)
# else:
#     for task in optim_df[tasks_col_name].unique():
#         plot_df = optim_df[optim_df[tasks_col_name] == task]
#         fig, ax = plt.subplots(1, 1)
#         sns.boxplot(x=x, y=y, hue=hue, ax=ax, data=plot_df, order=x_order)
#         fig.canvas.manager.set_window_title(str(task))
#
# # X by tasks
# fig, ax = plt.subplots(1, 1)
# sns.boxplot(x=tasks_col_name, y=y, hue="full_likelihood", ax=ax, data=optim_df)
# fig.canvas.manager.set_window_title("ALL in")
#
# ## ===========================
