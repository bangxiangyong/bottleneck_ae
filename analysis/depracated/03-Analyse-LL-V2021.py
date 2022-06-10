# 1. Aggregates the results on LL and posterior sampling methods
# 2. Outputs the table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from thesis_experiments.util_analyse import concat_csv_files, apply_optim_df
from pprint import pprint

# ===========OPTIONS============
## select dataset
# dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
dataset = "Images"

metric_key_word = "AUROC"
csv_suffix = "_" + metric_key_word + "_reboot_ll.csv"

# ll_order = ["bernoulli", "cbernoulli", "mse"]
ll_order = [
    "bernoulli",
    "cbernoulli",
    "mse",
    "homo-gauss",
    "homo-tgauss",
]
bae_type_order = ["ae", "vae", "mcd", "vi", "ens"]

## choice to output a table for each task or aggregate all tasks under a single table
aggregate_all_tasks = False  ## aggregate/split results by target_dim or tasks
# aggregate_all_tasks = True  ## aggregate/split results by target_dim or tasks
# =================================
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False


## define path to load the csv
paths = {
    # "ZEMA": "../results/zema-ll-20220415",
    # "STRATH": "../results/strath-ll-20220415",
    # "ODDS": "../results/odds-ll-20220414",
    "ZEMA": "../results/zema-reboot-ll-20220418",
    "STRATH": "../results/strath-reboot-ll-20220418",
    # "STRATH": "../results/strath-topk-reboot-20220418",
    # "ZEMA": "../results/zema-full-ll-20220417",
    # "STRATH": "../results/strath-full-ll-20220417",
    "ODDS": "../results/odds-reboot-ll-20220420",
    "Images": "../results/images-ll-20220420",
}
path = paths[dataset]

## define maps
bae_type_map = {
    "ae": "Deterministic AE",
    "ens": "BAE-Ensemble",
    "mcd": "BAE-MCD",
    "vi": "BAE-BBB",
    "sghmc": "BAE-SGHMC",
    "vae": "VAE",
}

ll_type_map = {
    "bernoulli": "Ber($\hat{{x}}^*$)",
    "cbernoulli": "C-Ber($\hat{{x}}^*$)",
    "mse": "N($\hat{{x}}^*$,1)",
}

## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word)

##======OPTIMISE BY TRAIN EPOCHS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "full_likelihood"],
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)
##=======================================

## results will be group by these columns
## if aggregate_all_tasks, then further aggregate by the task column
groupby_cols = ["bae_type", "full_likelihood"]
if not aggregate_all_tasks:
    groupby_cols += [tasks_col_name]

## aggregate results by bae_type and ll
## apply aggregate function mean or sem (on random seeds)
perf_cols = [
    "E_" + metric_key_word,
    "V_" + metric_key_word,
    "WAIC_" + metric_key_word,
    "VX_" + metric_key_word,
]
aggregated_df = optim_df.groupby(groupby_cols).mean()[perf_cols].reset_index()

## Postprocessing starts here
## COSMETIC: rearrange columns and rows
aggregated_df = aggregated_df[groupby_cols + perf_cols]
rearrranged_df = []
for bae_type in bae_type_order:
    for ll in ll_order:
        rearrranged_df.append(
            aggregated_df[
                (aggregated_df["bae_type"] == bae_type)
                & (aggregated_df["full_likelihood"] == ll)
            ]
        )
rearrranged_df = pd.concat(rearrranged_df).reset_index(drop=True)

## COSMETIC: Rename variables
for key, val in bae_type_map.items():
    rearrranged_df = rearrranged_df.replace(key, val)
for key, val in ll_type_map.items():
    rearrranged_df = rearrranged_df.replace(key, val)
rearrranged_df = rearrranged_df.round(3)

## Save to csv
if aggregate_all_tasks:
    rearrranged_df.to_csv(dataset + "AGG" + csv_suffix, index=False)
    print("Aggregated DF results")
    print(rearrranged_df)
else:
    for task in rearrranged_df[tasks_col_name].unique():
        rearrranged_df_bytask = rearrranged_df[rearrranged_df[tasks_col_name] == task]
        rearrranged_df_bytask.to_csv(
            dataset + "_" + str(task) + csv_suffix, index=False
        )
        print("TASK: " + str(task))
        print(rearrranged_df_bytask)

## ========BOX PLOTS=========
# x = "full_likelihood"
# hue = "bae_type"
# x_order = ll_order

x = "bae_type"
hue = "full_likelihood"
x_order = bae_type_order

y = "E_" + metric_key_word
if aggregate_all_tasks:
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(x=x, y=y, hue=hue, ax=ax, data=optim_df, order=x_order)
else:
    for task in optim_df[tasks_col_name].unique():
        plot_df = optim_df[optim_df[tasks_col_name] == task]
        fig, ax = plt.subplots(1, 1)
        sns.boxplot(x=x, y=y, hue=hue, ax=ax, data=plot_df, order=x_order)
        fig.canvas.manager.set_window_title(str(task))

# X by tasks
fig, ax = plt.subplots(1, 1)
sns.boxplot(x=tasks_col_name, y=y, hue="full_likelihood", ax=ax, data=optim_df)
fig.canvas.manager.set_window_title("ALL in")

## ===========================
