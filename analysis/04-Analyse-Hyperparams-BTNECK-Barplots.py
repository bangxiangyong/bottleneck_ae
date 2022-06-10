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
metric_col = "E_" + metric_key_word
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
raw_df = []
for dataset, path in paths.items():
    tasks_col_name = tasks_col_map[dataset]
    raw_df_ = concat_csv_files(path, key_word=metric_key_word)
    if dataset == "STRATH":
        raw_df_ = raw_df_[raw_df_["resample_factor"] == 50]
    raw_df_["dataset-targetdim"] = dataset + "-" + raw_df_[tasks_col_name].astype(str)
    raw_df.append(raw_df_)
raw_df = pd.concat(raw_df)

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

##======OPTIMISE PARAMS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "BTNECK_TYPE"],
    optim_params=[
        "current_epoch",
        "latent_factor",
        "skip",
        "layer_norm",
        "n_dense_layers",
        # "n_conv_layers",
    ],
    perf_key=metric_col,
    target_dim_col="dataset-targetdim",
)

# ===========BAE INF===============
## handle inf
inf_paths = {
    "ZEMA": "../results/zema-inf-noise-revamp-20220528",
    "STRATH": "../results/strath-inf-noise-revamp-20220528",
    "ODDS": "../results/odds-inf-noise-repair-20220527",
    "Images": "../results/images-inf-noise-20220527",
}

## start reading csv
inf_df = []
for dataset, path in inf_paths.items():
    tasks_col_name = tasks_col_map[dataset]
    inf_df_ = concat_csv_files(results_folder=path, key_word=metric_key_word)
    # if dataset == "STRATH":
    #     inf_df_ = inf_df_[inf_df_["resample_factor"] == 50]
    inf_df_["dataset-targetdim"] = dataset + "-" + inf_df_[tasks_col_name].astype(str)
    inf_df.append(inf_df_)
inf_df = pd.concat(inf_df)

if "noise_scale" in inf_df.columns:
    inf_df = inf_df[
        (inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")
    ].reset_index(drop=True)

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
# fixed_inf_params = ["bae_type", "BTNECK_TYPE", "HAS_BTNECK"]
fixed_inf_params = ["bae_type", "HAS_BTNECK"]
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
    target_dim_col="dataset-targetdim",
)
optim_df_combined = optim_df.append(optim_inf_df)

print("===========================================")
print("==========BEST BTNECK VS NON-BTNECK=========")

# ==========BEST BTNECK VS NONBTNECK==========
groupby_cols = ["bae_type", "HAS_BTNECK"]
optim_df_hasbtneck = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "HAS_BTNECK"],
    optim_params=[
        "current_epoch",
        "latent_factor",
        "skip",
        "layer_norm",
        "n_conv_layers" if dataset != "ODDS" else "n_dense_layers",
    ],
    perf_key=metric_col,
    target_dim_col="dataset-targetdim",
)
# append inf bae
optim_df_hasbtneck = optim_df_hasbtneck.append(optim_inf_df)

# Labels
dataset_labels = {
    "Images-CIFAR": "CIFAR vs SVHN",
    "Images-FashionMNIST": "F.MNIST vs MNIST",
    "ODDS-cardio": "ODDS (Cardio)",
    "ODDS-ionosphere": "ODDS (Ionosphere)",
    "ODDS-lympho": "ODDS (Lympho)",
    "ODDS-optdigits": "ODDS (Optdigits)",
    "ODDS-pendigits": "ODDS (Pendigits)",
    "ODDS-pima": "ODDS (Pima)",
    "ODDS-thyroid": "ODDS (Thyroid)",
    "ODDS-vowels": "ODDS (Vowels)",
    "ZEMA-0": "ZeMA(Cooler)",
    "ZEMA-1": "ZeMA(Valve)",
    "ZEMA-2": "ZeMA(Pump)",
    "ZEMA-3": "ZeMA(Accumulator)",
    "STRATH-2": "STRATH(Radial Forge)",
}
dataset_subsets_i = 0
### Select dataset subsets to be plotted
datasets_range = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)][
    dataset_subsets_i
]
datasets = np.array(list(dataset_labels.keys()))[datasets_range]
n_plots = 5
figsize = (16, 4)
fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
axes = axes.flatten()
# for target_dim in optim_df_hasbtneck[tasks_col_name].unique():
for i, dataset_targetdim in enumerate(datasets):
    ax = axes[i]
    bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
    res_groupby = (
        optim_df_hasbtneck[optim_df_hasbtneck["dataset-targetdim"] == dataset_targetdim]
        .groupby(groupby_cols)
        .mean()[metric_col]
        .reset_index()
    )

    # ========PLOT BEST BTNECK========

    bplot = sns.barplot(
        x="bae_type",
        hue="HAS_BTNECK",
        y=metric_col,
        data=optim_df_hasbtneck[
            optim_df_hasbtneck["dataset-targetdim"] == dataset_targetdim
        ],
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        # hatch="///",
        hue_order=["YES", "NO"],
        # order=bae_type_order,
    )
    # ====Labels====
    ax.set_xlabel("")
    ax.set_ylabel(metric_key_word)
    ymin = res_groupby[metric_col].min() - 0.05
    ymax = np.clip(res_groupby[metric_col].max() + 0.05, 0, 1.0)
    print(ymin)
    ax.set_ylim(ymin, ymax)
    fig.canvas.manager.set_window_title(str(dataset_targetdim))
    ax.set_title(str(dataset_targetdim))
