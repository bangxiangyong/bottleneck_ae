# PLOTS THE ABCD BAR PLOTS

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

metric_key_word = "AUROC"
metric_col = "E_" + metric_key_word
dataset_subsets_i = 2
save_fig = True
# save_fig = False
save_csv = False
# save_csv = True
# show_legend = True
show_legend = False
## define path to load the csv and mappings related to datasets
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
bottleneck_map = {
    "A": "Undercomplete \N{MINUS SIGN}skip (Bottlenecked)",
    "B": "Overcomplete \N{MINUS SIGN}skip (Not bottlenecked)",
    "C": "Undercomplete +skip (Not bottlenecked)",
    "D": "Overcomplete +skip (Not bottlenecked)",
    "INF": "Infinite-width (Not bottlenecked)",
}

optim_df_combined = []
for dataset in paths.keys():
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
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

    ##======OPTIMISE PARAMS=========
    optim_df = apply_optim_df(
        raw_df,
        fixed_params=["bae_type", "BTNECK_TYPE+INF"],
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

    # ================INF BAE=================
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
    fixed_inf_params = ["bae_type", "BTNECK_TYPE+INF", "HAS_BTNECK"]
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

    ## append into final results
    optim_df_combined_ = pd.concat([optim_df, optim_inf_df])
    optim_df_combined_["dataset_targetdim"] = (
        dataset + "-" + optim_df_combined_[tasks_col_name].astype(str)
    )
    optim_df_combined.append(optim_df_combined_)

optim_df_combined = pd.concat(optim_df_combined)

# ==========BARPLOT============
groupby_cols = ["bae_type", "BTNECK_TYPE+INF"]
bae_type_map = {
    "ae": "AE",
    "vae": "VAE",
    "mcd": "MCD",
    "vi": "BBB",
    "ens": "Ens",
    # "bae_inf": "BAE-" + r"$\infty$",
    "bae_inf": "BAE-" + r"$\infty$",
}

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
    "ZEMA-0": "ZeMA (Cooler)",
    "ZEMA-1": "ZeMA (Valve)",
    "ZEMA-2": "ZeMA (Pump)",
    "ZEMA-3": "ZeMA (Accumulator)",
    "STRATH-2": "STRATH (Radial Forge)",
}
### Select dataset subsets to be plotted
datasets_range = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)][
    dataset_subsets_i
]
datasets = np.array(list(dataset_labels.keys()))[datasets_range]

n_plots = 5
plt.rcParams.update({"font.size": 13})
# figsize = (18, 3) ## GOOOD
figsize = (18, 2)
fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
axes = axes.flatten()

for i, dataset_targetdim in enumerate(datasets):
    ax = axes[i]
    bae_type_order = list(bae_type_map.values())
    btneck_type_order = list(bottleneck_map.values())
    temp_df = optim_df_combined[
        optim_df_combined["dataset_targetdim"] == dataset_targetdim
    ]
    temp_df = temp_df.replace(list(bae_type_map.keys()), list(bae_type_map.values()))
    temp_df = temp_df.replace(
        list(bottleneck_map.keys()), list(bottleneck_map.values())
    )

    # ========PLOT BEST BTNECK========

    bplot = sns.barplot(
        x="bae_type",
        hue="BTNECK_TYPE+INF",
        # hue="bae_type",
        # x="BTNECK_TYPE+INF",
        y=metric_col,
        data=temp_df,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        # ci=95,
        ci=68,
        hue_order=btneck_type_order,
        order=bae_type_order,
        # hue_order=bae_type_order,
        # order=btneck_type_order,
    )
    bplot
    # ====Labels====
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(metric_key_word)
    else:
        ax.set_ylabel("")

    res_groupby = (
        temp_df.groupby(groupby_cols).mean()[metric_col].reset_index()
    )  # ymin-max

    # ===set y lim=======
    # ymin = res_groupby[metric_col].min() - 0.05
    # ymax = np.clip(res_groupby[metric_col].max() + 0.05, 0, 1.0)
    # print(ymin)
    # ax.set_ylim(ymin, ymax)

    # YTICKS AT 5 LEVELS
    # ================SET Y LIMS AND TICKS===================
    if dataset_targetdim != "ZEMA-0":
        ymin, ymax = ax.get_ylim()
        # fmt: off
        y_low = res_groupby[metric_col].min()
        # y_high = res_groupby[metric_col].max()
        y_high = np.clip(ymax,0,1)
        # fmt: on

        # set y lims
        y_min = y_low - 0.25 * (y_high - y_low)
        y_max = np.clip(y_high + 0.25 * (y_high - y_low), 0, 1)
        ax.set_ylim(y_min, y_max)

        # set y-ticks
        N = 5
        ymin, ymax = ax.get_ylim()
        new_ymin = (ymax - ymin) * 2 / 3
        ax.set_yticks(np.clip(np.round(np.linspace(ymin, ymax, N), 2), None, 1.0))

    # ===================
    fig.canvas.manager.set_window_title(str(dataset_targetdim))
    ax.set_title(dataset_labels[dataset_targetdim])
    # ax.tick_params(axis="x", rotation=45)
    ax.yaxis.grid(alpha=0.5)  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest

    # Legends
    ax.get_legend().remove()

fig.tight_layout(pad=0)

if show_legend:
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.332),
        fancybox=True,
        ncol=5,
    )

if save_fig:
    filename = "all-btneck-" + str(dataset_subsets_i) + "-ABCD.png"
    fig.savefig(filename, dpi=550)
    print("SAVED PNG:" + filename)
