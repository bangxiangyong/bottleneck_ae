from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    append_mean_ate_rows,
    rearrange_df,
    replace_df_label_maps,
)
import colorsys
import matplotlib.colors as mc
import matplotlib.pyplot as plt

metric_key_word = "AUROC"
metric_col = "E_" + metric_key_word
dataset_subsets_i = 0
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
optim_df_combined = []
for dataset in paths.keys():
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
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

    # raw_df = raw_df[raw_df["layer_norm"] == "none"]
    # raw_df = raw_df[raw_df["current_epoch"] == 300]
    raw_df["num_layers"] = raw_df[
        "n_dense_layers" if dataset == "ODDS" else "n_conv_layers"
    ]
    optim_df = apply_optim_df(
        raw_df,
        fixed_params=["BTNECK_TYPE+INF", "bae_type", "num_layers"],
        optim_params=[
            "current_epoch",
            "latent_factor",
            "skip",
            "layer_norm",
        ],
        perf_key="E_AUROC",
        target_dim_col=tasks_col_map[dataset],
    )

    # optim_df = apply_optim_df(
    #     raw_df,
    #     fixed_params=["bae_type", "BTNECK_TYPE"],
    #     optim_params=[
    #         "current_epoch",
    #         # "latent_factor",
    #         # "skip",
    #         "layer_norm",
    #     ],
    #     perf_key="E_" + metric_key_word,
    #     target_dim_col=tasks_col_map[dataset],
    # )
    # ================INF BAE=================
    # handle inf
    inf_paths = {
        # "ZEMA": "../results/zema-inf-20220515",
        # "STRATH": "../results/strath-inf-20220515",
        # "ODDS": "../results/odds-inf-20220515",
        # "Images": "../results/images-inf-20220515",
        "ZEMA": "../results/zema-inf-noise-revamp-20220528",
        "STRATH": "../results/strath-inf-noise-revamp-20220528",
        "ODDS": "../results/odds-inf-noise-repair-20220527",
        "Images": "../results/images-inf-noise-20220527",
    }
    inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC.csv")
    inf_df["bae_type"] = "bae_inf"
    inf_df["BTNECK_TYPE"] = "B"
    inf_df["BTNECK_TYPE+INF"] = "INF"
    inf_df["HAS_BTNECK"] = "INF"
    # inf_df["HAS_BTNECK"] = "NO"

    inf_df["current_epoch"] = 1
    inf_df["latent_factor"] = 1
    inf_df["layer_norm"] = inf_df["norm"]

    fixed_inf_params = ["bae_type", "BTNECK_TYPE+INF", "HAS_BTNECK"]
    optim_inf_params = [
        "W_std",
        "diag_reg",
        "norm",
        "activation",
        "skip",
        "current_epoch",
        "latent_factor",
        "layer_norm",
        "noise_scale",
        "noise_type",
        "num_layers",
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
# =========================================================================
# =======================BOX PLOTS OF ALL =================================
# ===========================================================================
groupby_cols = ["bae_type", "HAS_BTNECK"]
bae_type_map = {
    "ae": "AE",
    "vae": "VAE",
    "mcd": "MCD",
    "vi": "BBB",
    "ens": "Ens",
    # "bae_inf": "BAE-" + r"$\infty$",
    "bae_inf": "BAE-" + r"$\infty$",
}

bottleneck_map = {
    "A": "Undercomplete \N{MINUS SIGN}skip (Bottlenecked)",
    "B": "Overcomplete \N{MINUS SIGN}skip (Not bottlenecked)",
    "C": "Undercomplete +skip (Not bottlenecked)",
    "D": "Overcomplete +skip (Not bottlenecked)",
    "INF": "Infinite-width (Not bottlenecked)",
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
linewidth = 1.85
plt.rcParams.update({"font.size": 14})
figsize = (18, 2.5)
fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
axes = axes.flatten()
matplotlib.rcParams["axes.autolimit_mode"] = "round_numbers"
for dataset_i, dataset_targetdim in enumerate(datasets):
    ax = axes[dataset_i]

    bae_type_order = list(bae_type_map.values())
    temp_df = optim_df_combined[
        optim_df_combined["dataset_targetdim"] == dataset_targetdim
    ]
    temp_df = temp_df.replace(list(bae_type_map.keys()), list(bae_type_map.values()))
    temp_df = temp_df.replace(["YES", "NO"], ["Bottlenecked", "Not bottlenecked"])
    temp_df = temp_df.replace(
        list(bottleneck_map.keys()), list(bottleneck_map.values())
    )

    # ========PLOT BEST BTNECK========
    # bplot = sns.boxplot(
    #     x="dataset_targetdim",
    #     hue="BTNECK_TYPE+INF",
    #     y=metric_col,
    #     data=temp_df,
    #     capsize=0.1,
    #     ax=ax,
    #     errwidth=1.5,
    #     ci=95,
    #     # hue_order=["Bottlenecked", "Not bottlenecked"],
    #     # order=bae_type_order,
    # )

    flierprops = dict(markerfacecolor="0.75", markersize=2, linestyle="none")
    showfliers = False
    bplot = sns.boxplot(
        x="dataset_targetdim",
        hue="BTNECK_TYPE+INF",
        y=metric_col,
        data=temp_df,
        ax=ax,
        linewidth=linewidth,
        flierprops=flierprops,
        showfliers=showfliers,
        hue_order=list(bottleneck_map.values()),
    )
    ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.4)
    ax.set_xlabel("")
    if dataset_i == 0:
        ax.set_ylabel(metric_key_word)
    else:
        ax.set_ylabel("")

    # y axis formatting
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if dataset_targetdim == "ZEMA-0":
        ax.set_ylim(0.98, None)

    # legends
    if dataset_i == 0:
        l = ax.legend(fontsize="small")
        l.set_title("")
    else:
        ax.legend([], [], frameon=False)
    # change border color
    # https://stackoverflow.com/questions/55656683/change-seaborn-boxplot-line-rainbow-color
    def adjust_lightness(color, amount):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    for i, artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = adjust_lightness(artist.get_facecolor(), 0.7)
        artist.set_edgecolor(col)

        for j in range(i * 5, i * 5 + 5):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

    # ====Labels====
    # ax.set_xlabel("")
    # if i == 0:
    #     ax.set_ylabel(metric_key_word)
    # else:
    #     ax.set_ylabel("")

    # res_groupby = (
    #     temp_df.groupby(groupby_cols).mean()[metric_col].reset_index()
    # )  # ymin-max
    # ymin = res_groupby[metric_col].min() - 0.05
    # ymax = np.clip(res_groupby[metric_col].max() + 0.05, 0, 1.0)
    # print(ymin)
    # ax.set_ylim(ymin, ymax)
    fig.canvas.manager.set_window_title(str(dataset_targetdim))
    ax.set_title(dataset_labels[dataset_targetdim])
    # ax.tick_params(axis="x", rotation=45)
    ax.yaxis.grid(alpha=0.5)  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest

    # ax.set_xticks([])
    # ax.set_xticks([], minor=True)
    ax.set_xticks([])
    # Legends
    ax.get_legend().remove()
fig.tight_layout()

if save_fig:
    filename = "plots/ALL-BTNECK-BOXPLOTS-" + str(dataset_subsets_i) + "-v2.png"
    fig.savefig(filename, dpi=600)
    print("SAVED PNG:" + filename)

# fig.subplots_adjust(
#     top=0.787, bottom=0.106, left=0.054, right=0.987, hspace=0.2, wspace=0.255
# )
# if show_legend:
#     axes[2].legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.332),
#         fancybox=True,
#         ncol=5,
#     )
