# This script produces the main boxplot results by combining all tables

import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    rearrange_df,
    replace_df_label_maps,
)

mega_df_list = []
metric_key_word = "AUROC"
save_fig = False
# save_fig = True

## define path to load the csv and mappings related to datasets
paths = {
    # "ZEMA": "../results/zema-btneck-20220419",
    # "STRATH": "../results/STRATH-BOTTLENECKV3",
    # "ODDS": "../results/odds-btneck-20220422",
    # "Images": "../results/images-btneck-reboot-20220421",
    # "ZEMA": "../results/zema-btneck-20220515",
    # "STRATH": "../results/strath-btneck-20220515",
    # "ODDS": "../results/odds-btneck-20220515",
    # "Images": "../results/images-btneck-20220515",
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
# compare_col = "HAS_BTNECK"
compare_col = "BTNECK_TYPE+INF"

for dataset in ["STRATH", "ZEMA", "Images", "ODDS"]:
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
        fixed_params=["BTNECK_TYPE", "bae_type", "num_layers"],
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

    fixed_inf_params = ["bae_type", "BTNECK_TYPE", "HAS_BTNECK", "num_layers"]
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
    ]
    optim_inf_df = apply_optim_df(
        inf_df,
        fixed_params=fixed_inf_params,
        optim_params=optim_inf_params,
        perf_key="E_AUROC",
        target_dim_col=tasks_col_name,
    )
    ##======OPTIMISE PARAMS=========
    # Combine INF and Non-Inf
    df_combined = optim_df.append(optim_inf_df)
    df_combined["taskname"] = dataset + "-" + df_combined[tasks_col_name].astype(str)
    optim_df_hasbtneck = apply_optim_df(
        df_combined,
        fixed_params=[compare_col],
        optim_params=[
            "bae_type",
        ],
        perf_key="E_" + metric_key_word,
        target_dim_col="taskname",
    )

    # append
    mega_df_list.append(optim_df_hasbtneck.copy())
# ==============OPTIM BAR PLOTS-20220501====================

mega_df_concat = pd.concat(mega_df_list)

# plot_mode = "bar"
plot_mode = "box"
task_labels_map = {
    "Images-CIFAR": "CIFAR vs SVHN",
    "Images-FashionMNIST": "FashionMNIST vs MNIST",
    "ODDS-cardio": "ODDS (Cardio)",
    "ODDS-lympho": "ODDS (Lympho)",
    "ODDS-optdigits": "ODDS (Optdigits)",
    "ODDS-pendigits": "ODDS (Pendigits)",
    "ODDS-thyroid": "ODDS (Thyroid)",
    "ODDS-ionosphere": "ODDS (Ionosphere)",
    "ODDS-pima": "ODDS (Pima)",
    "ODDS-vowels": "ODDS (Vowels)",
    "ZEMA-0": "ZeMA (Cooler)",
    "ZEMA-1": "ZeMA (Valve)",
    "ZEMA-2": "ZeMA (Pump)",
    "ZEMA-3": "ZeMA (Accumulator)",
    "STRATH-2": "STRATH (Radial forge)",
}
# bottleneck_map = {
#     "A": "Undercomplete, No skip (Bottlenecked)",
#     "B": "Overcomplete, No skip (Not bottlenecked)",
#     "C": "Undercomplete, + skip (Not bottlenecked)",
#     "D": "Overcomplete, + skip (Not bottlenecked)",
#     "INF": "Infinite-width (Not bottlenecked)",
# }

bottleneck_map = {
    "A": "Undercomplete \N{MINUS SIGN}skip (Bottlenecked)",
    "B": "Overcomplete \N{MINUS SIGN}skip (Not bottlenecked)",
    "C": "Undercomplete +skip (Not bottlenecked)",
    "D": "Overcomplete +skip (Not bottlenecked)",
    "INF": "Infinite-width (Not bottlenecked)",
}
rearrange_task_labels = list(task_labels_map.keys())
rearrange_btneck = list(bottleneck_map.keys())
mega_df_concat = rearrange_df(
    mega_df_concat, col="taskname", labels=rearrange_task_labels
)
mega_df_concat = rearrange_df(
    mega_df_concat, col="BTNECK_TYPE+INF", labels=rearrange_btneck
)
# linewidth = 2.35
linewidth = 1.7
flierprops = dict(markerfacecolor="0.75", markersize=2, linestyle="none")
showfliers = False
# figsize = (11.5, 10)
figsize = (11.5, 6)
fig, axes = plt.subplots(3, 1, figsize=figsize)
for row_i, ax in enumerate(axes):

    # filter tasknames by rows
    temp_df = mega_df_concat[
        mega_df_concat["taskname"].isin(
            rearrange_task_labels[row_i * 5 : (row_i + 1) * 5]
        )
    ]
    temp_df = replace_df_label_maps(
        temp_df, bottleneck_map, col="BTNECK_TYPE+INF"
    )  # cosmetic
    temp_df = replace_df_label_maps(
        temp_df, task_labels_map, col="taskname"
    )  # cosmetic
    if plot_mode == "bar":
        bplot = sns.barplot(
            x="taskname",
            hue=compare_col,
            y="E_" + metric_key_word,
            data=temp_df,
            capsize=0.1,
            ax=ax,
            errwidth=1.5,
            ci=95,
            palette="tab10",
        )
    elif plot_mode == "box":
        bplot = sns.boxplot(
            x="taskname",
            hue=compare_col,
            y="E_" + metric_key_word,
            data=temp_df,
            ax=ax,
            # palette="tab10",
            linewidth=linewidth,
            flierprops=flierprops,
            showfliers=showfliers,
        )
    ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.4)
    ax.set_xlabel("")
    ax.set_ylabel(metric_key_word)

    # legends
    if row_i == 0:
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
axes[-1].set_xlabel("Datasets")
fig.tight_layout()
if save_fig:
    fig.savefig("plots/" + "ALL-BTNECK-BOXPLOTS.png", dpi=600)

# =====================SEPARATE TO DIFFERENT AXES=============================
# linewidth = 2.35
# linewidth = 1.7
# flierprops = dict(markerfacecolor="0.75", markersize=2, linestyle="none")
# showfliers = False
# # figsize = (11.5, 10)
# figsize = (11.5, 6)
# fig, axes = plt.subplots(3, 5, figsize=figsize)
#
# for row_i, ax in enumerate(axes):
#
#     # filter tasknames by rows
#     temp_df = mega_df_concat[
#         mega_df_concat["taskname"].isin(
#             rearrange_task_labels[row_i * 5 : (row_i + 1) * 5]
#         )
#     ]
#     temp_df = replace_df_label_maps(
#         temp_df, bottleneck_map, col="BTNECK_TYPE+INF"
#     )  # cosmetic
#     temp_df = replace_df_label_maps(
#         temp_df, task_labels_map, col="taskname"
#     )  # cosmetic
#     if plot_mode == "bar":
#         bplot = sns.barplot(
#             x="taskname",
#             hue=compare_col,
#             y="E_" + metric_key_word,
#             data=temp_df,
#             capsize=0.1,
#             ax=ax,
#             errwidth=1.5,
#             ci=95,
#             palette="tab10",
#         )
#     elif plot_mode == "box":
#         bplot = sns.boxplot(
#             x="taskname",
#             hue=compare_col,
#             y="E_" + metric_key_word,
#             data=temp_df,
#             ax=ax,
#             # palette="tab10",
#             linewidth=linewidth,
#             flierprops=flierprops,
#             showfliers=showfliers,
#         )
#     ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.4)
#     ax.set_xlabel("")
#     ax.set_ylabel(metric_key_word)
#
#     # legends
#     if row_i == 0:
#         l = ax.legend(fontsize="small")
#         l.set_title("")
#     else:
#         ax.legend([], [], frameon=False)
#     # change border color
#     # https://stackoverflow.com/questions/55656683/change-seaborn-boxplot-line-rainbow-color
#     def adjust_lightness(color, amount):
#         try:
#             c = mc.cnames[color]
#         except:
#             c = color
#         c = colorsys.rgb_to_hls(*mc.to_rgb(c))
#         return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
#
#     for i, artist in enumerate(ax.artists):
#         # Set the linecolor on the artist to the facecolor, and set the facecolor to None
#         col = adjust_lightness(artist.get_facecolor(), 0.7)
#         artist.set_edgecolor(col)
#
#         for j in range(i * 5, i * 5 + 5):
#             line = ax.lines[j]
#             line.set_color(col)
#             line.set_mfc(col)
#             line.set_mec(col)
# axes[-1].set_xlabel("Datasets")
# fig.tight_layout()
