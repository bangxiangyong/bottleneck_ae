# This script produces the main boxplot results by combining all tables

import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    rearrange_df,
    replace_df_label_maps,
)

mega_df_list = []
metric_key_word = "AUROC"
round_deci = 3
save_fig = True

## define path to load the csv and mappings related to datasets
paths = {
    "ZEMA": "../results/zema-btneck-20220419",
    "STRATH": "../results/strath-btneck-20220419",
    "ODDS": "../results/odds-btneck-20220422",
    # "ODDS": "../results/odds-btneck-old-20220117",
    "Images": "../results/images-btneck-reboot-20220421",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
# compare_col = "HAS_BTNECK"
compare_col = "BTNECK_TYPE+INF"

# for dataset in ["STRATH", "ZEMA", "Images", "ODDS"]:
for dataset in ["STRATH", "Images", "ODDS", "ZEMA"]:
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
    path = paths[dataset]

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
    raw_df["BTNECK_TYPE+INF"] = np.select(
        conditions, btneck_labels
    )  # apply condition and label

    # whether the model is bottlenecked type or not
    raw_df["HAS_BTNECK"] = np.select(
        [raw_df["BTNECK_TYPE"] == "A", raw_df["BTNECK_TYPE"] != "A"], ["YES", "NO"]
    )

    # raw_df = raw_df[raw_df["layer_norm"] == "none"]
    # raw_df = raw_df[raw_df["current_epoch"] == 300]
    optim_df = apply_optim_df(
        raw_df,
        fixed_params=["bae_type", "BTNECK_TYPE"],
        optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
        perf_key="E_" + metric_key_word,
        target_dim_col=tasks_col_map[dataset],
    )

    # ================VIEW HYPER PARAMS================

    plot_mode = "bar"
    res_A = raw_df[raw_df["BTNECK_TYPE"] == "A"]
    res_B = raw_df[raw_df["BTNECK_TYPE"] == "B"]
    res_C = raw_df[raw_df["BTNECK_TYPE"] == "C"]
    res_D = raw_df[raw_df["BTNECK_TYPE"] == "D"]

    # fig, ax = plt.subplots(1, 1)
    # bplot = sns.barplot(
    #     x="latent_factor",
    #     hue="bae_type",
    #     y="E_" + metric_key_word,
    #     data=res_btneck_type,
    #     capsize=0.1,
    #     ax=ax,
    #     errwidth=1.5,
    #     ci=95,
    #     palette="tab10",
    # )

    # fig, axes = plt.subplots(1, 4, sharey=True)
    # # axes = axes.flatten()
    # # ax.plot(res_A["latent_factor"], res_A["E_" + metric_key_word])
    # for ax, res_btneck_type in zip(axes, [res_A, res_B, res_C, res_D]):
    #     if plot_mode == "bar":
    #         bplot = sns.barplot(
    #             # x="latent_factor",
    #             # hue="bae_type",
    #             x="bae_type",
    #             hue="latent_factor",
    #             y="E_" + metric_key_word,
    #             data=res_btneck_type,
    #             capsize=0.1,
    #             ax=ax,
    #             errwidth=1.5,
    #             ci=95,
    #             palette="tab10",
    #         )
    raw_df_ = raw_df[raw_df["layer_norm"] == "none"]
    raw_df_["bae_type+skip"] = raw_df_["bae_type"] + "+" + raw_df_["skip"].astype(str)
    fig, ax = plt.subplots(1, 1)
    bplot = sns.barplot(
        x="bae_type+skip",
        hue="latent_factor",
        y="E_" + metric_key_word,
        data=raw_df_,
        capsize=0.1,
        ax=ax,
        errwidth=1.5,
        ci=95,
        palette="tab10",
    )

    # ================INF BAE====================
    # handle inf
    inf_paths = {
        "ZEMA": "../results/inf-zema-20220501",
        "STRATH": "../results/inf-strath-20220501",
        "ODDS": "../results/inf-odds-20220501",
        "Images": "../results/inf-images-20220501",
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
    "ODDS-cardio": "Cardio",
    "ODDS-lympho": "Lympho",
    "ODDS-optdigits": "Optdigits",
    "ODDS-pendigits": "Pendigits",
    "ODDS-thyroid": "Thyroid",
    "ODDS-ionosphere": "Ionosphere",
    "ODDS-pima": "Pima",
    "ODDS-vowels": "Vowels",
    "ZEMA-0": "ZeMA (Cooler)",
    "ZEMA-1": "ZeMA (Valve)",
    "ZEMA-2": "ZeMA (Pump)",
    "ZEMA-3": "ZeMA (Accumulator)",
    "STRATH-2": "STRATH",
}
bottleneck_map = {
    "A": "Undercomplete, No skip (Bottlenecked)",
    "B": "Overcomplete, No skip (Not bottlenecked)",
    "C": "Undercomplete, + skip (Not bottlenecked)",
    "D": "Overcomplete, + skip (Not bottlenecked)",
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
linewidth = 2.35
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
axes[-1].set_xlabel("Tasks")
fig.tight_layout()
if save_fig:
    fig.savefig("plots/" + "ALL-BTNECK-FULL.png", dpi=500)
