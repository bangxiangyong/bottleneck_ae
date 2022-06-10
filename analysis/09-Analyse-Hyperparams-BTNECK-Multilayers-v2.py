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
# save_fig = False
save_fig = True
# save_csv = False
include_inf = False
# include_inf = True

## define path to load the csv and mappings related to datasets
paths = {
    "ZEMA": "../results/zema-btneck-overhaul-repair-v2-20220517",
    # "STRATH": "../results/STRATH-BOTTLENECKV3",
    # "ODDS": "../results/odds-btneck-20220422",
    # "Images": "../results/images-btneck-reboot-20220421",
    # "ZEMA": "../results/zema-btneck-20220515",
    # "STRATH": "../results/strath-btneck-20220515",
    # "STRATH": "../results/strath-btneck-bxy20-20220519",
    # "STRATH": "../results/strath-btneck-overhaul-20220516",
    "STRATH": "../results/strath-btneck-incomp-v4",
    # "ODDS": "../results/odds-btneck-ovhaul-20220525",
    "ODDS": "../results/odds-btneck-246-20220527",
    # "Images": "../results/images-btneck-20220515",
    # "Images": "../results/cifar-btneck-overhaul-full-20220522",
    # "Images": "../results/fmnist-btneck-overhaul-v2-20220524",
    "Images": "../results/images-btneck-overhaul-20220525",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    # "STRATH": "ss_id",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
path = paths[dataset]
# bae_type_order = ["ae", "vae", "mcd", "vi", "ens"]
bae_type_order = ["ae", "ens"]
# bae_type_order = ["ens", "ae"]
all_pivots = []


## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word)
# raw_df = raw_df[raw_df["n_conv_layers"] == 1]
#
if dataset == "STRATH":
    raw_df = raw_df[raw_df["resample_factor"] == 50]

raw_df["num_layers"] = raw_df[
    "n_dense_layers" if dataset == "ODDS" else "n_conv_layers"
]
if dataset != "ODDS":
    raw_df["num_layers"] = raw_df["num_layers"] + 1
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

# ================INF BAE=================
# handle inf
inf_paths = {
    # "ZEMA": "../results/zema-inf-20220515",
    # "STRATH": "../results/strath-inf-20220515",
    # "ODDS": "../results/odds-inf-ovhaul-20220527",
    # "Images": "../results/images-inf-20220515",
    # ============INF TEST==================
    "ZEMA": "../results/inf-zema-test-noise-20220604",
    "STRATH": "../results/inf-strath-test-noise-20220604",
    "ODDS": "../results/inf-odds-test-noise-20220604",
    "Images": "../results/inf-images-test-noise-v3-20220605",
}
inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC.csv")
if "noise_scale" in inf_df.columns:
    inf_df = inf_df[(inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")]
inf_df["bae_type"] = "ae"
inf_df["BTNECK_TYPE"] = "B"
inf_df["BTNECK_TYPE+INF"] = "INF"
# inf_df["HAS_BTNECK"] = "INF"
inf_df["HAS_BTNECK"] = "NO"

inf_df["current_epoch"] = 1
inf_df["latent_factor"] = 1000
inf_df["layer_norm"] = inf_df["norm"]
inf_df["num_layers"] = (inf_df["num_layers"] - 1) // 2
# inf_df["n_dense_layers"] = (inf_df["num_layers"] - 1) // 2
# inf_df["n_conv_layers"] = (inf_df["num_layers"] - 1) // 2

# inf_df = inf_df[inf_df["num_layers"] == 2]  # limit num layers?
# inf_df["dataset"] = inf_df["dataset"] + ".mat"  # for odds?
fixed_inf_params = ["bae_type", "BTNECK_TYPE", "HAS_BTNECK", "num_layers"]
optim_inf_params = [
    "W_std",
    "diag_reg",
    "norm",
    # "num_layers",
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

# ==================PLOT HEATMAP=============================

##======OPTIMISE PARAMS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "BTNECK_TYPE"],
    # optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
    optim_params=["current_epoch", "layer_norm"],
    # optim_params=["layer_norm"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)
metric_col = "E_" + metric_key_word
display_order = [
    # "n_conv_layers" if dataset != "ODDS" else "n_dense_layers",
    "num_layers",
    "latent_factor",
]

# include BAE INF?
if include_inf:
    optim_df = optim_df.append(optim_inf_df)

# ==== BY TARGET DIM V2 ====
latent_factor_map = {
    0.1: r"$\times{\dfrac{1}{10}}$",
    0.5: r"$\times{\dfrac{1}{2}}$",
    1.0: r"$\times{1}$",
    2.0: r"$\times{2}$",
    10.0: r"$\times{10}$",
    1000: r"$\infty$",
}
annot = True


def rearrange_pivot(temp_df, display_order, metric_col, latent_factor_map):
    latent_factor_sorted = latent_factor_map.values()
    sorted_temp_df = (
        temp_df.pivot(*display_order, metric_col)
        .sort_index(level=0, ascending=False)
        .reindex(
            [
                col
                for col in list(latent_factor_sorted)
                if col in list(temp_df["latent_factor"])
            ],
            axis=1,
        )
    )
    return sorted_temp_df


for target_dim in optim_df[tasks_col_map[dataset]].unique():
    # for target_dim in [0]:
    for bae_type in bae_type_order[:-1]:
        optim_df_ = optim_df[
            (optim_df[tasks_col_map[dataset]] == target_dim)
            & (optim_df["bae_type"] == bae_type)
        ]
        for key, val in latent_factor_map.items():
            optim_df_["latent_factor"].replace(key, val, inplace=True)

        # type A
        temp_df_A = optim_df_[(optim_df_["BTNECK_TYPE"] == "A")]
        temp_df_B = optim_df_[(optim_df_["BTNECK_TYPE"] == "B")]
        temp_df_C = optim_df_[(optim_df_["BTNECK_TYPE"] == "C")]
        temp_df_D = optim_df_[(optim_df_["BTNECK_TYPE"] == "D")]

        temp_df_A = temp_df_A.groupby(display_order).mean()[metric_col].reset_index()
        temp_df_B = temp_df_B.groupby(display_order).mean()[metric_col].reset_index()
        temp_df_C = temp_df_C.groupby(display_order).mean()[metric_col].reset_index()
        temp_df_D = temp_df_D.groupby(display_order).mean()[metric_col].reset_index()

        temp_df_A = temp_df_A.round(round_deci)
        temp_df_B = temp_df_B.round(round_deci)
        temp_df_C = temp_df_C.round(round_deci)
        temp_df_D = temp_df_D.round(round_deci)

        # rearrange DF columns and rows:
        vmin = pd.concat((temp_df_A, temp_df_B, temp_df_C, temp_df_D))[metric_col].min()
        vmax = pd.concat((temp_df_A, temp_df_B, temp_df_C, temp_df_D))[metric_col].max()

        # rearrange DF columns and rows:
        temp_df_A = rearrange_pivot(
            temp_df_A,
            display_order=display_order,
            metric_col=metric_col,
            latent_factor_map=latent_factor_map,
        )
        temp_df_B = rearrange_pivot(
            temp_df_B,
            display_order=display_order,
            metric_col=metric_col,
            latent_factor_map=latent_factor_map,
        )
        temp_df_C = rearrange_pivot(
            temp_df_C,
            display_order=display_order,
            metric_col=metric_col,
            latent_factor_map=latent_factor_map,
        )
        temp_df_D = rearrange_pivot(
            temp_df_D,
            display_order=display_order,
            metric_col=metric_col,
            latent_factor_map=latent_factor_map,
        )
        cmap = "magma"
        linewidth = 0.5
        square = True
        width_ratios = [2, 3, 2, 3, 0.2]
        scaling_ratio = 0.85
        figsize = ((np.sum(width_ratios) - 1) * scaling_ratio, 3 * scaling_ratio)
        fig, (ax1, ax2, ax3, ax4, cbar_ax) = plt.subplots(
            ncols=len(width_ratios),
            figsize=figsize,
            gridspec_kw=dict(width_ratios=width_ratios),
        )
        sns.heatmap(
            temp_df_A,
            ax=ax1,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=".3f",
            cbar=False,
            linewidths=linewidth,
            square=square,
        )
        sns.heatmap(
            temp_df_B,
            ax=ax2,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=".3f",
            cbar=False,
            linewidths=linewidth,
            square=square,
        )
        sns.heatmap(
            temp_df_C,
            ax=ax3,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=".3f",
            cbar=False,
            linewidths=linewidth,
            square=square,
        )
        # cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
        hm4 = sns.heatmap(
            temp_df_D,
            ax=ax4,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=".3f",
            linewidths=linewidth,
            square=square,
            # cbar=False,
            cbar=True,
            cbar_ax=cbar_ax,
        )
        # fig.colorbar(ax1.collections[0], cax=cbar_ax)

        # ==== FIX LABELS ====
        for ax in [ax2, ax3, ax4]:  # empty ylabel and legends (2,3,4)
            ax.set_ylabel("")
        ax1.set_ylabel("Layers " + r"$L_{encoder}$")  # set first ylabel
        for ax in [ax2, ax3, ax4]:  # rename xlabel
            ax.set_ylabel("")
        for ax in [ax1, ax2, ax3, ax4]:  # rename xlabel
            ax.set_xlabel("Latent factor")
            ax.xaxis.set_label_coords(0.5, -0.3)

        title_size = "small"
        ax1.set_title("Undercomplete\nno skip", fontsize=title_size)
        ax2.set_title("Overcomplete\nNo skip", fontsize=title_size)
        ax3.set_title("Undercomplete\n+ skip", fontsize=title_size)
        ax4.set_title("Overcomplete\n+ skip", fontsize=title_size)

        # fig.tight_layout()
        fig.subplots_adjust(
            top=0.88, bottom=0.215, left=0.065, right=0.915, hspace=0.175, wspace=0.335
        )
        title_ = dataset + "-" + str(target_dim)
        if save_fig:
            fig.savefig(title_ + "-" + bae_type + "-heatmap.png", dpi=600)
        plt.get_current_fig_manager().set_window_title(title_)


# =================== AGGREGATED? ==============================

for bae_type in bae_type_order[:-1]:
    optim_df_ = optim_df[
        # (optim_df[tasks_col_map[dataset]] == target_dim)
        (optim_df["bae_type"] == bae_type)
    ]
    for key, val in latent_factor_map.items():
        optim_df_["latent_factor"].replace(key, val, inplace=True)

    # type A
    temp_df_A = optim_df_[(optim_df_["BTNECK_TYPE"] == "A")]
    temp_df_B = optim_df_[(optim_df_["BTNECK_TYPE"] == "B")]
    temp_df_C = optim_df_[(optim_df_["BTNECK_TYPE"] == "C")]
    temp_df_D = optim_df_[(optim_df_["BTNECK_TYPE"] == "D")]

    temp_df_A = temp_df_A.groupby(display_order).mean()[metric_col].reset_index()
    temp_df_B = temp_df_B.groupby(display_order).mean()[metric_col].reset_index()
    temp_df_C = temp_df_C.groupby(display_order).mean()[metric_col].reset_index()
    temp_df_D = temp_df_D.groupby(display_order).mean()[metric_col].reset_index()

    temp_df_A = temp_df_A.round(round_deci)
    temp_df_B = temp_df_B.round(round_deci)
    temp_df_C = temp_df_C.round(round_deci)
    temp_df_D = temp_df_D.round(round_deci)

    # rearrange DF columns and rows:
    vmin = pd.concat((temp_df_A, temp_df_B, temp_df_C, temp_df_D))[metric_col].min()
    vmax = pd.concat((temp_df_A, temp_df_B, temp_df_C, temp_df_D))[metric_col].max()

    # rearrange DF columns and rows:
    temp_df_A = rearrange_pivot(
        temp_df_A,
        display_order=display_order,
        metric_col=metric_col,
        latent_factor_map=latent_factor_map,
    )
    temp_df_B = rearrange_pivot(
        temp_df_B,
        display_order=display_order,
        metric_col=metric_col,
        latent_factor_map=latent_factor_map,
    )
    temp_df_C = rearrange_pivot(
        temp_df_C,
        display_order=display_order,
        metric_col=metric_col,
        latent_factor_map=latent_factor_map,
    )
    temp_df_D = rearrange_pivot(
        temp_df_D,
        display_order=display_order,
        metric_col=metric_col,
        latent_factor_map=latent_factor_map,
    )
    cmap = "magma"
    linewidth = 0.5
    square = True
    width_ratios = [2, 3, 2, 3, 0.2]
    scaling_ratio = 0.85
    figsize = ((np.sum(width_ratios) - 1) * scaling_ratio, 3 * scaling_ratio)
    fig, (ax1, ax2, ax3, ax4, cbar_ax) = plt.subplots(
        ncols=len(width_ratios),
        figsize=figsize,
        gridspec_kw=dict(width_ratios=width_ratios),
    )
    sns.heatmap(
        temp_df_A,
        ax=ax1,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=".3f",
        cbar=False,
        linewidths=linewidth,
        square=square,
    )
    sns.heatmap(
        temp_df_B,
        ax=ax2,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=".3f",
        cbar=False,
        linewidths=linewidth,
        square=square,
    )
    sns.heatmap(
        temp_df_C,
        ax=ax3,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=".3f",
        cbar=False,
        linewidths=linewidth,
        square=square,
    )
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    hm4 = sns.heatmap(
        temp_df_D,
        ax=ax4,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=".3f",
        linewidths=linewidth,
        square=square,
        # cbar=False,
        # cbar=True,
        cbar_ax=cbar_ax,
    )
    # fig.colorbar(ax1.collections[0], cax=cbar_ax)

    # ==== FIX LABELS ====
    for ax in [ax2, ax3, ax4]:  # empty ylabel and legends (2,3,4)
        ax.set_ylabel("")
    ax1.set_ylabel("Layers " + r"$L_{encoder}$")  # set first ylabel
    for ax in [ax2, ax3, ax4]:  # rename xlabel
        ax.set_ylabel("")
    for ax in [ax1, ax2, ax3, ax4]:  # rename xlabel
        ax.set_xlabel("Latent factor")
        ax.xaxis.set_label_coords(0.5, -0.3)

    title_size = "small"
    ax1.set_title("Undercomplete\nno skip", fontsize=title_size)
    ax2.set_title("Overcomplete\nNo skip", fontsize=title_size)
    ax3.set_title("Undercomplete\n+ skip", fontsize=title_size)
    ax4.set_title("Overcomplete\n+ skip", fontsize=title_size)
    # fig.tight_layout()
    # print(fig.get_size_inches())
    # fig.set_size_inches(*[7.66, 2.63], forward=True)

    fig.subplots_adjust(
        top=0.88, bottom=0.215, left=0.065, right=0.915, hspace=0.175, wspace=0.335
    )
    title_ = dataset + "-" + "AGG"
    bae_inf = "+inf" if include_inf else ""
    plt.get_current_fig_manager().set_window_title("-AGG-")
    if save_fig:
        fig.savefig(title_ + "-" + bae_type + bae_inf + "-heatmap.png", dpi=600)
    # plt.get_current_fig_manager().set_window_title(title_)


# ===============DEPRACATED CODE=====================================
# ====== V1 =======
# annot = True
# for bae_type in bae_type_order[:-1]:
#     for target_dim in optim_df[tasks_col_map[dataset]].unique():
#         optim_df_ = optim_df[
#             (optim_df[tasks_col_map[dataset]] == target_dim)
#             & (optim_df["bae_type"] == bae_type)
#         ]
#         # type A
#         temp_df_AB = optim_df_[
#             (optim_df_["BTNECK_TYPE"] == "A") | (optim_df_["BTNECK_TYPE"] == "B")
#         ]
#         temp_df_CD = optim_df_[
#             (optim_df_["BTNECK_TYPE"] == "C") | (optim_df_["BTNECK_TYPE"] == "D")
#         ]
#         temp_df_AB = temp_df_AB.groupby(display_order).mean()[metric_col].reset_index()
#         temp_df_CD = temp_df_CD.groupby(display_order).mean()[metric_col].reset_index()
#
#         vmin = pd.concat((temp_df_AB, temp_df_CD))[metric_col].min()
#         vmax = pd.concat((temp_df_AB, temp_df_CD))[metric_col].max()
#         cmap = "viridis"
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         sns.heatmap(
#             temp_df_AB.pivot(*display_order, "E_AUROC"),
#             ax=ax1,
#             vmin=vmin,
#             vmax=vmax,
#             cmap=cmap,
#             annot=annot,
#             fmt=".3f",
#         )
#         sns.heatmap(
#             temp_df_CD.pivot(*display_order, "E_AUROC"),
#             ax=ax2,
#             vmin=vmin,
#             vmax=vmax,
#             cmap=cmap,
#             annot=annot,
#             fmt=".3f",
#         )
#         plt.title(bae_type)


# ==========BEST BTNECK VS NONBTNECK==========
#
# print("===========================================")
# print("==========BEST BTNECK VS NON-BTNECK=========")
#
# groupby_cols = ["bae_type", "HAS_BTNECK"]
# optim_df_hasbtneck = apply_optim_df(
#     raw_df,
#     fixed_params=["bae_type", "HAS_BTNECK"],
#     optim_params=[
#         "current_epoch",
#         "latent_factor",
#         "skip",
#         "layer_norm",
#         "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
#         "resample_factor",
#     ],
#     perf_key="E_AUROC",
#     target_dim_col=tasks_col_map[dataset],
# )
#
# # Include Inf BAE?
# optim_df_hasbtneck = optim_df_hasbtneck.append(optim_inf_df)
#
# collect_pivot_dfs_btneck = []
# for target_dim in optim_df_hasbtneck[tasks_col_name].unique():
#     # for target_dim in [0]:
#     res_groupby = (
#         optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim]
#         .groupby(groupby_cols)
#         .mean()["E_" + metric_key_word]
#         .reset_index()
#     )
#
#     # convert to pivot table
#     pivot_df = res_groupby.pivot(
#         index="bae_type", columns="HAS_BTNECK", values="E_" + metric_key_word
#     ).reset_index()
#
#     # SUMMARY ROW: Mean and ATE of tasks
#     pivot_df_mean_ate = append_mean_ate_rows(
#         pivot_df, label_col="bae_type", baseline_col="YES"
#     )
#
#     # Collect to make final table
#     collect_pivot_dfs_btneck.append(pivot_df)
#
#     # print results
#     print("RESULT PER TASK:" + str(target_dim))
#     pprint(pivot_df_mean_ate.round(round_deci))
#
#     # ========PLOT BEST BTNECK========
#     fig, ax = plt.subplots(1, 1)
#     bplot = sns.barplot(
#         x="bae_type",
#         hue="HAS_BTNECK",
#         y="E_" + metric_key_word,
#         data=optim_df_hasbtneck[optim_df_hasbtneck[tasks_col_name] == target_dim],
#         capsize=0.1,
#         ax=ax,
#         errwidth=1.5,
#         ci=95,
#         hatch="///",
#         hue_order=["YES", "NO"],
#     )
#
#     fig.canvas.manager.set_window_title(str(target_dim))
#
# # Aggregated tables for all tasks
# collect_pivot_dfs_btneck = pd.concat(collect_pivot_dfs_btneck)
# collect_pivot_dfs_btneck = (
#     collect_pivot_dfs_btneck.groupby(["bae_type"]).mean().reset_index()
# )
# collect_pivot_dfs_btneck = append_mean_ate_rows(
#     collect_pivot_dfs_btneck, label_col="bae_type", baseline_col="YES"
# )
#
# # print results
# print("AGGREGATED RESULTS:")
# pprint(collect_pivot_dfs_btneck.round(round_deci))
