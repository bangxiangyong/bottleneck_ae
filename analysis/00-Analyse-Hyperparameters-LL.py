# This establishes the baseline bottlenecked AE

import matplotlib.pyplot as plt
import pandas as pd
import hiplot
import numpy as np
import seaborn as sns

from thesis_experiments.util_analyse import (
    apply_optim_df,
    apply_optim_df_v2,
    concat_csv_files,
)

sns.set_theme()

dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"

results_folders = {
    "ZEMA": "../results/zema-hyp-ll-20220505",
    "STRATH": "../results/strath-hyp-ll-20220505",
    "ODDS": "../results/odds-hyp-ll-20220505",
    "Images": "../results/images-hyp-ll-20220505",
    # "ODDS": "../results/odds-hyp-btneck-temp-20220505",
    # "ZEMA": "../results/zema-hyp-btneck-20220506",
    # "STRATH": "../results/strath-hyp-btneck-20220506",
    # "Images": "../results/odds-hyp-btneck-temp-20220505",
}
results_folder = results_folders[dataset]
res_df = concat_csv_files(results_folder, key_word="AUROC.csv")
metric_key_word = "AUROC"

tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]
# ================================================================


selected_cols = ["latent_factor", "n_dense_layers", "full_likelihood", "activation"]

# selected_cols = ["latent_factor", "n_dense_layers", "skip", "activation"]

each_figs = tasks_col_name
aggregate = True
# aggregate = False

# optim_df = apply_optim_df(
#     res_df,
#     # fixed_params=selected_cols + [each_figs],
#     fixed_params=selected_cols,
#     optim_params=["current_epoch"],
#     perf_key="E_" + metric_key_word,
#     target_dim_col=tasks_col_name,
# )

optim_df = apply_optim_df(
    res_df,
    fixed_params=selected_cols + [each_figs],
    # fixed_params=selected_cols,
    optim_params=["current_epoch", "layer_norm"],
    perf_key="E_" + metric_key_word,
    target_dim_col=tasks_col_name,
)

if aggregate:
    optim_df[each_figs] = 100

# =======LL=============
cmap = "viridis"
for condition in optim_df[each_figs].unique():
    optim_df_ = optim_df[optim_df[each_figs] == condition]
    grouped_res_df = optim_df_.groupby(selected_cols).mean().reset_index()
    uniq_cols = grouped_res_df[selected_cols[-1]].unique()
    uniq_rows = grouped_res_df[selected_cols[-2]].unique()
    num_cols = len(uniq_cols)
    num_rows = len(uniq_rows)
    vmin, vmax = (
        grouped_res_df["E_" + metric_key_word].min(),
        grouped_res_df["E_" + metric_key_word].max(),
    )
    # annot = False
    annot = True
    share_cbar = True
    figsize = (num_rows * 3, num_cols * 3)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if len(axes.shape) == 1:
        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        else:
            axes = np.expand_dims(axes, axis=0)

    for row_i, row_name in enumerate(uniq_rows):
        for col_i, col_name in enumerate(uniq_cols):
            ax = axes[row_i][col_i]
            pivot_df_ = grouped_res_df[
                (grouped_res_df[selected_cols[-1]] == col_name)
                & (grouped_res_df[selected_cols[-2]] == row_name)
            ].pivot(selected_cols[0], selected_cols[1], "E_" + metric_key_word)
            # show_cbar = True if
            sns.heatmap(
                pivot_df_.round(3),
                ax=ax,
                vmin=vmin if share_cbar else None,
                vmax=vmax if share_cbar else None,
                annot=annot,
                fmt=".3f",
                cmap=cmap,
            )
            ax.set_title(col_name)
            ax.set_ylabel(row_name, rotation=0)
    fig.tight_layout()

# =====BTNECK==========


# =====HIPLOT=========

# Modified from JohanC:
# https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
#
# import matplotlib.pyplot as plt
# from matplotlib.path import Path
# import matplotlib.patches as patches
# import numpy as np
# from sklearn import datasets
#
# iris = datasets.load_iris()
# ynames = iris.feature_names
# ys = iris.data
#
# ynames = hyperparams + ["E_" + metric_key_word]
# ys = grouped_res_df[ynames].values
# target_names = grouped_res_df["target_dim"].unique()
#
# ymins = ys.min(axis=0)
# ymaxs = ys.max(axis=0)
# dys = ymaxs - ymins
# ymins -= dys * 0.05  # add 5% padding below and above
# ymaxs += dys * 0.05
#
# ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
# dys = ymaxs - ymins
#
# # transform all data to be compatible with the main axis
# zs = np.zeros_like(ys)
# zs[:, 0] = ys[:, 0]
# zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
#
# fig, host = plt.subplots(figsize=(10, 4))
#
# axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
# for i, ax in enumerate(axes):
#     ax.set_ylim(ymins[i], ymaxs[i])
#     ax.spines["top"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     if ax != host:
#         ax.spines["left"].set_visible(False)
#         ax.yaxis.set_ticks_position("right")
#         ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
#
# host.set_xlim(0, ys.shape[1] - 1)
# host.set_xticks(range(ys.shape[1]))
# host.set_xticklabels(ynames, fontsize=14)
# host.tick_params(axis="x", which="major", pad=7)
# host.spines["right"].set_visible(False)
# host.xaxis.tick_top()
# host.set_title("Parallel Coordinates Plot — Iris", fontsize=18, pad=12)
#
# colors = plt.cm.Set2.colors
# legend_handles = [None for _ in target_names]
# for j in range(len(grouped_res_df)):
#     # create bezier curves
#     verts = list(
#         zip(
#             [x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
#             np.repeat(zs[j, :], 3)[1:-1],
#         )
#     )
#     codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
#     path = Path(verts, codes)
#     patch = patches.PathPatch(
#         path,
#         facecolor="none",
#         lw=2,
#         alpha=0.7,
#         edgecolor=colors[grouped_res_df["target_dim"].values[j]],
#     )
#     legend_handles[grouped_res_df["target_dim"].values[j]] = patch
#     host.add_patch(patch)
# host.legend(
#     legend_handles,
#     target_names,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.18),
#     ncol=len(target_names),
#     fancybox=True,
#     shadow=True,
# )
# plt.tight_layout()
# plt.show()
