# This accompanying script plots the pickles from script 10-Analyse-BTNECK-SensitivityNoise.py

import pickle
import matplotlib.pyplot as plt
import numpy as np

pickle_names = [
    "Images-sens-traces.p",
    "ZEMA-sens-traces.p",
    "ODDS-sens-traces.p",
    "STRATH-sens-traces.p",
]

# combine traces from all datasets
traces_dataset = {}
for pickle_name in pickle_names:
    traces_dataset.update(pickle.load(open(pickle_name, "rb")))

# =====Seettings=====
display_legend = False
# display_legend = True
# save_fig = False
save_fig = True
# noise_type = "uniform"
noise_type = "normal"
dataset_subsets_i = 2  ## [0,1,2]

### Select dataset subsets to be plotted
datasets_range = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)][
    dataset_subsets_i
]
# =========PLOTS AND MAPS=============
colors = {
    "hasbtneck": "tab:blue",
    "nobtneck": "tab:orange",
    "inf": "tab:purple",
}
n_sems = 2
n_plots = 5
# figsize = (3.1 * n_plots, 2.5 * 1)
# figsize = (3.55 * n_plots, 3.8 * 1)  ## GOOD
figsize = (2.2 * n_plots, 1.9 * 1)

# figsize = (4.8 * n_plots, 4.5 * 3)
plt.rcParams.update({"font.size": 13})  ## GOOD
linewidth = 2
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
noise_level_factor = 1
y_scale = 10
datasets = np.array(list(dataset_labels.keys()))[datasets_range]
# datasets = np.array(list(dataset_labels.keys()))[:]
fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
axes = axes.flatten()
alpha = 0.1
line_alpha = 0.8
for i, dataset_targetdim in enumerate(datasets):
    ax = axes[i]
    # Scientific y-axis
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    # Set Title
    ax.set_title(dataset_labels[dataset_targetdim], fontsize="small")
    # ax.set_title(dataset_labels[dataset_targetdim], x=0.5, y=1.15, fontsize="small")
    ax.grid(axis="x")
    # Get traces for each dataset
    traces = traces_dataset[dataset_targetdim]
    # ===========INF BAE===================
    df_inf_ = traces[noise_type]["bae-inf"]
    df_hasbtneck_ = traces[noise_type]["hasbtneck-ae"]
    df_nobtneck_ = traces[noise_type]["nobtneck-ae"]

    (inf_line,) = ax.plot(
        df_inf_["noise_scale"] * noise_level_factor,
        np.clip(df_inf_["mean"], None, 0) * y_scale,
        color=colors["inf"],
        linewidth=linewidth,
        alpha=line_alpha,
    )

    ax.fill_between(
        df_inf_["noise_scale"] * noise_level_factor,
        np.clip(df_inf_["mean"] + n_sems * df_inf_["sem"], None, 0) * y_scale,
        np.clip(df_inf_["mean"] - n_sems * df_inf_["sem"], None, 0) * y_scale,
        color=colors["inf"],
        alpha=alpha,
    )
    # ==========FINITE: YES BTNECK BAE====================
    (hasbtneck_line,) = ax.plot(
        df_hasbtneck_["noise_scale"] * noise_level_factor,
        np.clip(df_hasbtneck_["mean"], None, 0) * y_scale,
        color=colors["hasbtneck"],
        linewidth=linewidth,
        alpha=line_alpha,
    )

    ax.fill_between(
        df_hasbtneck_["noise_scale"] * noise_level_factor,
        np.clip(df_hasbtneck_["mean"] + n_sems * df_hasbtneck_["sem"], None, 0)
        * y_scale,
        np.clip(df_hasbtneck_["mean"] - n_sems * df_hasbtneck_["sem"], None, 0)
        * y_scale,
        color=colors["hasbtneck"],
        alpha=alpha,
    )
    # ==========FINITE: NO BTNECK BAE====================
    (nobtneck_line,) = ax.plot(
        df_nobtneck_["noise_scale"] * noise_level_factor,
        np.clip(df_nobtneck_["mean"], None, 0) * y_scale,
        color=colors["nobtneck"],
        linewidth=linewidth,
        alpha=line_alpha,
    )

    ax.fill_between(
        df_nobtneck_["noise_scale"] * noise_level_factor,
        np.clip(df_nobtneck_["mean"] + n_sems * df_nobtneck_["sem"], None, 0) * y_scale,
        np.clip(df_nobtneck_["mean"] - n_sems * df_nobtneck_["sem"], None, 0) * y_scale,
        color=colors["nobtneck"],
        alpha=alpha,
    )

    # ============SET Y TICK=========
    all_means = (
        np.array(
            [
                df_hasbtneck_["mean"].values,
                df_nobtneck_["mean"].values,
                df_inf_["mean"].values,
            ]
        )
        * y_scale
    )
    min_, max_ = np.min(all_means), np.max(all_means)
    diff = (max_ - min_) / 20
    ax.set_ylim(
        [
            min_ - diff,
            # -0.403,
            max_ + diff,
        ]
    )
    print((min_ - diff).round(1))
    lim = ax.get_ylim()
    # ax.set_yticks(list(ax.get_yticks()))
    ax.set_ylim(lim)

    # YTICKS AT 5 LEVELS
    N = 3
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.clip(np.round(np.linspace(ymin, ymax, N), 1), None, 0))

    # ======LABELS=========
    noise_label_map = {
        "normal": r"$\sigma^{+\mathrm{Gauss}}$",
        "uniform": r"$\sigma^{+\mathrm{Uni}}$",
    }
    if "normal" in dataset_targetdim:
        noise_type = "normal"
    elif "uniform" in dataset_targetdim:
        noise_type = "uniform"

    xlabel = noise_label_map[noise_type] + r"($\times10^{-1}$)"
    ax.set_xlabel(xlabel, fontsize="small")
    ax.set_ylabel(r"$\Delta$" + "AUROC" + r"($\times10^{-1}$)")

    # xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # xticklabels = [0, "2", "4", "6", "8", "10"]
    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    xticklabels = [0, "1", "2", "3", "4", "5"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    fig.canvas.manager.set_window_title(dataset_targetdim)

for i, ax in enumerate(axes):
    if i not in [0, 5, 10]:
        ax.set_ylabel("")
fig.tight_layout(pad=0.15)
# fig.tight_layout()

if display_legend:
    # Shrink current axis's height by 10% on the bottom
    ax = axes[2]
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        [hasbtneck_line, nobtneck_line, inf_line],
        [
            "Det. AE (Bottlenecked)",
            "Det. AE (No bottleneck)",
            "BAE-" + r"$\infty$" + "(No bottleneck)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.53),
        fancybox=True,
        ncol=5,
        fontsize="small",
    )
# =======save figure======
if save_fig:
    figname = "plots/ALL-" + str(dataset_subsets_i) + noise_type + "-noise.png"
    fig.savefig(figname, dpi=600)
    print("SAVE FIG:" + figname)

# ============================COMBINE BOTH UNIFORM AND GAUSSIAN PLOTS=======================================
#
# noise_level_factor = 2
# datasets = np.array(list(dataset_labels.keys()))[datasets_range]
# # datasets = np.array(list(dataset_labels.keys()))[:]
# fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# # fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
# axes = axes.flatten()
# alpha = 0.1
# line_alpha = 0.8
#
# for i, dataset_targetdim in enumerate(datasets):
#     for noise_type in ["uniform", "normal"]:
#         ax = axes[i]
#         # Scientific y-axis
#         # ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, -4), useMathText=True)
#         # ax.ticklabel_format(axis="x", style="sci", scilimits=(4, -4), useMathText=True)
#         ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
#         # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
#         # ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=True)
#         # Set Title
#         ax.set_title(dataset_labels[dataset_targetdim], x=0.5, y=1.12, fontsize="small")
#         ax.grid(axis="x")
#         # Get traces for each dataset
#         traces = traces_dataset[dataset_targetdim]
#         # ===========INF BAE===================
#         df_inf_ = traces[noise_type]["bae-inf"]
#         df_hasbtneck_ = traces[noise_type]["hasbtneck-ae"]
#         df_nobtneck_ = traces[noise_type]["nobtneck-ae"]
#
#         (inf_line,) = ax.plot(
#             df_inf_["noise_scale"] * noise_level_factor,
#             df_inf_["mean"],
#             color=colors["inf"],
#             linewidth=linewidth,
#             alpha=line_alpha,
#         )
#
#         ax.fill_between(
#             df_inf_["noise_scale"] * noise_level_factor,
#             df_inf_["mean"] + n_sems * df_inf_["sem"],
#             df_inf_["mean"] - n_sems * df_inf_["sem"],
#             color=colors["inf"],
#             alpha=alpha,
#         )
#         # ==========FINITE: YES BTNECK BAE====================
#         (hasbtneck_line,) = ax.plot(
#             df_hasbtneck_["noise_scale"] * noise_level_factor,
#             df_hasbtneck_["mean"],
#             color=colors["hasbtneck"],
#             linewidth=linewidth,
#             alpha=line_alpha,
#         )
#
#         ax.fill_between(
#             df_hasbtneck_["noise_scale"] * noise_level_factor,
#             df_hasbtneck_["mean"] + n_sems * df_hasbtneck_["sem"],
#             df_hasbtneck_["mean"] - n_sems * df_hasbtneck_["sem"],
#             color=colors["hasbtneck"],
#             alpha=alpha,
#         )
#         # ==========FINITE: NO BTNECK BAE====================
#         (nobtneck_line,) = ax.plot(
#             df_nobtneck_["noise_scale"] * noise_level_factor,
#             df_nobtneck_["mean"],
#             color=colors["nobtneck"],
#             linewidth=linewidth,
#             alpha=line_alpha,
#         )
#
#         ax.fill_between(
#             df_nobtneck_["noise_scale"] * noise_level_factor,
#             df_nobtneck_["mean"] + n_sems * df_nobtneck_["sem"],
#             df_nobtneck_["mean"] - n_sems * df_nobtneck_["sem"],
#             color=colors["nobtneck"],
#             alpha=alpha,
#         )
#
#         all_means = [df_hasbtneck_["mean"], df_nobtneck_["mean"], df_inf_["mean"]]
#         min_, max_ = np.min(all_means), np.max(all_means)
#         diff = (max_ - min_) / 20
#         ax.set_ylim(
#             [
#                 min_ - diff,
#                 # -0.403,
#                 max_ + diff,
#             ]
#         )
#         print((min_ - diff).round(1))
#         lim = ax.get_ylim()
#         ax.set_yticks(list(ax.get_yticks()))
#         ax.set_ylim(lim)
#         # ======LABELS=========
#         noise_label_map = {
#             "normal": r"$2\sigma^{+Gaussian}$",
#             "uniform": r"$\sigma^{+Uniform}$",
#         }
#         if "normal" in dataset_targetdim:
#             noise_type = "normal"
#         elif "uniform" in dataset_targetdim:
#             noise_type = "uniform"
#
#         xlabel = noise_label_map[noise_type] + r"($\times10^{-1}$)"
#         # ax.set_xlabel("Noise level\n" + xlabel)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(r"$\Delta$" + "AUROC")
#
#         # xticks = (df_inf_["noise_scale"] * noise_level_factor).to_list()
#         # xticks = [0, 0.5, 1.0]
#         # xticklabels = [0, 0.5, 1]
#         xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
#         xticklabels = [0, "2", "4", "6", "8", "10"]
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)
#         fig.canvas.manager.set_window_title(dataset_targetdim)
#
# for i, ax in enumerate(axes):
#     if i not in [0, 5, 10]:
#         ax.set_ylabel("")
# fig.tight_layout()
#
# if display_legend:
#     # Shrink current axis's height by 10% on the bottom
#     ax = axes[2]
#     box = ax.get_position()
#     # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
#
#     # Put a legend below current axis
#     ax.legend(
#         [hasbtneck_line, nobtneck_line, inf_line],
#         [
#             "Det. AE (Bottlenecked)",
#             "Det. AE (No bottleneck)",
#             "BAE-" + r"$\infty$" + "(No bottleneck)",
#         ],
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.53),
#         fancybox=True,
#         ncol=5,
#     )
#
#     axes[0].legend(
#         [hasbtneck_line, nobtneck_line, inf_line],
#         [
#             "Det. AE (Bottlenecked)",
#             "Det. AE (No bottleneck)",
#             "BAE-" + r"$\infty$" + "(No bottleneck)",
#         ],
#         fontsize="xx-small",
#         loc="lower center",
#     )
# fig.subplots_adjust(
#     top=0.735, bottom=0.206, left=0.061, right=0.981, hspace=0.2, wspace=0.554
# )


# =================BACKUP=============================
# for i, dataset_targetdim in enumerate(datasets):
#     ax = axes[i]
#     # Scientific y-axis
#     # ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, -4), useMathText=True)
#     # ax.ticklabel_format(axis="x", style="sci", scilimits=(4, -4), useMathText=True)
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
#     # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
#     # ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=True)
#     # Set Title
#     ax.set_title(dataset_labels[dataset_targetdim], x=0.5, y=1.12, fontsize="small")
#     ax.grid(axis="x")
#     # Get traces for each dataset
#     traces = traces_dataset[dataset_targetdim]
#     # ===========INF BAE===================
#     df_inf_ = traces[noise_type]["bae-inf"]
#     df_hasbtneck_ = traces[noise_type]["hasbtneck-ae"]
#     df_nobtneck_ = traces[noise_type]["nobtneck-ae"]
#
#     (inf_line,) = ax.plot(
#         df_inf_["noise_scale"] * noise_level_factor,
#         df_inf_["mean"],
#         color=colors["inf"],
#         linewidth=linewidth,
#         alpha=line_alpha,
#     )
#
#     ax.fill_between(
#         df_inf_["noise_scale"] * noise_level_factor,
#         df_inf_["mean"] + n_sems * df_inf_["sem"],
#         df_inf_["mean"] - n_sems * df_inf_["sem"],
#         color=colors["inf"],
#         alpha=alpha,
#     )
#     # ==========FINITE: YES BTNECK BAE====================
#     (hasbtneck_line,) = ax.plot(
#         df_hasbtneck_["noise_scale"] * noise_level_factor,
#         df_hasbtneck_["mean"],
#         color=colors["hasbtneck"],
#         linewidth=linewidth,
#         alpha=line_alpha,
#     )
#
#     ax.fill_between(
#         df_hasbtneck_["noise_scale"] * noise_level_factor,
#         df_hasbtneck_["mean"] + n_sems * df_hasbtneck_["sem"],
#         df_hasbtneck_["mean"] - n_sems * df_hasbtneck_["sem"],
#         color=colors["hasbtneck"],
#         alpha=alpha,
#     )
#     # ==========FINITE: NO BTNECK BAE====================
#     (nobtneck_line,) = ax.plot(
#         df_nobtneck_["noise_scale"] * noise_level_factor,
#         df_nobtneck_["mean"],
#         color=colors["nobtneck"],
#         linewidth=linewidth,
#         alpha=line_alpha,
#     )
#
#     ax.fill_between(
#         df_nobtneck_["noise_scale"] * noise_level_factor,
#         df_nobtneck_["mean"] + n_sems * df_nobtneck_["sem"],
#         df_nobtneck_["mean"] - n_sems * df_nobtneck_["sem"],
#         color=colors["nobtneck"],
#         alpha=alpha,
#     )
#
#     all_means = [df_hasbtneck_["mean"], df_nobtneck_["mean"], df_inf_["mean"]]
#     min_, max_ = np.min(all_means), np.max(all_means)
#     diff = (max_ - min_) / 20
#     ax.set_ylim(
#         [
#             min_ - diff,
#             # -0.403,
#             max_ + diff,
#         ]
#     )
#     print((min_ - diff).round(1))
#     lim = ax.get_ylim()
#     ax.set_yticks(list(ax.get_yticks()))
#     ax.set_ylim(lim)
#     # ======LABELS=========
#     noise_label_map = {
#         "normal": r"$2\sigma^{+Gaussian}$",
#         "uniform": r"$\sigma^{+Uniform}$",
#     }
#     if "normal" in dataset_targetdim:
#         noise_type = "normal"
#     elif "uniform" in dataset_targetdim:
#         noise_type = "uniform"
#
#     xlabel = noise_label_map[noise_type] + r"($\times10^{-1}$)"
#     # ax.set_xlabel("Noise level\n" + xlabel)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(r"$\Delta$" + "AUROC")
#
#     # xticks = (df_inf_["noise_scale"] * noise_level_factor).to_list()
#     # xticks = [0, 0.5, 1.0]
#     # xticklabels = [0, 0.5, 1]
#     xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
#     xticklabels = [0, "2", "4", "6", "8", "10"]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels)
#     fig.canvas.manager.set_window_title(dataset_targetdim)
