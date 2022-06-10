from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FormatStrFormatter
from numpy import trapz
from sklearn.metrics import auc

from thesis_experiments.util_analyse import (
    concat_csv_files,
    apply_optim_df,
    append_mean_ate_rows,
    rearrange_df,
    replace_df_label_maps,
    get_mean_sem,
)
import pickle

# dataset = "STRATH"
dataset = "ZEMA"
# dataset = "Images"
# dataset = "ODDS"

metric_key_word = "AUROC"
save_fig = False
# save_fig = True
on_marker = True
# display_legend = True
display_legend = False


## define path to load the csv and mappings related to datasets
paths = {
    # "ZEMA": "../results/zema-noise-test-20220527",
    "ZEMA": "../results/zema-noise-20220526",
    # "STRATH": "../results/STRATH-BOTTLENECKV3",
    # "ODDS": "../results/odds-btneck-20220422",
    # "Images": "../results/images-btneck-reboot-20220421",
    # "ZEMA": "../results/zema-btneck-20220515",
    # "STRATH": "../results/strath-btneck-20220515",
    # "STRATH": "../results/strath-btneck-bxy20-20220519",
    # "STRATH": "../results/strath-btneck-overhaul-20220516",
    "STRATH": "../results/strath-noise-20220526",
    # "STRATH": "../results/ae-strath-test-noise-20220604",
    # "ODDS": "../results/odds-noise-full-20220525",
    # "ODDS": "../results/odds-noise-246-20220527",
    "ODDS": "../results/odds-noise-246-20220527",
    # "ODDS": "../results/odds-ae+bae-noise-20220529",
    # "ODDS": "../results/odds-sae+ae-noise-20220603",
    # "Images": "../results/images-btneck-20220515",
    # "Images": "../results/cifar-btneck-overhaul-full-20220522",
    # "Images": "../results/fmnist-btneck-overhaul-v2-20220524",
    # "Images": "../results/images-noise-20220526",
    "Images": "../results/images-noise-v3-20220527",
    # ==========TEST NOISE==============
    "ZEMA": "../results/zema-wdecay-fixed-20220604",
    "STRATH": "../results/strath-wdecay-20220604",
    "ODDS": "../results/odds-test-noise-20220604",
    "Images": "../results/images-test-noise-20220604",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}

dataset_labels = {
    "ZEMA-0": "ZeMA(Cooler)",
    "ZEMA-1": "ZeMA(Valve)",
    "ZEMA-2": "ZeMA(Pump)",
    "ZEMA-3": "ZeMA(Accumulator)",
    "STRATH-2": "STRATH(Radial Forge)",
}

tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
path = paths[dataset]
all_pivots = []


## start reading csv
# raw_df = concat_csv_files(path, key_word=metric_key_word)
raw_df = concat_csv_files(path, key_word="NOISE_TEST.csv")

if dataset == "STRATH":
    raw_df = raw_df[raw_df["resample_factor"] == 50]

## ========================================
def get_delta_aurocs(temp_df):
    # AUROC and DELTA AUROC
    aurocs = temp_df[temp_df["noise_scale"] == 0].sort_values("weight_decay")["E_AUROC"]
    e_delta_aurocs = []
    weight_decays = np.sort(temp_df["weight_decay"].unique())
    for weight_decay in weight_decays:
        wdecay_df = temp_df[temp_df["weight_decay"] == weight_decay].sort_values(
            "noise_scale"
        )
        aurocs_ = wdecay_df["E_AUROC"].values
        delta_aurocs = aurocs_ - aurocs_[0]
        e_delta_aurocs.append(np.mean(delta_aurocs))
    e_delta_aurocs = np.array(e_delta_aurocs)
    return aurocs, e_delta_aurocs, weight_decays


for target_dim in np.sort(raw_df[tasks_col_name].unique()):
    temp_df = raw_df[raw_df[tasks_col_name] == target_dim]
    print(temp_df)
    for noise_type in ["uniform", "normal"]:
        # SAE
        temp_df_sae = (
            temp_df[
                (temp_df["bae_type"] == "sae") & (temp_df["noise_type"] == noise_type)
            ]
            .groupby(["noise_scale", "noise_type", "weight_decay"])
            .mean()
            .reset_index()
        )

        # SAE
        temp_df_ae = (
            temp_df[
                (temp_df["bae_type"] == "ae") & (temp_df["noise_type"] == noise_type)
            ]
            .groupby(["noise_scale", "noise_type", "weight_decay"])
            .mean()
            .reset_index()
        )

        # AUROC and DELTA AUROC
        aurocs_sae, e_delta_aurocs_sae, weight_decays = get_delta_aurocs(temp_df_sae)
        aurocs_ae, e_delta_aurocs_ae, _ = get_delta_aurocs(temp_df_ae)

        e_delta_aurocs_sae = np.clip(e_delta_aurocs_sae, None, 0)
        e_delta_aurocs_ae = np.clip(e_delta_aurocs_ae, None, 0)
        print(e_delta_aurocs_sae[-1])

        # apply exponent?
        e_delta_aurocs_sae = np.exp(e_delta_aurocs_sae)
        e_delta_aurocs_ae = np.exp(e_delta_aurocs_ae)

        # linestyles
        ae_style = "-"
        sae_style = "--"
        robustness_color = "tab:blue"
        accuracy_color = "tab:green"

        if on_marker:
            ae_marker = "o"
            sae_marker = "^"
        else:
            ae_marker = None
            sae_marker = None

        # figsize = (5.5, 4)
        # plt.rcParams.update({"font.size": 16})

        # figsize = (5.5, 3)
        # plt.rcParams.update({"font.size": 20})

        figsize = (5.5, 4)
        plt.rcParams.update({"font.size": 20})

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.locator_params(axis="y", nbins=5)
        ax_twin = ax.twinx()
        (line_acc_ae,) = ax.plot(
            aurocs_ae,
            color=accuracy_color,
            linestyle=ae_style,
            marker=ae_marker,
            markersize=4.5,
        )
        (line_acc_sae,) = ax.plot(
            aurocs_sae, color=accuracy_color, linestyle=sae_style, marker=sae_marker
        )
        (line_rob_ae,) = ax_twin.plot(
            e_delta_aurocs_ae,
            color=robustness_color,
            linestyle=ae_style,
            marker=ae_marker,
            markersize=4.5,
        )
        (line_rob_sae,) = ax_twin.plot(
            e_delta_aurocs_sae,
            color=robustness_color,
            linestyle=sae_style,
            marker=sae_marker,
        )

        ## ===label texts===
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_twin.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # ax.set_ylabel("Accuracy (AUROC)", color=accuracy_color)
        # ax_twin.set_ylabel(
        #     "Robustness " + r"$(\mathbb{E}{}_{\sigma^{+}}[\Delta{\mathrm{AUROC}}])$",
        #     color=robustness_color,
        # )
        ax.set_ylabel("Accuracy", color=accuracy_color)
        ax_twin.set_ylabel("Robustness", color=robustness_color)
        ax.set_xlabel("Weight decay " + r"$\lambda$")
        ax.set_xticks(np.arange(len(weight_decays)))

        xlabels = ["0", r"$10^{-10}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"]
        ax.set_xticklabels(xlabels, fontsize="small")
        # ax_twin.yaxis.set_label_coords(1.25, 0.40)

        # SET NUM TICKS
        # N = 5
        # ymin, ymax = ax_twin.get_ylim()
        # ytwin_ticks = np.clip(np.round(np.linspace(ymin, ymax, N), 2), None, 0)
        # ax_twin.set_yticks(ytwin_ticks)
        # N = 5
        # ymin, ymax = ax.get_ylim()
        # ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))

        ## ===color labels===
        ax.spines["left"].set_color(accuracy_color)
        ax.yaxis.label.set_color(accuracy_color)
        ax.tick_params(axis="y", colors=accuracy_color)

        ax_twin.spines["right"].set_color(robustness_color)
        ax_twin.yaxis.label.set_color(robustness_color)
        ax_twin.tick_params(axis="y", colors=robustness_color)

        # ===legend====
        if display_legend:
            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(
                [(line_acc_sae, line_rob_sae), (line_acc_ae, line_rob_ae)],
                [
                    r"$L_1$ regularisation (Laplace prior)",
                    r"$L_2$ regularisation (Gaussian prior)",
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.53),
                fancybox=True,
                ncol=5,
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                handlelength=3,
            )

            # plt.legend(
            #     [(line_acc_sae, line_rob_sae), (line_acc_ae, line_rob_ae)],
            #     [
            #         r"$L_1$ regularisation (Laplace prior)",
            #         r"$L_2$ regularisation (Gaussian prior)",
            #     ],
            #     numpoints=1,
            #     handler_map={tuple: HandlerTuple(ndivide=None)},
            #     handlelength=3,
            # )
        ax.set_title(dataset_labels[dataset + "-" + str(target_dim)], fontsize="small")
        fig.tight_layout(pad=0)

        if save_fig:
            filename = (
                "plots/"
                + dataset
                + "-"
                + str(target_dim)
                + "-"
                + str(noise_type)
                + "-wdecay.png"
            )
            fig.savefig(
                filename,
                dpi=600,
            )
            print("SAVED PNG:" + str(filename))

## ============================================
