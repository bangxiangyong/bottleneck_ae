# This accompanying script plots the pickles from script 10-Analyse-BTNECK-SensitivityNoise.py
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

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

# =====Settings=====
savefig = True
# savefig = False

### Select dataset subsets to be plotted
dataset_subsets_i = 0  ## [0,1,2]
datasets_range = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)][
    dataset_subsets_i
]

# =========START PROCESSING ==============
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
dataset_labels = {
    key: val for key, val in np.array(list(dataset_labels.items()))[datasets_range]
}

# ======================UNFOLD-DETAILED TABLE====================================
noise_type = "normal"
final_res = {}
for dataset in ["CIFAR", "FashionMNIST", "ODDS", "ZEMA", "STRATH"]:
    datasets = [dataset_ for dataset_ in dataset_labels if dataset in dataset_]
    for noise_type in ["uniform", "normal"]:
        for i, dataset_targetdim in enumerate(datasets):
            inf_mean_all = []
            hasbtneck_mean_all = []
            nobtneck_mean_all = []
            traces = traces_dataset[dataset_targetdim]
            # ===========INF BAE===================
            df_inf_ = traces[noise_type]["bae-inf"]
            df_hasbtneck_ = traces[noise_type]["hasbtneck-ae"]
            df_nobtneck_ = traces[noise_type]["nobtneck-ae"]

            # append
            inf_mean_all.append(df_inf_["mean"][1:].values)
            hasbtneck_mean_all.append(df_hasbtneck_["mean"][1:].values)
            nobtneck_mean_all.append(df_nobtneck_["mean"][1:].values)

            print(dataset_targetdim)
            print(df_hasbtneck_["mean"][1:].values)
            print(df_nobtneck_["mean"][1:].values)
            print(np.mean(df_hasbtneck_["mean"][1:].values))
            print(np.mean(df_nobtneck_["mean"][1:].values))

            # ===== new ====
            inf_mean_all = np.array(inf_mean_all).flatten()
            hasbtneck_mean_all = np.array(hasbtneck_mean_all).flatten()
            nobtneck_mean_all = np.array(nobtneck_mean_all).flatten()

            inf_mean_all = np.exp(inf_mean_all)
            hasbtneck_mean_all = np.exp(hasbtneck_mean_all)
            nobtneck_mean_all = np.exp(nobtneck_mean_all)

            # ======MEAN & SEM DELTA AUROCS=====
            hasbtneck_mean_change = np.mean(hasbtneck_mean_all)
            hasbtneck_sem_change = sem(hasbtneck_mean_all)

            nobtneck_mean_change = np.mean(nobtneck_mean_all)
            nobtneck_sem_change = sem(nobtneck_mean_all)

            inf_mean_change = np.mean(inf_mean_all)
            inf_sem_change = sem(inf_mean_all)

            # ======SAVE TO DICT=====
            new_res = {
                noise_type: {
                    "bae-inf": {"mean": inf_mean_change, "sem": inf_sem_change},
                    "hasbtneck-ae": {
                        "mean": hasbtneck_mean_change,
                        "sem": hasbtneck_sem_change,
                    },
                    "nobtneck-ae": {
                        "mean": nobtneck_mean_change,
                        "sem": nobtneck_sem_change,
                    },
                }
            }

            if dataset_targetdim in final_res.keys():
                final_res[dataset_targetdim].update(new_res)
            else:
                final_res.update({dataset_targetdim: new_res})

# PLOTTING
res_vals_all = list(final_res.values())
res_keys_all = list(final_res.keys())

# figsize = (3, 2)
figsize = (3 * 5, 1.85)
yerr_params = {
    "capsize": 5,
    "ecolor": "black",
    "error_kw": {"markeredgewidth": 1},
    "zorder": 3,
}
width = 0.01
bar_dist = (width * 3) + width / 2

fig, axes = plt.subplots(1, 5, figsize=figsize)
axes = axes.flatten()
for i, ax in enumerate(axes):
    res_vals = res_vals_all[i]
    res_key = res_keys_all[i]
    # =====plot bars of robustness=========
    # fmt: off
    hasbtneck_mean_uniform = res_vals["uniform"]["hasbtneck-ae"]["mean"]
    nobtneck_mean_uniform = res_vals["uniform"]["nobtneck-ae"]["mean"]
    inf_mean_uniform = res_vals["uniform"]["bae-inf"]["mean"]
    hasbtneck_sem_uniform = res_vals["uniform"]["hasbtneck-ae"]["sem"]
    nobtneck_sem_uniform = res_vals["uniform"]["nobtneck-ae"]["sem"]
    inf_sem_uniform = res_vals["uniform"]["bae-inf"]["sem"]
    hasbtneck_mean_normal = res_vals["normal"]["hasbtneck-ae"]["mean"]
    nobtneck_mean_normal = res_vals["normal"]["nobtneck-ae"]["mean"]
    inf_mean_normal = res_vals["normal"]["bae-inf"]["mean"]
    hasbtneck_sem_normal = res_vals["normal"]["hasbtneck-ae"]["sem"]
    nobtneck_sem_normal = res_vals["normal"]["nobtneck-ae"]["sem"]
    inf_sem_normal = res_vals["normal"]["bae-inf"]["sem"]
    ax.bar(0.00,hasbtneck_mean_uniform,color="tab:blue",width=width,yerr=hasbtneck_sem_uniform,**yerr_params)
    ax.bar(width,nobtneck_mean_uniform,color="tab:orange",width=width,yerr=nobtneck_sem_uniform,**yerr_params)
    ax.bar(width*2,inf_mean_uniform,color="tab:purple",width=width,yerr=inf_sem_uniform,**yerr_params)
    ax.bar(0.00 + bar_dist,hasbtneck_mean_normal,color="tab:blue",width=width,yerr=hasbtneck_sem_normal,**yerr_params)
    ax.bar(width + bar_dist, nobtneck_mean_normal, color="tab:orange", width=width, yerr=nobtneck_sem_normal,**yerr_params)
    ax.bar(width*2 + bar_dist,inf_mean_normal,color="tab:purple",width=width,yerr=inf_sem_normal,**yerr_params)
    # fmt: on
    # =====================================
    # x label
    ax.set_xticks([width, width + bar_dist])
    ax.set_xticklabels([r"$\sigma^{+}$(Uniform)", r"$\sigma^{+}$(Gaussian)"])

    # y labels
    if i == 0:
        ax.set_ylabel("Robustness")
    else:
        ax.set_ylabel("")

    # ================SET Y LIMS AND TICKS===================
    # fmt: off
    y_low = np.min([hasbtneck_mean_uniform-hasbtneck_sem_uniform,nobtneck_mean_uniform-nobtneck_sem_uniform,inf_mean_uniform-inf_sem_uniform, hasbtneck_mean_normal-hasbtneck_sem_normal, nobtneck_mean_normal-nobtneck_sem_normal, inf_mean_normal-inf_sem_normal])
    y_high = np.max([hasbtneck_mean_uniform+hasbtneck_sem_uniform,nobtneck_mean_uniform+nobtneck_sem_uniform,inf_mean_uniform+inf_sem_uniform, hasbtneck_mean_normal+hasbtneck_sem_normal, nobtneck_mean_normal+nobtneck_sem_normal, inf_mean_normal+inf_sem_normal])
    # fmt: on

    # set y lims
    y_min = y_low - 0.25 * (y_high - y_low)
    y_max = y_high + 0.1 * (y_high - y_low)
    ax.set_ylim(y_min, y_max)

    # set y-ticks
    N = 5
    ymin, ymax = ax.get_ylim()
    new_ymin = (ymax - ymin) * 2 / 3
    ax.set_yticks(np.clip(np.round(np.linspace(ymin, ymax, N), 2), None, 1.0))

    # set grid
    ax.yaxis.grid(alpha=0.5, zorder=0)
    ax.set_title(dataset_labels[res_key], fontsize="small")
    # =======================================================

fig.tight_layout()

if savefig:
    filepath = "plots/all-robust-" + str(dataset_subsets_i) + ".png"
    fig.savefig(filepath, dpi=600)
    print("SAVE FIG:" + filepath)
