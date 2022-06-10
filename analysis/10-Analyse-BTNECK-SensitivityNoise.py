from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
round_deci = 3
save_fig = False
# save_fig = True
save_csv = False
# save_pickle = True
save_pickle = False

display_legend = True
# display_legend = False


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
    "ZEMA": "../results/zema-test-noise-20220604",
    "STRATH": "../results/strath-test-noise-20220604",
    "ODDS": "../results/odds-test-noise-20220604",
    "Images": "../results/images-test-noise-20220604",
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
# raw_df = concat_csv_files(path, key_word=metric_key_word)
raw_df = concat_csv_files(path, key_word="NOISE")


if dataset == "STRATH":
    raw_df = raw_df[raw_df["resample_factor"] == 50]

# if dataset == "Images":
#     # raw_df = raw_df[raw_df["current_epoch"] == 20]
#     raw_df = raw_df[
#         # (raw_df["noise_scale"] != 0.1)
#         # & (raw_df["noise_scale"] != 0.2)
#         (raw_df["noise_scale"] <= 0.5)
#     ]

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

isBAE = raw_df["bae_type"] != "ae"
isnotBAE = raw_df["bae_type"] == "ae"
raw_df["isBAE"] = np.select([isBAE, isnotBAE], ["bae", "ae"])

# raw_df = raw_df[raw_df["weight_decay"] == 1e-10]
##======OPTIMISE PARAMS=========
# raw_df = raw_df[raw_df["current_epoch"] == raw_df["current_epoch"].max()]
# optim_df = apply_optim_df(
#     raw_df,
#     fixed_params=["bae_type", "BTNECK_TYPE"],
#     optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
#     # optim_params=["latent_factor", "skip"],
#     perf_key="E_AUROC",
#     target_dim_col=tasks_col_map[dataset],
# )

# ================INF BAE=================
## handle inf
inf_paths = {
    "ZEMA": "../results/zema-inf-noise-revamp-20220528",
    # "STRATH": "../results/strath-inf-noise-revamp-20220528",
    "ODDS": "../results/odds-inf-noise-repair-20220527",
    "Images": "../results/images-inf-noise-20220527",
    # ============INF TEST==================
    "ZEMA": "../results/inf-zema-test-noise-20220604",
    "STRATH": "../results/inf-strath-test-noise-20220604",
    "ODDS": "../results/inf-odds-test-noise-20220604",
    "Images": "../results/inf-images-test-noise-v3-20220605"
    # "STRATH": "../results/inf-strath-test-noise-20220604-optim",
}
# inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC")
inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="NOISE")

inf_df = inf_df[~inf_df.iloc[:, 1:].duplicated()]  # drop duplicated
inf_df["bae_type"] = "bae_inf"
inf_df["BTNECK_TYPE"] = "B"
inf_df["BTNECK_TYPE+INF"] = "INF"
inf_df["HAS_BTNECK"] = "INF"
# inf_df["HAS_BTNECK"] = "NO"

inf_df["current_epoch"] = 1
inf_df["latent_factor"] = 1
inf_df["layer_norm"] = inf_df["norm"]

# inf_df = inf_df[inf_df["num_layers"] == 2]  # limit num layers?
# inf_df["dataset"] = inf_df["dataset"] + ".mat"  # for odds?
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
    fixed_params=["noise_type"],
    optim_params=["num_layers", "layer_norm", "W_std"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_name,
)

##======OPTIMISE PARAMS=========
optim_df = apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "HAS_BTNECK", "noise_type", "noise_scale"],
    # fixed_params=["bae_type", "HAS_BTNECK", "noise_scale"],
    # optim_params=["current_epoch", "latent_factor", "skip", "layer_norm"],
    optim_params=["current_epoch", "layer_norm"],
    perf_key="E_AUROC",
    target_dim_col=tasks_col_map[dataset],
)
metric_col = "E_" + metric_key_word
display_order = [
    "n_conv_layers" if dataset != "ODDS" else "n_dense_layers",
    "latent_factor",
]
# ===========plot inf ===========
def calc_srm(df_inf_mean_):
    """
    Standardised response mean (SRM)
    """
    mean_change = np.nanmean(df_inf_mean_["E_AUROC"][1:])
    mean_change = np.exp(mean_change)
    return mean_change


colors = {
    "hasbtneck": "tab:blue",
    "nobtneck": "tab:orange",
    "inf": "tab:purple",
}
temp_inf_df = optim_inf_df
n_sems = 2
noise_level_factor = 2
plt.rcParams.update({"font.size": 17})
linewidth = 2.5
figsize = (4.1, 3.5)
final_dict_res = {}  # to save and accumulate traces
for target_dim in optim_inf_df[tasks_col_name].unique():
    for noise_type in ["normal", "uniform"]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # ===========INF BAE===================
        df_inf_mean_, df_inf_sem_ = get_mean_sem(
            temp_inf_df,
            {tasks_col_name: target_dim, "noise_type": noise_type},
            ["noise_type", "noise_scale"],
        )

        (inf_line,) = ax.plot(
            df_inf_mean_["noise_scale"] * noise_level_factor,
            df_inf_mean_["E_AUROC"],
            color=colors["inf"],
            linewidth=linewidth,
        )

        ax.fill_between(
            df_inf_mean_["noise_scale"] * noise_level_factor,
            df_inf_mean_["E_AUROC"] + n_sems * df_inf_sem_["E_AUROC"],
            df_inf_mean_["E_AUROC"] - n_sems * df_inf_sem_["E_AUROC"],
            color=colors["inf"],
            alpha=0.1,
        )
        # ==========FINITE: YES BTNECK BAE====================
        df_hasbtneck_mean_, df_hasbtneck_sem_ = get_mean_sem(
            optim_df,
            {
                tasks_col_name: target_dim,
                "noise_type": noise_type,
                "HAS_BTNECK": "YES",
                "bae_type": "ae",
            },
            ["noise_type", "noise_scale"],
        )
        (hasbtneck_line,) = ax.plot(
            df_hasbtneck_mean_["noise_scale"] * noise_level_factor,
            df_hasbtneck_mean_["E_AUROC"],
            color=colors["hasbtneck"],
            linewidth=linewidth,
        )
        ax.fill_between(
            df_hasbtneck_mean_["noise_scale"] * noise_level_factor,
            df_hasbtneck_mean_["E_AUROC"] + n_sems * df_hasbtneck_sem_["E_AUROC"],
            df_hasbtneck_mean_["E_AUROC"] - n_sems * df_hasbtneck_sem_["E_AUROC"],
            color=colors["hasbtneck"],
            alpha=0.1,
        )
        # ==========FINITE: NO BTNECK BAE====================
        df_nobtneck_mean_, df_nobtneck_sem_ = get_mean_sem(
            optim_df,
            {
                tasks_col_name: target_dim,
                "noise_type": noise_type,
                "HAS_BTNECK": "NO",
                # "isBAE": "bae",
                "bae_type": "ae",
            },
            ["noise_type", "noise_scale"],
        )
        (nobtneck_line,) = ax.plot(
            df_nobtneck_mean_["noise_scale"] * noise_level_factor,
            df_nobtneck_mean_["E_AUROC"],
            color=colors["nobtneck"],
            linewidth=linewidth,
        )
        ax.fill_between(
            df_nobtneck_mean_["noise_scale"] * noise_level_factor,
            df_nobtneck_mean_["E_AUROC"] + n_sems * df_nobtneck_sem_["E_AUROC"],
            df_nobtneck_mean_["E_AUROC"] - n_sems * df_nobtneck_sem_["E_AUROC"],
            color=colors["nobtneck"],
            alpha=0.1,
        )

        sens_inf = calc_srm(df_inf_mean_)
        sens_btneck = calc_srm(df_hasbtneck_mean_)
        sens_nobtneck = calc_srm(df_nobtneck_mean_)

        print("====YES BTNECK====")
        print(str(target_dim) + " " + noise_type)
        print("HAS-BTNECK" + str(sens_btneck.round(3)))
        print("NO-BTNECK" + str(sens_nobtneck.round(3)))
        print("INF:" + str(sens_inf.round(3)))
        print("====INF====")

        if display_legend:
            ax.legend(
                [hasbtneck_line, nobtneck_line, inf_line],
                [
                    "Det. AE (Bottlenecked)",
                    "Det. AE (No bottleneck)",
                    # "BAE-Inf (No bottleneck)",
                    "BAE-" + r"$\infty$" + "(No bottleneck)",
                ],
                fontsize="x-small",
            )
        # ======LABELS=========
        # ax.set_aspect("equal")
        noise_label_map = {
            "normal": r"$2\sigma^{Gaussian}$",
            "uniform": r"$\sigma^{Uniform}$",
        }
        xlabel = noise_label_map[noise_type]
        ax.set_xlabel("Noise level " + xlabel)
        ax.set_ylabel(r"$\Delta$" + "AUROC")
        ax.set_xticks(df_nobtneck_mean_["noise_scale"] * noise_level_factor)

        # ==== set yscale?===
        # ax.set_yscale("log")

        fig.tight_layout()

        fig.canvas.manager.set_window_title(str(target_dim) + "-" + str(noise_type))

        # save and accumulate traces
        trace_title = dataset + "-" + str(target_dim) + noise_type
        new_dict = {
            noise_type: {
                "bae-inf": {
                    "noise_scale": df_inf_mean_["noise_scale"],
                    "mean": df_inf_mean_["E_AUROC"],
                    "sem": df_inf_sem_["E_AUROC"],
                },
                "nobtneck-ae": {
                    "noise_scale": df_nobtneck_mean_["noise_scale"],
                    "mean": df_nobtneck_mean_["E_AUROC"],
                    "sem": df_nobtneck_sem_["E_AUROC"],
                },
                "hasbtneck-ae": {
                    "noise_scale": df_hasbtneck_mean_["noise_scale"],
                    "mean": df_hasbtneck_mean_["E_AUROC"],
                    "sem": df_hasbtneck_sem_["E_AUROC"],
                },
            }
        }
        key_ = dataset + "-" + str(target_dim)
        if key_ not in final_dict_res.keys():
            final_dict_res.update({key_: new_dict})
        else:
            final_dict_res[key_].update(new_dict)
        # =======save figure======
        if save_fig:
            fig.savefig(
                trace_title + ("-legend" if display_legend else "") + "-noise.png",
                dpi=550,
            )

# for aggregation plots
if save_pickle:
    filename = dataset + "-sens-traces.p"
    pickle.dump(final_dict_res, open(dataset + "-sens-traces.p", "wb"))
    print("Saved traces: " + filename)
