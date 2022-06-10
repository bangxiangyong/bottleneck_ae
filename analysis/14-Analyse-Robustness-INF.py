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
# optim_inf_df = apply_optim_df(
#     inf_df,
#     fixed_params=["noise_type"],
#     optim_params=["num_layers", "layer_norm", "W_std"],
#     perf_key="E_AUROC",
#     target_dim_col=tasks_col_name,
# )

optim_inf_df = apply_optim_df(
    inf_df,
    fixed_params=["noise_type", "W_std", "noise_scale"],
    optim_params=["num_layers", "layer_norm"],
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
        temp_df = optim_inf_df[optim_inf_df[tasks_col_name] == target_dim]
        temp_df = temp_df[temp_df["noise_type"] == noise_type]
        temp_df = temp_df.groupby(["W_std", "noise_scale"]).mean().reset_index()
        print("=================================")
        print(target_dim)
        print(noise_type)
        for w_std in temp_df["W_std"].unique():
            w_std_df = temp_df[temp_df["W_std"] == w_std]
            ax.plot(w_std_df["noise_scale"], w_std_df[metric_col])

            print("WSTD:" + str(w_std))
            print(
                np.exp(np.mean(w_std_df[metric_col][1:] - w_std_df[metric_col].iloc[0]))
            )
# for aggregation plots
if save_pickle:
    filename = dataset + "-sens-traces.p"
    pickle.dump(final_dict_res, open(dataset + "-sens-traces.p", "wb"))
    print("Saved traces: " + filename)
