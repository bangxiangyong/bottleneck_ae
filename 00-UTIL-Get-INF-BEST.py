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
dataset = "ZEMA"
# dataset = "Images"
# dataset = "ODDS"

metric_key_word = "AUROC"
round_deci = 3
save_fig = False
# save_csv = False
save_csv = True

## define path to load the csv and mappings related to datasets
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False

all_pivots = []

bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
bae_type_map = {
    "ae": "Deterministic AE",
    "ens": "BAE-Ensemble",
    "mcd": "BAE-MCD",
    "vi": "BAE-BBB",
    "sghmc": "BAE-SGHMC",
    "vae": "VAE",
    "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
}

##======CREATE LABELS OF BTNECK TYPE=====

## handle inf
inf_paths = {
    "ZEMA": "results/zema-inf-noise-revamp-20220528",
    "STRATH": "results/strath-inf-noise-revamp-20220528",
    "ODDS": "results/odds-inf-noise-repair-20220527",
    "Images": "results/images-inf-noise-20220527",
}
inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC.csv")

if "noise_scale" in inf_df.columns:
    inf_df = inf_df[(inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")]
    # inf_df = inf_df.drop(["noise_scale", "noise_type"])

inf_df["bae_type"] = "bae_inf"
inf_df["BTNECK_TYPE"] = "B"
inf_df["BTNECK_TYPE+INF"] = "INF"
# inf_df["HAS_BTNECK"] = "INF"
inf_df["HAS_BTNECK"] = "NO"

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
    fixed_params=fixed_inf_params,
    optim_params=optim_inf_params,
    perf_key="E_AUROC",
    target_dim_col=tasks_col_name,
)

# ==================================

cols = [tasks_col_name, "W_std", "diag_reg", "norm", "num_layers", "activation", "skip"]
best_params = {}
for target_dim in optim_inf_df[tasks_col_name].unique():
    # for target_dim in [0]:
    res_groupby = optim_inf_df[optim_inf_df[tasks_col_name] == target_dim][cols]
    best_entry = res_groupby.drop_duplicates().to_dict("records")[0]
    best_params.update({target_dim: best_entry})
print(best_params)
{
    2: {
        "target_dim": 2,
        "W_std": 0.75,
        "diag_reg": 1e-10,
        "norm": "none",
        "num_layers": 13,
        "activation": "leakyrelu",
        "skip": False,
    }
}

# ===================================
