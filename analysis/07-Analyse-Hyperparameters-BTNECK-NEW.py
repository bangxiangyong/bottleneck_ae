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
    # "STRATH": "../results/strath-hyp-btneck-partial-v3-20220513",
    # "STRATH": "../results/strath-hyp-btneck-revamp-20220514",
    "STRATH": "../results/strath-hyp-small-cap-20220514",
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
# compare_col = "BTNECK_TYPE+INF"
dataset = "STRATH"

tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
path = paths[dataset]

## start reading csv
raw_df = concat_csv_files(path, key_word=metric_key_word, drop_duplicated=True)

num_layer = 1

optim_df = apply_optim_df(
    raw_df,
    # fixed_params=["bae_type", "BTNECK_TYPE"],
    fixed_params=["latent_factor", "skip", "n_conv_layers", "n_enc_capacity"],
    optim_params=["current_epoch"],
    perf_key="E_" + metric_key_word,
    target_dim_col=tasks_col_map[dataset],
)
# temp_df = optim_df[optim_df["n_conv_layers"] == 2]
# optim_df = raw_df[raw_df["current_epoch"] == 200]

# =======CAPACITY VS LATENT FACTOR=======
# optim_df = raw_df.copy()
optim_df = raw_df[raw_df["current_epoch"] == 200]
n_conv_layers = optim_df["n_conv_layers"].unique()[-1]
latent_factor = optim_df["latent_factor"].unique()[0]

skip = False
# skip = True
# n_layers = 3
legends = []
plt.figure()
for latent_factor in optim_df["latent_factor"].unique():
    temp_df = optim_df[
        (optim_df["latent_factor"] == latent_factor)
        & (optim_df["n_conv_layers"] == n_conv_layers)
        & (optim_df["skip"] == skip)
    ]

    temp_df_mean = temp_df.groupby("n_enc_capacity").mean().reset_index()
    # temp_df_mean = temp_df.groupby("n_enc_capacity").median().reset_index(drop=True)
    temp_df_sem = temp_df.groupby("n_enc_capacity").sem().reset_index(drop=True)

    (line,) = plt.plot(temp_df_mean["E_" + metric_key_word])
    legends.append(line)
    print(latent_factor)
# print(temp_df_mean["latent_factor"].unique())
plt.legend(legends, optim_df["latent_factor"].unique())

# =======SKIP VS DEPTH===============
# choose and select the nicest ones?
optim_df = raw_df[raw_df["current_epoch"] == 200]
# optim_df = raw_df.copy()
# skip = False
# skip = True
n_enc_capacity = optim_df["n_enc_capacity"].unique()[-1]
latent_factor = optim_df["latent_factor"].unique()[0]
legends = []

plt.figure()
for skip in [True, False]:
    for latent_factor in [latent_factor]:
        temp_df = optim_df[
            (optim_df["latent_factor"] == latent_factor)
            & (optim_df["n_enc_capacity"] == n_enc_capacity)
            & (optim_df["skip"] == skip)
        ]

        temp_df_mean = temp_df.groupby("n_conv_layers").mean().reset_index()
        # temp_df_mean = temp_df.groupby("n_enc_capacity").median().reset_index(drop=True)
        temp_df_sem = temp_df.groupby("n_conv_layers").sem().reset_index(drop=True)

        (line,) = plt.plot(
            temp_df_mean["n_conv_layers"], temp_df_mean["E_" + metric_key_word]
        )
        legends.append(line)
        print(latent_factor)
plt.legend(
    legends,
    [
        "+SKIP",
        "NO SKIP",
    ],
)

# =========LATENT========================

optim_df = raw_df[raw_df["current_epoch"] == 200]
n_conv_layers = 0
for n_enc_capacity in np.sort(optim_df["n_enc_capacity"].unique()):

    temp_df = raw_df[
        (raw_df["n_enc_capacity"] == n_enc_capacity)
        & (raw_df["n_conv_layers"] == n_conv_layers)
    ]
    optim_df = temp_df.groupby(["latent_factor"]).mean().reset_index()

    plt.figure()
    plt.plot(optim_df["latent_factor"], optim_df["E_" + metric_key_word])
    plt.title(str(n_enc_capacity))

# =============================================
