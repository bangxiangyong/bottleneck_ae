import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from thesis_experiments.util_analyse import concat_csv_files
import numpy as np

matplotlib.rcParams.update({"errorbar.capsize": 2})

latent_factor_map = {
    # 0.1: r"$\times{\dfrac{1}{10}}$",
    0.5: r"$\times{\dfrac{1}{2}}$",
    1.0: r"$\times{1}$",
    2.0: r"$\times{2}$",
    # 10.0: r"$\times{10}$",
    # 1000: r"$\infty$",
}
# dataset = "ZEMA"
dataset = "STRATH"

zema_folder = "../results/time-zema-v3-20220609"
strath_folder = "../results/time-strath-v3-20220609"

if dataset == "ZEMA":
    time_df = concat_csv_files(zema_folder, key_word="TIME")
elif dataset == "STRATH":
    time_df = concat_csv_files(strath_folder, key_word="TIME")

# ae_zema_df = zema_df[zema_df["bae_type"] == "ae"]
# ae_zema_df_noskip = ae_zema_df[ae_zema_df["skip"] == False]
# ae_zema_df_skip = ae_zema_df[ae_zema_df["skip"] == True]
# ae_zema_df_noskip_mean = (
#     ae_zema_df_noskip.groupby(["latent_factor"]).mean().reset_index()
# )
# ae_zema_df_skip_mean = ae_zema_df_skip.groupby(["latent_factor"]).mean().reset_index()


bae_zema_df = time_df[time_df["bae_type"] == "ens"]
bae_zema_df["TIME-FIT"] *= 200
bae_zema_df["TIME-PRED"] /= bae_zema_df["n_samples_test"].unique()[0]

# divide by number of samples?
bae_zema_df["TIME-FIT"] /= bae_zema_df["n_bae_samples"].unique()[0]
bae_zema_df["TIME-PRED"] /= bae_zema_df["n_bae_samples"].unique()[0]

bae_zema_df_noskip = bae_zema_df[bae_zema_df["skip"] == False]
bae_zema_df_skip = bae_zema_df[bae_zema_df["skip"] == True]
bae_zema_df_noskip_mean = (
    bae_zema_df_noskip.groupby(["latent_factor"]).mean().reset_index()
)
bae_zema_df_noskip_sem = (
    bae_zema_df_noskip.groupby(["latent_factor"]).sem().reset_index()
)
bae_zema_df_skip_mean = bae_zema_df_skip.groupby(["latent_factor"]).mean().reset_index()
bae_zema_df_skip_sem = bae_zema_df_skip.groupby(["latent_factor"]).sem().reset_index()

len_points = np.arange(len(bae_zema_df_noskip_mean["latent_factor"]))


marker_size = 3
figsize_scale = 2.0
figsize = (figsize_scale * 2, figsize_scale)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
# ax1.plot(len_points, ae_zema_df_noskip_mean["TIME-FIT"] * fit_epoch_scale, marker="o")
# ax1.plot(len_points, ae_zema_df_skip_mean["TIME-FIT"] * fit_epoch_scale, marker="o")
# ax2.plot(len_points, ae_zema_df_noskip_mean["TIME-PRED"], marker="o")
# ax2.plot(len_points, ae_zema_df_skip_mean["TIME-PRED"], marker="o")

ax1.errorbar(
    len_points,
    bae_zema_df_noskip_mean["TIME-FIT"].sort_values(),
    marker="o",
    markersize=marker_size,
    yerr=bae_zema_df_noskip_sem["TIME-FIT"][
        bae_zema_df_noskip_mean["TIME-FIT"].sort_values().index
    ],
)
ax1.errorbar(
    len_points,
    bae_zema_df_skip_mean["TIME-FIT"].sort_values(),
    marker="o",
    markersize=marker_size,
    yerr=bae_zema_df_skip_sem["TIME-FIT"][
        bae_zema_df_skip_mean["TIME-FIT"].sort_values().index
    ],
)


ax2.errorbar(
    len_points,
    bae_zema_df_noskip_mean["TIME-PRED"].sort_values(),
    marker="o",
    markersize=marker_size,
    yerr=bae_zema_df_skip_sem["TIME-PRED"][
        bae_zema_df_skip_mean["TIME-PRED"].sort_values().index
    ],
)
ax2.errorbar(
    len_points,
    bae_zema_df_skip_mean["TIME-PRED"].sort_values(),
    marker="o",
    markersize=marker_size,
    yerr=bae_zema_df_skip_sem["TIME-PRED"][
        bae_zema_df_skip_mean["TIME-PRED"].sort_values().index
    ],
)

ax1.set_xticks(len_points)
ax2.set_xticks(len_points)

ax1.set_xticklabels(list(latent_factor_map.values()))
ax2.set_xticklabels(list(latent_factor_map.values()))

ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
# ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

# label
ax1.set_ylabel("Time (s)")
ax1.set_xlabel("Latent factor")
ax2.set_xlabel("Latent factor")

plt.rcParams["axes.titley"] = 1.0  # y is in axes-relative coordinates.
plt.rcParams["axes.titlepad"] = -14  # pad is in points...

ax1.set_title("Training")
ax2.set_title("Prediction")
ax1.yaxis.grid(alpha=0.5)
ax2.yaxis.grid(alpha=0.5)

fig.tight_layout(pad=0.1, w_pad=1)
fig.savefig("plots/" + dataset + "-time.png", dpi=600)
