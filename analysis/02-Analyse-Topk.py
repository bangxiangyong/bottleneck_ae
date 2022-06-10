# 1. Creates graph of Topk sensors vs AUROC to optimise the number of sensors required
# 2. Augments and outputs experimental grid with best Topk sensors for each target dim


import itertools
import pandas as pd
import matplotlib.pyplot as plt
from thesis_experiments.case_study import Params_STRATH, Params_ZEMA
from thesis_experiments.util_analyse import (
    concat_csv_files,
    save_pickle_grid,
    add_df_cols,
    apply_optim_df,
    rearrange_df,
)
import seaborn as sns
import numpy as np
import json
from pprint import pprint

plt.rcParams.update({"font.size": 20})
dataset = "ZEMA"
# dataset = "STRATH"

# OPTION: Run second half of the script to select only the best TopK
# save_best_topk_grid = True
save_best_topk_grid = False
save_fig = True
# save_fig = False
save_fig_folder = "plots/"
# pickle_grid_name = "-reboot-ll-grid-20220418.p"
# pickle_grid_name = "-bottleneck-grid-20220418.p"
# pickle_grid_name = "-stgauss-ll-grid-20220428.p"
pickle_grid_name = "-stdmse-ll-grid-20220429.p"

results_folders = {
    # "ZEMA": "../results/zema-topk-reboot-20220418",
    # "STRATH": "../results/strath-topk-reboot-20220418",
    # "ZEMA": "../results/zema-stgauss-topk-20220428",
    # "STRATH": "../results/strath-stgauss-topk-20220428",
    # "ZEMA": "../results/zema-stdmse-topk-20220429",
    # "STRATH": "../results/strath-stdmse-topk-20220429",
    "ZEMA": "../results/zema-topk-20220514",
    "STRATH": "../results/strath-topk-20220514",
}
results_folder = results_folders[dataset]

raw_auroc_topk = concat_csv_files(results_folder, key_word="AUROC.csv")

# ======================================
optim_df = apply_optim_df(
    raw_auroc_topk,
    fixed_params=["bae_type", "full_likelihood"],
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col="target_dim",
)

# ========================================

# mean over random seeds
groupby_columns = ["target_dim", "k_sens", "full_likelihood", "ss_id"]
topk_mean = optim_df.groupby(groupby_columns).mean()["E_AUROC"].reset_index()

topk_res = []
topk_df = []
for target_dim in topk_mean["target_dim"].unique():
    temp_targetdim = topk_mean[topk_mean["target_dim"] == target_dim]
    # get max params for layer_norm
    for ll in temp_targetdim["full_likelihood"].unique():
        temp_ll = temp_targetdim[temp_targetdim["full_likelihood"] == ll]
        max_row = temp_ll.iloc[temp_ll["E_AUROC"].argmax()]
        max_params = {
            "k_sens": max_row["k_sens"],
        }

        # add raw AUROCS into lists for boxplot
        # by filtering using the max args found
        max_topk_ll = optim_df[
            (optim_df["full_likelihood"] == ll) & (optim_df["target_dim"] == target_dim)
        ]

        # now collect the topk results into list of AUROCs
        topk_aurocs = []
        for k in range(max_topk_ll["k_sens"].max()):
            topk_aurocs.append(
                max_topk_ll[max_topk_ll["k_sens"] == k + 1]["E_AUROC"].values
            )

        # update results by target dim and
        topk_res.append(
            {
                "target_dim": target_dim,
                "full_likelihood": ll,
                "topk_scores": topk_aurocs,
            }
        )
        topk_df.append(max_topk_ll.copy())

topk_df = pd.concat(topk_df)

# PLOT: box plots
# for target_dim in topk_df["target_dim"].unique():
#     topk_df_tgt_dim = topk_df[topk_df["target_dim"] == target_dim]
#     plt.figure()
#     sns.boxplot(
#         x="k_sens",
#         y="E_AUROC",
#         hue="full_likelihood",
#         data=topk_df_tgt_dim,
#         showfliers=False,
#     )


# PLOT: Error bars
# figsize = (5.5, 4)
figsize = (7.5, 4)
# figsize = (4, 3)
perf_key = "E_AUROC"
rearrange_labels = [
    "bernoulli",
    "cbernoulli",
    "mse",
    "std-mse",
    "static-tgauss",
]
ll_map = {
    "bernoulli": "Bernoulli",
    "cbernoulli": "C-Bernoulli",
    "mse": "Gaussian",
    "std-mse": "Gaussian (Z-std)",
    "static-tgauss": "Trunc. Gaussian",
}
legend_lines = []
total_targets = len(topk_df["target_dim"].unique())
for tgt_i, target_dim in enumerate(topk_df["target_dim"].unique()):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for likelihood in rearrange_labels:
        temp_df = topk_df[
            (topk_df["full_likelihood"] == likelihood)
            & (topk_df["target_dim"] == target_dim)
        ]
        temp_groupby = temp_df.groupby("k_sens")

        # get mean and sem
        temp_df_mean = temp_groupby.mean().reset_index()
        temp_df_sem = temp_groupby.sem().reset_index()

        # plot
        x_range_sensors = np.arange(topk_df["k_sens"].max()) + 1
        (line,) = ax.plot(
            x_range_sensors,
            temp_df_mean[perf_key],
            marker="o",
            markersize=5.5,
            alpha=0.6,
            linewidth=2.5,
        )
        ax.fill_between(
            x_range_sensors,
            np.clip(temp_df_mean[perf_key] + temp_df_sem[perf_key] * 2, 0, 1),
            np.clip(temp_df_mean[perf_key] - temp_df_sem[perf_key] * 2, 0, 1),
            alpha=0.2,
        )
        legend_lines.append(line)
    # labels
    ax.grid(True, color="grey", linewidth="0.5", axis="y", alpha=0.5)
    if dataset == "STRATH":
        ax.set_xticks(x_range_sensors)
    else:
        ax.set_xticks([1, 5, 10, 14])
    ax.set_xlabel("Top-k sensors")
    ax.set_ylabel("AUROC")
    ax.set_ylim(ymax=1.05)

    # add legend
    # if tgt_i == (total_targets - 4):
    if tgt_i == 0:
        ### Put legends outside
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.96, box.height])
        ax.legend(
            legend_lines,
            [ll_map[l] for l in rearrange_labels],
            fontsize="x-small",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        ## Put legends inside
        # ax.legend(
        #     legend_lines,
        #     [ll_map[l] for l in rearrange_labels],
        #     fontsize="x-small",
        #     loc="lower right",
        # )
    fig.tight_layout()
    if save_fig:
        fig.savefig(
            save_fig_folder + dataset + "-" + str(target_dim) + "-TOPK+LEGEND.png",
            dpi=500,
        )
# ===========CREATE GRID FOR BEST TOPK SENSORS===========
# GET BEST TOPK SENS for each target dim
# likelihood = "mse"
target_dims_topk = {}
for target_dim in topk_mean["target_dim"].unique():
    target_dims_topk.update({target_dim: {}})
    for likelihood in topk_mean["full_likelihood"].unique():
        topk_mean_temp = topk_mean[
            (topk_mean["full_likelihood"] == likelihood)
            & (topk_mean["target_dim"] == target_dim)
        ]
        max_row = topk_mean_temp.iloc[topk_mean_temp["E_AUROC"].argmax()]
        best_sensors = json.loads(max_row["ss_id"])
        target_dims_topk[target_dim].update(
            {
                likelihood: {
                    "ss_id": best_sensors,
                    "k_sens": len(best_sensors),
                    "E_AUROC": np.round(max_row["E_AUROC"], 3),
                }
            }
        )
print("DATASET:" + str(dataset))
print("TARGET DIM BEST-TOPK SENSORS:")
pprint(target_dims_topk)

# ===================================
# import base grid and unpack it
if dataset == "ZEMA":
    grid_base = Params_ZEMA.grid_ZEMA
elif dataset == "STRATH":
    grid_base = Params_STRATH.grid_STRATH

# ============CREATE BEST TOPK GRID FOR FULL LL vs BAE TABLE=============

if save_best_topk_grid:
    grid_keys = list(grid_base.keys())
    grid_list = list(itertools.product(*grid_base.values()))

    new_grid_list = []
    # iterate through grid list, and replace
    for values in grid_list:
        exp_params = dict(zip(grid_keys, values))

        # update ss_id with the top K
        target_dim = exp_params["target_dim"]
        likelihood = exp_params["full_likelihood"]
        exp_params.update({"ss_id": target_dims_topk[target_dim][likelihood]["ss_id"]})

        # add to new grid list
        new_grid_list.append(list(exp_params.values()))

    # VALIDATE GRID LENGTH
    print("GRID LENGTH:" + str(len(new_grid_list)))
    validate_grid = len(new_grid_list) == len(grid_list)
    print("VALIDATE GRID LENGTH:" + str(validate_grid))
    if not validate_grid:
        raise Exception("Grid length not the same as expected.")

    # save pickle grid
    save_pickle_grid(
        grid_keys,
        new_grid_list,
        grid_filename="../grids/" + dataset + pickle_grid_name,
    )
