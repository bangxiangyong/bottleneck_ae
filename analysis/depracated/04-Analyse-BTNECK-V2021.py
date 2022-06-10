import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from pprint import pprint

# dataset = "FashionMNIST"
# dataset = "CIFAR"
dataset = "STRATH"
# dataset = "ZEMA"
# dataset = "BENCHMARK"

all_pivots = []

# res_type = "sem"
res_type = "mean"

for dataset in [
    "CIFAR",
    "FashionMNIST",
    "BENCHMARK",
    "ZEMA",
    "STRATH",
]:

    path_name = {"ZEMA": "ZEMA_HYD_", "STRATH": "STRATH_FORGE_"}
    task_name = {
        "ZEMA": "target_dim",
        "STRATH": "ss_id",
        "CIFAR": "id_dataset",
        "FashionMNIST": "id_dataset",
        "BENCHMARK": "dataset",
    }
    subtitle_labels = {
        "STRATH": [
            "(i) L-ACTpos",
            "(ii) A-ACTspd",
            "(iii) Feedback-SPA",
            "(iv) All sensors",
        ],
        "ZEMA": [
            "(i) Cooler",
            "(ii) Valve",
            "(iii) Pump",
            "(iv) Accumulator",
        ],
    }
    target_maps = {
        "cardio.mat": "Cardio",
        "lympho.mat": "Lympho",
        "optdigits.mat": "Optdigits",
        "ionosphere.mat": "Ionosphere",
        "pendigits.mat": "Pendigits",
        "thyroid.mat": "Thyroid",
        "pima.mat": "Pima",
        "vowels.mat": "Vowels",
        0: "ZeMA(i)",
        1: "ZeMA(ii)",
        2: "ZeMA(iii)",
        3: "ZeMA(iv)",
        "13": "STRATH(i)",
        "25": "STRATH(ii)",
        "71": "STRATH(iii)",
        "[13, 25, 71]": "STRATH(iv)",
        "FashionMNIST": "FashionMNIST",
        "CIFAR": "CIFAR"
        # "0": "ZeMA:Cooler",
        # "1": "ZeMA:Valve",
        # "2": "ZeMA:Pump",
        # "3": "ZeMA:Accumulator",
        # "13": "STRA:L-ACTpos",
        # "25": "STRA:A-ACTspd",
        # "71": "STRA:Feedback-SPA",
        # "[13, 25, 71]": "STRA:All",
    }

    color_bae = {"ae": "tab:blue", "ens": "tab:orange", "vae": "tab:green"}
    # define maps
    bae_type_map = {
        "ae": "Deterministic AE",
        "ens": "BAE-Ensemble",
        "mcd": "BAE-MCD",
        "vi": "BAE-BBB",
        "sghmc": "BAE-SGHMC",
        "vae": "VAE",
        "INFBAE": "BAE-" + "\scalebox{1.25}{$\infty$}",
    }
    # bae_type_map = {
    #     "ae": "Det. AE",
    #     "ens": "BAE",
    #     "mcd": "BAE",
    #     "vi": "BAE",
    #     "vae": "VAE",
    # }
    bottleneck_map = {
        "A": "Undercomplete, no skip",
        "B": "Undercomplete +skip",
        "C": "Overcomplete, no skip",
        "D": "Overcomplete +skip",
    }

    # load results folder
    res_folders = {
        "STRATH": "STRATH-BOTTLENECKV3",
        "ZEMA": "ZEMA-BOTTLENECKV3",
        "FashionMNIST": "FashionMNIST_BOTTLENECK2022",
        "CIFAR": "CIFAR_BOTTLENECK2022",
        "BENCHMARK": "BENCHMARK_BOTTLENECK",
    }
    suffix = os.path.join("results", res_folders[dataset])
    res_paths = [file for file in os.listdir(suffix) if "AUROC" in file]
    data = pd.concat([pd.read_csv(os.path.join(suffix, file)) for file in res_paths])

    # SELECT MAX MEAN AUROC
    def select_best_auroc(ae_type_res):
        mean_auroc = ae_type_res.groupby(["latent_factor", "skip"]).mean().reset_index()
        max_res = mean_auroc["E_AUROC"].max()
        best_res = mean_auroc[mean_auroc["E_AUROC"] == max_res]
        # print(best_res)
        print(best_res)
        best_skip = best_res["skip"].iloc[0]
        print(best_skip)
        best_latent_factor = best_res["latent_factor"].iloc[0]
        return ae_type_res[
            (ae_type_res["skip"] == best_skip)
            & (ae_type_res["latent_factor"] == best_latent_factor)
        ]

    def extract_ae_type(data, bae_type, target_col):
        ae_data = data[data["bae_type"] == bae_type]

        ae_typeA = ae_data[
            (ae_data["skip"] == False) & (ae_data["latent_factor"] <= 0.5)
        ]
        ae_typeB = ae_data[
            (ae_data["skip"] == True) & (ae_data["latent_factor"] <= 0.5)
        ]
        ae_typeC = ae_data[
            (ae_data["skip"] == False) & (ae_data["latent_factor"] >= 1.0)
        ]
        ae_typeD = ae_data[
            (ae_data["skip"] == True) & (ae_data["latent_factor"] >= 1.0)
        ]

        # typeABCD are selected on mean/max?
        max_res = ae_typeA["E_AUROC"].mean()

        # try select
        ae_typeA = select_best_auroc(ae_typeA)
        ae_typeB = select_best_auroc(ae_typeB)
        ae_typeC = select_best_auroc(ae_typeC)
        ae_typeD = select_best_auroc(ae_typeD)

        ae_typeA["key_type"] = "A"
        ae_typeB["key_type"] = "B"
        ae_typeC["key_type"] = "C"
        ae_typeD["key_type"] = "D"

        # ae_typeA["type"] = bae_type + ae_typeA["type"]
        # ae_typeB["type"] = bae_type + ae_typeB["type"]
        # ae_typeC["type"] = bae_type + ae_typeC["type"]
        # ae_typeD["type"] = bae_type + ae_typeD["type"]
        final_data = pd.concat((ae_typeA, ae_typeB, ae_typeC, ae_typeD))
        final_data["type"] = bae_type + "-" + final_data["key_type"]
        final_data["target"] = final_data[target_col]

        return final_data

    def get_inf_bae(dataset="ZEMA", skip=False, target_col="dataset"):
        pickle_file = "results/InfiniteBAE/" + dataset + "_res_grid_best.p"
        if not os.path.exists(pickle_file):
            return False
        inf_res = pickle.load(open(pickle_file, "rb"))

        inf_res_temp = inf_res[inf_res["skip"] == skip]
        inf_res_temp["type"] = "INF-BAE"
        inf_res_temp["E_AUROC"] = inf_res_temp["AUROC"]
        inf_res_temp[target_col] = inf_res_temp["target"]
        inf_res_temp["key_type"] = "C"
        inf_res_temp["bae_type"] = "INFBAE"
        return inf_res_temp

    def map_legend(full_type):
        if full_type == "INF-BAE":
            # legend_map = "BAE-" + r"$\mathlarger{\mathlarger{\infty}}$"
            # legend_map = "BAE-" + r"$\infty$"
            legend_map = "BAE-" + r"$\infty$" + "(Overcomplete, no skip)"
        else:
            bae_type, bt_type = full_type.split("-")
            legend_map = bae_type_map[bae_type] + "(" + bottleneck_map[bt_type] + ")"
        return legend_map

    def map_legend_col(data_col):
        new_types = []
        for row in data_col:
            new_types.append(map_legend(row))

        return new_types

    def map_targets_col(data_col):
        new_types = []
        for row in data_col:
            new_types.append(target_maps[row])

        return new_types

    concat_temp = [
        extract_ae_type(data=data, bae_type="ae", target_col=task_name[dataset]),
        extract_ae_type(data=data, bae_type="mcd", target_col=task_name[dataset]),
        extract_ae_type(data=data, bae_type="vi", target_col=task_name[dataset]),
        extract_ae_type(data=data, bae_type="ens", target_col=task_name[dataset]),
        extract_ae_type(data=data, bae_type="vae", target_col=task_name[dataset]),
    ]

    inf_bae_pd = get_inf_bae(dataset=dataset, skip=False, target_col=task_name[dataset])

    if not (isinstance(inf_bae_pd, bool) and not inf_bae_pd):
        concat_temp.append(inf_bae_pd)

    all_data = pd.concat(concat_temp)

    # filter STRATH
    # all_data = all_data[~all_data["target"].isin(["13", "25", "71"])]

    # final_data["type"] = map_legend_col(final_data["type"])
    # final_data["target"] = map_targets_col(final_data["target"])

    # =====
    def split_type_col(data_col):
        bae_types = []
        key_types = []
        for row in data_col:
            if row != "INF-BAE":
                bae_type, key_type = row.split("-")
            else:
                bae_type = "INFBAE"
                key_type = "C"

            bae_types.append(bae_type)
            key_types.append(key_type)

        return bae_types, key_types

    # all_data["E_AUROC"] *= 100

    if res_type == "mean":
        new_df_mean = all_data.groupby(["type"]).mean().reset_index()
    elif res_type == "sem":
        new_df_mean = all_data.groupby(["type"]).sem().reset_index()
    bae_types, key_types = split_type_col(new_df_mean["type"])
    new_df_mean["bae_type"] = bae_types
    new_df_mean["key_type"] = key_types

    pivot_df = new_df_mean.pivot(
        index="bae_type", columns="key_type", values="E_AUROC"
    ).reset_index()

    # rearrange by bae_type
    final_df = []
    for bae_type in ["ae", "vae", "mcd", "vi", "ens", "INFBAE"]:
        final_df.append(pivot_df[pivot_df["bae_type"] == bae_type])
    final_df = pd.concat(final_df)

    final_df["bae_type"] = [bae_type_map[row] for row in final_df["bae_type"]]
    final_df = final_df.replace(np.nan, "-")

    # create a header row for each dataset
    header_row = pd.DataFrame(
        [{"bae_type": dataset, "A": "-", "B": "-", "C": "-", "D": "-"}]
    )
    all_pivots.append(header_row)
    all_pivots.append(final_df)

all_pivots_csv = pd.concat(all_pivots).reset_index(drop=True)

# add last mean row
all_pivots_mean = all_pivots_csv.replace("-", np.nan).mean(0, skipna=True)
all_pivots_ate = all_pivots_mean.round(3) - all_pivots_mean["A"].round(3)
all_pivots_mean["bae_type"] = "Mean"
all_pivots_ate["bae_type"] = "ATE"
all_pivots_csv = all_pivots_csv.append(all_pivots_mean, ignore_index=True)
all_pivots_csv = all_pivots_csv.append(all_pivots_ate, ignore_index=True)

# round to 3 deci or percentage
# all_pivots_csv = all_pivots_csv.replace("-", np.nan).round(3).replace(np.nan, "-")
# all_pivots_csv = all_pivots_csv.replace("-", np.nan).round(1)
all_pivots_csv = all_pivots_csv.replace("-", np.nan).round(3)

all_pivots_csv = all_pivots_csv.replace(np.nan, "-")
# all_pivots_csv.to_csv("bottleneck_full_" + res_type + ".csv", index=False)

# ===============
pprint(all_pivots_csv)

# ===============
print(all_pivots_csv)


indices = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]

for index_ in indices:
    rr = all_pivots_csv[index_[0] : index_[1]].replace("-", np.nan).mean(0)

    print(rr)
