# This accompanying script plots the pickles from script 10-Analyse-BTNECK-SensitivityNoise.py

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

# =====Seettings=====
# save_csv = False
save_csv = True

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
# dataset = "CIFAR"  # {"CIFAR","FashionMNIST", "ODDS", "ZEMA","STRATH"}
# datasets = [dataset_ for dataset_ in dataset_labels if dataset in dataset_]
#

final_res = {}
for dataset in ["CIFAR", "FashionMNIST", "ODDS", "ZEMA", "STRATH"]:
    datasets = [dataset_ for dataset_ in dataset_labels if dataset in dataset_]
    for noise_type in ["uniform", "normal"]:
        inf_mean_all = []
        hasbtneck_mean_all = []
        nobtneck_mean_all = []
        for i, dataset_targetdim in enumerate(datasets):
            # inf_mean_all = []
            # hasbtneck_mean_all = []
            # nobtneck_mean_all = []
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

        hasbtneck_mean_all = np.clip(hasbtneck_mean_all, None, 0)
        inf_mean_all = np.clip(inf_mean_all, None, 0)
        nobtneck_mean_all = np.clip(nobtneck_mean_all, None, 0)

        # ======MEAN & SEM DELTA AUROCS=====
        round_deci = 3
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

        # add new entry to dict
        if dataset in final_res.keys():
            final_res[dataset].update(new_res)
        else:
            final_res.update({dataset: new_res})

        # if dataset_targetdim in final_res.keys():
        #     final_res[dataset_targetdim].update(new_res)
        # else:
        #     final_res.update({dataset_targetdim: new_res})
# ===========DISPLAY AS TABLE===============

# masuk table
# final_res["CIFAR"]
# pd.DataFrame({"CIFAR-uniform":})

all_res = []
for dataset, noise_type_dict in final_res.items():
    for noise_type, vals in noise_type_dict.items():
        bae_inf_res = "{:.3f}".format(
            vals["bae-inf"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["bae-inf"]["sem"])
        hasbtneck_res = "{:.3f}".format(
            vals["hasbtneck-ae"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["hasbtneck-ae"]["sem"])

        nobtneck_res = "{:.3f}".format(
            vals["nobtneck-ae"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["nobtneck-ae"]["sem"])

        all_res.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "hasbtneck": hasbtneck_res,
                "nobtneck": nobtneck_res,
                "bae-inf": bae_inf_res,
            }
        )

# Save csv
all_res_df = pd.DataFrame(all_res)
print(all_res_df)
if save_csv:
    filepath = "tables/all-sens-noise-table.csv"
    print("SAVE TABLE CSV:" + filepath)
    all_res_df.to_csv(filepath, index=False)


# ======================UNFOLD-DETAILED TABLE====================================

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

            hasbtneck_mean_all = np.clip(hasbtneck_mean_all, None, 0)
            inf_mean_all = np.clip(inf_mean_all, None, 0)
            nobtneck_mean_all = np.clip(nobtneck_mean_all, None, 0)

            # ======MEAN & SEM DELTA AUROCS=====
            round_deci = 3
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
# ===========DISPLAY AS TABLE===============

# masuk table
# final_res["CIFAR"]
# pd.DataFrame({"CIFAR-uniform":})

detailed_res = []
for dataset, noise_type_dict in final_res.items():
    for noise_type, vals in noise_type_dict.items():
        bae_inf_res = "{:.3f}".format(
            vals["bae-inf"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["bae-inf"]["sem"])
        hasbtneck_res = "{:.3f}".format(
            vals["hasbtneck-ae"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["hasbtneck-ae"]["sem"])

        nobtneck_res = "{:.3f}".format(
            vals["nobtneck-ae"]["mean"]
        ) + r"$\pm$" "{:.3f}".format(vals["nobtneck-ae"]["sem"])

        detailed_res.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "hasbtneck": hasbtneck_res,
                "nobtneck": nobtneck_res,
                "bae-inf": bae_inf_res,
            }
        )

detailed_res_df = pd.DataFrame(detailed_res)
# ====get minimum====
argmins = np.argmin(
    detailed_res_df[["hasbtneck", "nobtneck", "bae-inf"]].values, axis=1
)
for row_i in range(len(detailed_res_df)):
    # val = "\\textbf{" + detailed_res_df.iloc[row_i, argmins[row_i] + 2] + "}"
    val = "*" + detailed_res_df.iloc[row_i, argmins[row_i] + 2]

    detailed_res_df.iloc[row_i, argmins[row_i] + 2] = val

# Save csv
print(detailed_res_df)
if save_csv:
    filepath = "tables/detailed-sens-noise-table.csv"
    print("SAVE TABLE CSV:" + filepath)
    detailed_res_df.to_csv(filepath, index=False)
