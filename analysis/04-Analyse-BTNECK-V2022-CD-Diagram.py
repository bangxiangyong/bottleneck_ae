from scikit_posthocs import posthoc_nemenyi
from scipy.stats import friedmanchisquare

from all_analysis.cd_diagram import draw_cd_diagram, graph_ranks_orange

# from thesis_experiments.cd_diagram import draw_cd_diagram
from pprint import pprint
import Orange
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
    add_df_cols,
)

metric_key_word = "AUROC"
metric_col = "E_" + metric_key_word
dataset_subsets_i = 0
save_fig = True
# save_fig = False
save_csv = False
# save_csv = True
show_legend = True
# show_legend = False
## define path to load the csv and mappings related to datasets
paths = {
    "ZEMA": "../results/zema-btneck-overhaul-repair-v2-20220517",
    "STRATH": "../results/strath-btneck-incomp-v4",
    "ODDS": "../results/odds-btneck-246-20220527",
    "Images": "../results/images-btneck-overhaul-20220525",
}
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
optim_df_list = []
for dataset in paths.keys():
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
    path = paths[dataset]

    all_pivots = []

    ## start reading csv
    raw_df = concat_csv_files(path, key_word=metric_key_word, drop_duplicated=True)

    if dataset == "STRATH":
        raw_df = raw_df[raw_df["resample_factor"] == 50]

    bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
    bae_type_map = {
        "ae": "Deterministic AE",
        "ens": "BAE-Ensemble",
        "mcd": "BAE-MCD",
        "vi": "BAE-BBB",
        "vae": "VAE",
        "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
    }

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

    # group by bae/vae
    isBAE = (raw_df["bae_type"] != "ae") & (raw_df["bae_type"] != "vae")
    isnotBAE = raw_df["bae_type"] == "ae"
    isVAE = raw_df["bae_type"] == "vae"
    raw_df["isBAE"] = np.select([isBAE, isnotBAE, isVAE], ["bae", "ae", "vae"])

    ##======OPTIMISE PARAMS=========
    # raw_df = raw_df[(raw_df["bae_type"] == "ae") | (raw_df["bae_type"] == "ens")]
    # raw_df = raw_df[raw_df["bae_type"] == "ens"]
    optim_df = apply_optim_df(
        raw_df,
        fixed_params=["bae_type", "HAS_BTNECK"],
        # fixed_params=["isBAE", "HAS_BTNECK"],
        optim_params=[
            # "bae_type",
            "current_epoch",
            "latent_factor",
            "skip",
            "layer_norm",
            "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
        ],
        # optim_params=["latent_factor", "skip"],
        perf_key="E_AUROC",
        target_dim_col=tasks_col_map[dataset],
    )

    # ================INF BAE=================
    ## handle inf
    inf_paths = {
        "ZEMA": "../results/zema-inf-noise-revamp-20220528",
        "STRATH": "../results/strath-inf-noise-revamp-20220528",
        "ODDS": "../results/odds-inf-noise-repair-20220527",
        "Images": "../results/images-inf-noise-20220527",
    }
    inf_df = concat_csv_files(
        results_folder=inf_paths[dataset], key_word="AUROC.csv", drop_duplicated=True
    )

    if "noise_scale" in inf_df.columns:
        inf_df = inf_df[
            (inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")
        ]

    inf_df["bae_type"] = "bae_inf"
    inf_df["BTNECK_TYPE"] = "B"
    inf_df["BTNECK_TYPE+INF"] = "INF"
    # inf_df["HAS_BTNECK"] = "INF"
    inf_df["HAS_BTNECK"] = "NO"
    inf_df["isBAE"] = "bae"

    inf_df["current_epoch"] = 1
    inf_df["latent_factor"] = 1
    inf_df["layer_norm"] = inf_df["norm"]

    # inf_df = inf_df[inf_df["num_layers"] == 2]  # limit num layers?
    # inf_df["dataset"] = inf_df["dataset"] + ".mat"  # for odds?
    # fixed_inf_params = ["bae_type", "HAS_BTNECK"]
    fixed_inf_params = ["isBAE", "HAS_BTNECK"]
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

    print("DUPLICATED?:" + str(dataset))
    print(inf_df.iloc[:, 1:].duplicated().sum())

    ## append into final results
    optim_df_combined = pd.concat([optim_df, optim_inf_df])
    # optim_df_combined_ = optim_df
    optim_df_combined["dataset_targetdim"] = (
        dataset + "-" + optim_df_combined[tasks_col_name].astype(str)
    )
    optim_df_list.append(optim_df_combined)

optim_df_list = pd.concat(optim_df_list)

# ==========BARPLOT============
groupby_cols = ["bae_type", "HAS_BTNECK"]
bae_type_map = {
    "ae": "AE",
    "vae": "VAE",
    "mcd": "MCD",
    "vi": "BBB",
    "ens": "Ens",
    # "bae_inf": "BAE-" + r"$\infty$",
    "bae_inf": "BAE-" + r"$\infty$",
}

# Labels
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
    "ZEMA-0": "ZeMA (Cooler)",
    "ZEMA-1": "ZeMA (Valve)",
    "ZEMA-2": "ZeMA (Pump)",
    "ZEMA-3": "ZeMA (Accumulator)",
    "STRATH-2": "STRATH (Radial Forge)",
}
### Select dataset subsets to be plotted
# datasets = np.array(list(dataset_labels.keys()))
#
# n_plots = 5
# plt.rcParams.update({"font.size": 13})
# figsize = (18, 3)
# fig, axes = plt.subplots(1, n_plots, figsize=figsize)
# # fig, axes = plt.subplots(3, n_plots, figsize=figsize, sharex=True)
# axes = axes.flatten()
#
# for i, dataset_targetdim in enumerate(datasets):
#     ax = axes[i]
#     bae_type_order = list(bae_type_map.values())
#     temp_df = optim_df_combined[
#         optim_df_combined["dataset_targetdim"] == dataset_targetdim
#     ]
#     temp_df = temp_df.replace(list(bae_type_map.keys()), list(bae_type_map.values()))
#     temp_df = temp_df.replace(["YES", "NO"], ["Bottlenecked", "Not bottlenecked"])

df_perf = optim_df_list.copy()

df_perf["classifier_name"] = df_perf["bae_type"] + "-" + df_perf["HAS_BTNECK"]
# df_perf["classifier_name"] = df_perf["isBAE"] + "-" + df_perf["HAS_BTNECK"]
# df_perf["dataset_name"] = df_perf["dataset_targetdim"] + df_perf["random_seed"].astype(
# str
# )
df_perf["dataset_name"] = df_perf["dataset_targetdim"]
# df_perf = df_perf[df_perf["dataset_name"] != "ZEMA-0"]  # drop zema0
df_perf["accuracy"] = df_perf[metric_col]

df_perf = (
    # df_perf.groupby(["bae_type", "HAS_BTNECK", "dataset_targetdim"])
    df_perf.groupby(["classifier_name", "dataset_name"])
    .mean()
    .reset_index()
)


# df_perf = df_perf[
#     (df_perf["dataset_name"] == "ZEMA-0891")
#     | (df_perf["dataset_name"] == "ZEMA-040")
#     | (df_perf["dataset_name"] == "ZEMA-0894")
#     | (df_perf["dataset_name"] == "ZEMA-054")
# ]
# classifier_name
# dataset_name
# accuracy
cd_diagram_df = df_perf[["classifier_name", "dataset_name", "accuracy"]]

# cd_diagram_df.loc[cd_diagram_df["classifier_name"] == "ens-NO", "accuracy"] = (
#     cd_diagram_df.loc[cd_diagram_df["classifier_name"] == "ens-NO", "accuracy"] * 100
# )
print("DUPLICATE:")
print(cd_diagram_df.duplicated().sum())

# draw_cd_diagram(df_perf=cd_diagram_df)

draw_cd_diagram(df_perf=cd_diagram_df.drop_duplicates())

# cd_diagram_df[
#     (cd_diagram_df["classifier_name"] == "bae_inf-NO")
#     & (cd_diagram_df["dataset_name"] == "Images-FashionMNIST930")
# ]


# ===========DIY RANKING=========

# cd_diagram_df["score_ranked"] = cd_diagram_df["accuracy"].rank(ascending=0)
all_ranks = []
dataset_names = cd_diagram_df["dataset_name"].unique()
for dataset_name in dataset_names:
    sub_df = cd_diagram_df[cd_diagram_df["dataset_name"] == dataset_name]
    sub_df["score_ranked"] = sub_df["accuracy"].rank(ascending=0)
    print(sub_df)
    all_ranks.append(sub_df["score_ranked"].values)
all_ranks = np.array(all_ranks)

avg_rank = np.mean(all_ranks, axis=0)
classifier_names = cd_diagram_df["classifier_name"].unique()

rank_df = {"classifier_name": classifier_names, "ranks": avg_rank}

print(rank_df)


import Orange
import matplotlib.pyplot as plt

names = classifier_names
avranks = avg_rank
# cd = Orange.evaluation.compute_CD(avranks, len(dataset_names),test="bonferroni-dunn")  # tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5,reverse =True,cdmethod=0)

cd = Orange.evaluation.compute_CD(avranks, len(dataset_names))  # tested on 30 datasets
plt.figure()
graph_ranks_orange(
    avranks, names, cd=cd, width=8, textspace=1.5, reverse=True, lowv=2, highv=9
)

# 11 groups, 15 trials
# all_ranks
pval_friedman = friedmanchisquare(*(all_ranks.T))[1]
print("FRIEDMAN P-VAL:" + str(pval_friedman))

# nemenyi_tests = posthoc_nemenyi(all_ranks.T)
# nemenyi_tests.columns = names
# nemenyi_tests.index = names
#
# plt.figure()
# sns.heatmap(nemenyi_tests)


# ========================Compare ABCD=====================
optim_df_list = []
for dataset in paths.keys():
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
    path = paths[dataset]

    all_pivots = []

    ## start reading csv
    raw_df = concat_csv_files(path, key_word=metric_key_word, drop_duplicated=True)

    if dataset == "STRATH":
        raw_df = raw_df[raw_df["resample_factor"] == 50]

    bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
    bae_type_map = {
        "ae": "Deterministic AE",
        "ens": "BAE-Ensemble",
        "mcd": "BAE-MCD",
        "vi": "BAE-BBB",
        "vae": "VAE",
        "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
    }

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

    # group by bae/vae
    isBAE = (raw_df["bae_type"] != "ae") & (raw_df["bae_type"] != "vae")
    isnotBAE = raw_df["bae_type"] == "ae"
    isVAE = raw_df["bae_type"] == "vae"
    raw_df["isBAE"] = np.select([isBAE, isnotBAE, isVAE], ["bae", "ae", "vae"])

    ##======OPTIMISE PARAMS=========
    # raw_df = raw_df[(raw_df["bae_type"] == "ae") | (raw_df["bae_type"] == "ens")]
    # raw_df = raw_df[raw_df["bae_type"] == "ens"]
    raw_df["num_layers"] = raw_df[
        "n_dense_layers" if dataset == "ODDS" else "n_conv_layers"
    ]
    optim_df = apply_optim_df(
        raw_df,
        fixed_params=["BTNECK_TYPE", "bae_type", "num_layers"],
        optim_params=[
            "current_epoch",
            "latent_factor",
            "skip",
            "layer_norm",
        ],
        perf_key="E_AUROC",
        target_dim_col=tasks_col_map[dataset],
    )

    # ============== INCLUDE BAE INF?===========
    # handle inf
    inf_paths = {
        "ZEMA": "../results/zema-inf-noise-revamp-20220528",
        "STRATH": "../results/strath-inf-noise-revamp-20220528",
        "ODDS": "../results/odds-inf-noise-repair-20220527",
        "Images": "../results/images-inf-noise-20220527",
    }
    inf_df = concat_csv_files(results_folder=inf_paths[dataset], key_word="AUROC.csv")
    inf_df["bae_type"] = "bae_inf"
    inf_df["BTNECK_TYPE"] = "B"
    inf_df["BTNECK_TYPE+INF"] = "INF"
    inf_df["HAS_BTNECK"] = "INF"
    # inf_df["HAS_BTNECK"] = "NO"

    inf_df["current_epoch"] = 1
    inf_df["latent_factor"] = 1
    inf_df["layer_norm"] = inf_df["norm"]

    fixed_inf_params = ["bae_type", "BTNECK_TYPE", "HAS_BTNECK", "num_layers"]
    optim_inf_params = [
        "W_std",
        "diag_reg",
        "norm",
        "activation",
        "skip",
        "current_epoch",
        "latent_factor",
        "layer_norm",
        "noise_scale",
        "noise_type",
    ]
    optim_inf_df = apply_optim_df(
        inf_df,
        fixed_params=fixed_inf_params,
        optim_params=optim_inf_params,
        perf_key="E_AUROC",
        target_dim_col=tasks_col_name,
    )

    # ==========================================
    optim_inf_df["dataset_targetdim"] = (
        dataset + "-" + optim_inf_df[tasks_col_name].astype(str)
    )
    optim_df["dataset_targetdim"] = dataset + "-" + optim_df[tasks_col_name].astype(str)
    optim_df_list.append(optim_df)
    optim_df_list.append(optim_inf_df)
optim_df_combined = pd.concat(optim_df_list)

group_cols = ["BTNECK_TYPE"]
# condition_cols = ["dataset_targetdim", "bae_type", "num_layers"]
# condition_cols = ["dataset_targetdim", "bae_type"]
condition_cols = ["dataset_targetdim"]
optim_df_combined["group"] = add_df_cols(optim_df_combined, group_cols)
optim_df_combined["condition"] = add_df_cols(optim_df_combined, condition_cols)
optim_df_combined_mean = (
    optim_df_combined.groupby(["group", "condition"]).mean().reset_index()
)[["group", "condition", metric_col]]


# ====gather all ranks of groups based on the condition====
all_ranks = []
condition_names = optim_df_combined_mean["condition"].unique()
for condition in condition_names:
    print(sub_df)
    sub_df = optim_df_combined_mean[optim_df_combined_mean["condition"] == condition]
    sub_df["score_ranked"] = sub_df[metric_col].rank(ascending=0)
    all_ranks.append(sub_df["score_ranked"].values)
all_ranks = np.array(all_ranks)
# ===========================================================
classifier_name_map = {
    "A": "(A) Undercomplete -skip",
    "B": "(B) Overcomplete -skip",
    "C": "(C) Undercomplete +skip",
    "D": "(D) Overcomplete +skip",
}
classifier_names = optim_df_combined_mean["group"].unique()
ABCD_names = [
    classifier_name_map[i] if i in classifier_name_map else i for i in classifier_names
]

print("K Group classifiers:" + str(len(optim_df_combined_mean["group"].unique())))
print("N Conditions:" + str(len(condition_names)))
plt.rcParams.update({"font.size": 13})
nobtneck_color = "tab:orange"
hasbtneck_color = "tab:blue"
avg_rank = np.mean(all_ranks, axis=0)
cd = Orange.evaluation.compute_CD(avg_rank, len(condition_names))
fig_cd_1, ax_cd_1 = graph_ranks_orange(
    avg_rank,
    ABCD_names,
    cd=cd,
    width=8,
    textspace=2.3,
    reverse=True,
    show_highest=True,
    linewidth=1.85,
    show_cdlabel=False,
    conditional_colors={
        "(A)": hasbtneck_color,
        "(B)": nobtneck_color,
        "(C)": nobtneck_color,
        "(D)": nobtneck_color,
    },
    thick_color="tab:green",
)

# =====================================================================================
# ================================COMPARE BTNECK MODELS================================
# =====================================================================================
optim_df_list = []
for dataset in paths.keys():
    tasks_col_name = tasks_col_map[
        dataset
    ]  ## enabled only if aggregate_all_tasks is False
    path = paths[dataset]

    all_pivots = []

    ## start reading csv
    raw_df = concat_csv_files(path, key_word=metric_key_word, drop_duplicated=True)

    if dataset == "STRATH":
        raw_df = raw_df[raw_df["resample_factor"] == 50]

    bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
    bae_type_map = {
        "ae": "Deterministic AE",
        "ens": "BAE-Ensemble",
        "mcd": "BAE-MCD",
        "vi": "BAE-BBB",
        "vae": "VAE",
        "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
    }

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

    # group by bae/vae
    isBAE = (raw_df["bae_type"] != "ae") & (raw_df["bae_type"] != "vae")
    isnotBAE = raw_df["bae_type"] == "ae"
    isVAE = raw_df["bae_type"] == "vae"
    raw_df["isBAE"] = np.select([isBAE, isnotBAE, isVAE], ["bae", "ae", "vae"])

    ##======OPTIMISE PARAMS=========
    # raw_df = raw_df[(raw_df["bae_type"] == "ae") | (raw_df["bae_type"] == "ens")]
    # raw_df = raw_df[raw_df["bae_type"] == "ens"]
    raw_df["num_layers"] = raw_df[
        "n_dense_layers" if dataset == "ODDS" else "n_conv_layers"
    ]
    optim_df = apply_optim_df(
        raw_df,
        # fixed_params=["bae_type", "HAS_BTNECK", "num_layers"],
        fixed_params=["bae_type", "HAS_BTNECK"],
        optim_params=[
            # "bae_type",
            "num_layers",
            "current_epoch",
            "latent_factor",
            "skip",
            "layer_norm",
            # "n_dense_layers" if dataset == "ODDS" else "n_conv_layers",
        ],
        # optim_params=["latent_factor", "skip"],
        perf_key="E_AUROC",
        target_dim_col=tasks_col_map[dataset],
    )

    # ================INF BAE=================
    ## handle inf
    inf_paths = {
        "ZEMA": "../results/zema-inf-noise-revamp-20220528",
        "STRATH": "../results/strath-inf-noise-revamp-20220528",
        "ODDS": "../results/odds-inf-noise-repair-20220527",
        "Images": "../results/images-inf-noise-20220527",
    }
    inf_df = concat_csv_files(
        results_folder=inf_paths[dataset], key_word="AUROC.csv", drop_duplicated=True
    )

    if "noise_scale" in inf_df.columns:
        inf_df = inf_df[
            (inf_df["noise_scale"] == 0) & (inf_df["noise_type"] == "uniform")
        ]

    inf_df["bae_type"] = "bae_inf"
    inf_df["BTNECK_TYPE"] = "B"
    inf_df["BTNECK_TYPE+INF"] = "INF"
    # inf_df["HAS_BTNECK"] = "INF"
    inf_df["HAS_BTNECK"] = "NO"
    inf_df["isBAE"] = "bae"

    inf_df["current_epoch"] = 1
    inf_df["latent_factor"] = 1
    inf_df["layer_norm"] = inf_df["norm"]

    # inf_df = inf_df[inf_df["num_layers"] == 2]  # limit num layers?
    # inf_df["dataset"] = inf_df["dataset"] + ".mat"  # for odds?
    fixed_inf_params = ["HAS_BTNECK", "bae_type"]
    # fixed_inf_params = ["HAS_BTNECK", "bae_type", "num_layers"]
    # fixed_inf_params = ["isBAE", "HAS_BTNECK"]
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

    print("DUPLICATED?:" + str(dataset))
    print(inf_df.iloc[:, 1:].duplicated().sum())

    ## append into final results
    optim_df_combined_ = pd.concat([optim_df, optim_inf_df])
    optim_df_combined_["dataset_targetdim"] = (
        dataset + "-" + optim_df_combined_[tasks_col_name].astype(str)
    )
    optim_df_list.append(optim_df_combined_)
optim_df_combined = pd.concat(optim_df_list)

bae_type_map = {
    "ae": "Deterministic AE",
    "vae": "VAE",
    "mcd": "BAE-MCD",
    "vi": "BAE-BBB",
    "ens": "BAE-Ensemble",
    "bae_inf": "BAE-" + r"$\infty$",
}
optim_df_combined = optim_df_combined.replace(
    list(bae_type_map.keys()), list(bae_type_map.values())
)
optim_df_combined = optim_df_combined.replace(["YES", "NO"], [r"$(-)$", r"$(+)$"])
# specify conditions and group cols
group_cols = ["bae_type", "HAS_BTNECK"]
condition_cols = ["dataset_targetdim"]

# combine and rename cols
optim_df_combined["group"] = add_df_cols(optim_df_combined, group_cols)
optim_df_combined["condition"] = add_df_cols(optim_df_combined, condition_cols)
optim_df_combined_mean = (
    optim_df_combined.groupby(["group", "condition"]).mean().reset_index()
)[["group", "condition", metric_col]]


# ====gather all ranks of groups based on the condition====
all_ranks = []
condition_names = optim_df_combined_mean["condition"].unique()
for condition in condition_names:
    print(sub_df)
    sub_df = optim_df_combined_mean[optim_df_combined_mean["condition"] == condition]
    sub_df["score_ranked"] = sub_df[metric_col].rank(ascending=0)
    all_ranks.append(sub_df["score_ranked"].values)
all_ranks = np.array(all_ranks)
# ===========================================================
classifier_names = optim_df_combined_mean["group"].unique()
# ABCD_names = [classifier_name_map[i] for i in classifier_names]

print("K Group classifiers:" + str(len(optim_df_combined_mean["group"].unique())))
print("N Conditions:" + str(len(condition_names)))
plt.rcParams.update({"font.size": 13})
nobtneck_color = "tab:orange"
hasbtneck_color = "tab:blue"
avg_rank = np.mean(all_ranks, axis=0)
cd = Orange.evaluation.compute_CD(avg_rank, len(condition_names))
fig_cd_2, ax_cd_2 = graph_ranks_orange(
    avg_rank,
    classifier_names,
    cd=cd,
    width=8,
    textspace=1.85,
    reverse=True,
    show_highest=True,
    linewidth=1.85,
    show_cdlabel=False,
    conditional_colors={"-": hasbtneck_color, "+": nobtneck_color},
    thick_color="tab:green",
    # highv=9,
)

# ====SAVEFIG=====

if save_fig:
    path_1 = "plots/ABCD-cd-diagram.png"
    path_2 = "plots/btneck-cd-diagram.png"
    fig_cd_1.savefig(path_1, dpi=550)
    fig_cd_2.savefig(path_2, dpi=550)
    print("SAVE CD DIAGRAM:" + path_1)
    print("SAVE CD DIAGRAM:" + path_2)
