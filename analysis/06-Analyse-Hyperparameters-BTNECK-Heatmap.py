import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =======================VARY ENC CAPACITY======================================
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TESTAUROC.csv")
from thesis_experiments.util_analyse import apply_optim_df

res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TESTAUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TESTAVGPRC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_v2AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_long_AUROC.csv")
# metric_key_word = "AVGPRC"
metric_key_word = "AUROC"

fixed_factors = ["n_enc_capacity", "latent_factor"]
mean_df = res_df.groupby(fixed_factors).mean().reset_index()
pivot_df = mean_df.pivot(*fixed_factors, "E_" + metric_key_word)

# VIEW 1
plt.figure()
sns.heatmap(
    pivot_df,
    cmap="viridis",
    annot=True,
    fmt=".3f",
)

# VIEW 2
# cat = "n_enc_capacity"
# x_axis = "latent_factor"
cat = "latent_factor"
x_axis = "n_enc_capacity"
all_lines = []
fig, ax = plt.subplots(1, 1)
for enc_cap in mean_df[cat].unique():
    temp = mean_df[mean_df[cat] == enc_cap]
    x_length = np.arange(len(temp[x_axis].unique()))
    (line,) = ax.plot(x_length, temp["E_" + metric_key_word])
    ax.set_xticks(x_length)
    ax.set_xticklabels(temp[x_axis].unique())
    all_lines.append(line)
plt.legend(all_lines, mean_df[cat].unique())

# =====================VARY NUM LAYERS===============================
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_long_AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_long3AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_SKIP2AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW2AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW5AUROC.csv")
# # res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW6AUROC.csv")
# metric_key_word = "AUROC"
# fixed_factors = ["n_dense_layers", "latent_factor", "skip"]
# # res_df = apply_optim_df(
# #     res_df,
# #     fixed_params=fixed_factors,
# #     optim_params=["current_epoch"],
# #     perf_key="E_AUROC",
# #     target_dim_col="target_dim",
# # )
# for target_dim in res_df["target_dim"].unique():
#     temp_df = res_df[res_df["target_dim"] == target_dim]
#     mean_df = temp_df.groupby(fixed_factors).mean().reset_index()
#
#     # VIEW 1
#     fig, axes = plt.subplots(1, 2)
#     vmin, vmax = (
#         mean_df["E_" + metric_key_word].min(),
#         mean_df["E_" + metric_key_word].max(),
#     )
#     for ax, skip in zip(axes, [False, True]):
#         pivot_df = mean_df[mean_df["skip"] == skip].pivot(
#             "n_dense_layers", "latent_factor", "E_" + metric_key_word
#         )
#         sns.heatmap(
#             pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax, vmin=vmin, vmax=vmax
#         )
#         ax.set_title(str(skip))
#
#     # VIEW 2: SKIP VS N LAYERS
#     all_lines = []
#     fig, axes = plt.subplots(1, 2, sharey=True)
#     for i, latent_factor in enumerate([0.1, 10]):
#         ax = axes[i]
#         for skip in [False, True]:
#             x_axis = "n_dense_layers"
#             mean_df_ = (
#                 mean_df[
#                     (mean_df["skip"] == skip)
#                     & (mean_df["latent_factor"] == latent_factor)
#                 ]
#                 .groupby([x_axis, "skip"])
#                 .mean()
#                 .reset_index()
#             )
#             (line,) = ax.plot(mean_df_[x_axis], mean_df_["E_" + metric_key_word])
#             all_lines.append(line)
#             # .heatmap(pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax)
#
#         plt.legend(all_lines, [False, True])
#
#     # VIEW 3: SKIP VS N EPOCHS
#     temp_df = apply_optim_df(
#         res_df,
#         fixed_params=["current_epoch", "latent_factor", "skip"],
#         optim_params=["n_dense_layers"],
#         perf_key="E_AUROC",
#         target_dim_col="target_dim",
#     )
#     mean_df = (
#         temp_df[temp_df["target_dim"] == target_dim]
#         .groupby(["current_epoch", "latent_factor", "skip"])
#         .mean()
#         .reset_index()
#     )
#     all_lines = []
#     fig, axes = plt.subplots(1, 2, sharey=True)
#     for i, latent_factor in enumerate([0.1, 10]):
#         ax = axes[i]
#         for skip in [False, True]:
#             x_axis = "current_epoch"
#             mean_df_ = (
#                 mean_df[
#                     (mean_df["skip"] == skip)
#                     & (mean_df["latent_factor"] == latent_factor)
#                 ]
#                 .groupby([x_axis, "skip"])
#                 .mean()
#                 .reset_index()
#             )
#             (line,) = ax.plot(mean_df_[x_axis], mean_df_["E_" + metric_key_word])
#             all_lines.append(line)
#             # .heatmap(pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax)
#
#         plt.legend(all_lines, [False, True])

# # VIEW 2
# all_lines = []
# fig, ax = plt.subplots(1, 1)
# for skip in [False, True]:
#     x_axis = "n_dense_layers"
#     mean_df_ = (
#         mean_df[mean_df["skip"] == skip].groupby([x_axis, "skip"]).mean().reset_index()
#     )
#     (line,) = ax.plot(mean_df_[x_axis], mean_df_["E_" + metric_key_word])
#     all_lines.append(line)
#     # .heatmap(pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax)
#
# plt.legend(all_lines, [False, True])

# ====================NUMBER OF CONV VS DENSE LAYERS======================
res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW6AUROC.csv")
metric_key_word = "AUROC"
fixed_factors = ["n_dense_layers", "n_conv_layers", "latent_factor"]
res_df = apply_optim_df(
    res_df,
    fixed_params=fixed_factors,
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col="target_dim",
)
for target_dim in res_df["target_dim"].unique():
    temp_df = res_df[res_df["target_dim"] == target_dim]
    mean_df = temp_df.groupby(fixed_factors).mean().reset_index()

    # VIEW 1
    fig, axes = plt.subplots(1, 2)
    vmin, vmax = (
        mean_df["E_" + metric_key_word].min(),
        mean_df["E_" + metric_key_word].max(),
    )
    sup_y_col = "latent_factor"
    for ax, skip in zip(axes, mean_df[sup_y_col].unique()):
        pivot_df = mean_df[mean_df[sup_y_col] == skip].pivot(
            "n_conv_layers", "n_dense_layers", "E_" + metric_key_word
        )
        sns.heatmap(
            pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax, vmin=vmin, vmax=vmax
        )
        ax.set_title(str(skip))


# ======================================================================

# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_long_AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_TEST_long3AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_SKIP2AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW2AUROC.csv")
raw_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW7AUROC.csv")
metric_key_word = "AUROC"
fixed_factors = ["n_conv_layers", "latent_factor", "skip"]

# for target_dim in res_df["target_dim"].unique():

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
# btneck_labels = ["A", "B", "C", "D"]  # label for each condition
btneck_labels = [
    "A (no btneck)",
    "B (overparam w/o skip)",
    "C (underparam w skip)",
    "D (overparam w skip)",
]  # label for each condition
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

optim_df = apply_optim_df(
    raw_df,
    fixed_params=["n_conv_layers", "BTNECK_TYPE"],
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col="target_dim",
)


for target_dim in optim_df["target_dim"].unique():
    all_lines = []
    fig, ax = plt.subplots(1, 1)
    for btneck_type in btneck_labels:
        temp_df = optim_df[
            (optim_df["BTNECK_TYPE"] == btneck_type)
            & (optim_df["target_dim"] == target_dim)
        ]
        (line,) = ax.plot(temp_df["n_conv_layers"], temp_df["E_" + metric_key_word])
        all_lines.append(line)
    ax.legend(all_lines, btneck_labels)


# ===============N LAYERS VS CAPACITY======================
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW8AUROC.csv")
res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW9AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_DUMMYAUROC.csv")
metric_key_word = "AUROC"
fixed_factors = ["n_conv_layers", "n_enc_capacity"]
res_df = apply_optim_df(
    res_df,
    fixed_params=fixed_factors,
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col="target_dim",
)
for target_dim in res_df["target_dim"].unique():
    temp_df = res_df[res_df["target_dim"] == target_dim]
    mean_df = temp_df.groupby(fixed_factors).mean().reset_index()

    # VIEW 1
    fig, ax = plt.subplots(1, 1)
    vmin, vmax = (
        mean_df["E_" + metric_key_word].min(),
        mean_df["E_" + metric_key_word].max(),
    )
    # sup_y_col = "latent_factor"
    # for ax, skip in zip(axes, mean_df[sup_y_col].unique()):
    pivot_df = mean_df.pivot("n_conv_layers", "n_enc_capacity", "E_" + metric_key_word)
    sns.heatmap(
        pivot_df, cmap="viridis", annot=True, fmt=".3f", ax=ax, vmin=vmin, vmax=vmax
    )
    # ax.set_title(str(skip))

# ================EFFECT OF OVERPARAM========================
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_NEW8AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_OVERPARAMAUROC.csv")
res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_OVERPARAM2AUROC.csv")
# res_df = pd.read_csv("../experiments/ZEMA_EFFECTIVE_CAP_DUMMYAUROC.csv")

fixed_factors = ["n_enc_capacity", "latent_factor"]
mean_df = res_df.groupby(fixed_factors).mean().reset_index()
pivot_df = mean_df.pivot(*fixed_factors, "E_" + metric_key_word)

# VIEW 2
# cat = "n_enc_capacity"
# x_axis = "latent_factor"
cat = "latent_factor"
x_axis = "n_enc_capacity"
all_lines = []
fig, ax = plt.subplots(1, 1)
for enc_cap in mean_df[cat].unique():
    temp = mean_df[mean_df[cat] == enc_cap]
    x_length = np.arange(len(temp[x_axis].unique()))
    (line,) = ax.plot(x_length, temp["E_" + metric_key_word])
    ax.set_xticks(x_length)
    ax.set_xticklabels(temp[x_axis].unique())
    all_lines.append(line)
plt.legend(all_lines, mean_df[cat].unique())

# VIEW 3 - AGG
for target_dim in res_df["target_dim"].unique():
    temp_res = res_df[res_df["target_dim"] == target_dim]
    mean_df = temp_res.groupby(fixed_factors).mean().reset_index()
    cat = "latent_factor"
    x_axis = "n_enc_capacity"
    all_lines = []
    fig, ax = plt.subplots(1, 1)
    for enc_cap in mean_df[cat].unique():
        temp = mean_df[mean_df[cat] == enc_cap]
        x_length = np.arange(len(temp[x_axis].unique()))
        (line,) = ax.plot(x_length, temp["E_" + metric_key_word])
        ax.set_xticks(x_length)
        ax.set_xticklabels(temp[x_axis].unique())
        all_lines.append(line)
    plt.legend(all_lines, mean_df[cat].unique())
