import itertools
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import truncnorm, pearsonr, shapiro, norm, anderson
import numpy as np
from statsmodels.distributions import ECDF

from baetorch.baetorch.models_v2.base_layer import flatten_np

warnings.simplefilter(action="ignore", category=FutureWarning)

# ======SPECIFY DATASET=======
dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"


check_row_exists = True
# check_row_exists = False

exp_names = {
    "ZEMA": "ZEMA_INF_FULL_",
    "STRATH": "STRATH_INF_FULL_",
    "ODDS": "ODDS_INF_FULL_",
    "Images": "IMAGES_INF_FULL_",
}
exp_name_prefix = exp_names[dataset]

# ========LOAD DATASET===========

filenames = {
    "ZEMA": "ZEMA_np_data.p",
    "STRATH": "STRATH_np_data.p",
    "Images": "Images_np_data.p",
    "ODDS": "ODDS_np_data.p",
}
dataset_folder = "np_datasets"
pickled_dataset = pickle.load(
    open(os.path.join(dataset_folder, filenames[dataset]), "rb")
)
tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]
# ================================


# =========SPECIFY GRID===========

## FULL GRID
# grid = {
#     "W_std": [1.4, 1.2, 1.0, 0.8],
#     "diag_reg": [1e-5, 1e-4, 1e-3],
#     "norm": ["layer", "none"],
#     "skip": [False],
#     "num_layers": [2, 3, 4, 5],
#     "activation": ["leakyrelu", "gelu", "erf"],
# }

## STANDARD SINGLE TRY
grid = {
    "W_std": [1.2],
    "diag_reg": [1e-5],
    "norm": ["layer"],
    "skip": [False],
    "num_layers": [4],
    "activation": ["leakyrelu"],
}

grid_keys = grid.keys()
grid_list = list(itertools.product(*grid.values()))
# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))

# exp_man = ExperimentManager(folder_name="thesis_experiments/inf_experiments")
# start_exp_time = time.time()
# final_res = []


for target_key, data_list in pickled_dataset.items():
    ## handle images nested structure differently
    if dataset == "Images":
        iterate_list = data_list["train"]
    else:
        iterate_list = data_list

    # N data lists , 1 for each random seed split
    for data_dict in iterate_list:
        random_seed = data_dict["random_seed"]
        x_id_train = data_dict["x_id_train"]

        # unpack for images differently
        if dataset == "Images":
            x_id_train = x_id_train[: len(x_id_train) // 3]
            x_id_test = data_list["x_id_test"]
            x_ood_test = data_list["x_ood_test"]
        else:
            x_id_test = data_dict["x_id_test"]
            x_ood_test = data_dict["x_ood_test"]

# plt.figure()
# sm.qqplot(x_id_train[0].reshape(-1), line="45")
# sm.qqplot(x_ood_test.reshape(-1), line="45")
# sm.qqplot(df.data, line="45")

# # dt_samples = x_id_train[:, 0].reshape(-1)
# dt_samples = x_id_train[:, 0, 5]
# # dt_samples = x_id_train.reshape(-1)
#
# plt.figure()
# plt.hist(dt_samples, density=True)
#
# # ============================================
# xa = np.min(dt_samples)
# xb = np.max(dt_samples)
#
# x = np.linspace(xa, xb, 10000)
# par = truncnorm.fit(dt_samples)
# # par = truncnorm.fit(dt_samples, a=0, b=1)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, truncnorm.pdf(x, *par), "b-", lw=1, alpha=0.6, label="truncnorm fit")
# ax.hist(dt_samples, density=True, histtype="stepfilled", alpha=0.3)
#
#
# # plt.figure()
# # sm.qqplot(dt_samples, line="45")
#
# # ============================================

# mean, std = norm.fit(dt_samples)
#
# plt.figure()
# # plt.hist(dt_samples, bins=30, density=True)
# plt.hist(dt_samples, density=True)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# y_1 = norm.pdf(x, mean, 1)
# y_homo = norm.pdf(x, mean, std)
#
# plt.plot(x, y_1)
# plt.plot(x, y_homo)
# plt.show()

# ===========================
target_dim = 2
x_id_train = pickled_dataset[target_dim][0]["x_id_train"]


i_feature = 0
num_samples = x_id_train.shape[0]
dt_samples = flatten_np(x_id_train)
dt_samples = dt_samples[:, i_feature]
# dt_samples = x_id_train.reshape(num_samples, -1)

mean, std = norm.fit(dt_samples)
ecdf_f = ECDF(dt_samples)
x = np.linspace(np.min(dt_samples), np.max(dt_samples))
y_ecdf = ecdf_f(x)
y_cdf = norm.cdf(x, mean, std)
y_cdf_1 = norm.cdf(x, mean, 1)

calibration_error = np.mean(np.abs(y_ecdf - y_cdf))
print("CALIB ERROR: " + str(calibration_error))

# calculate ECE for all
# target_dim = "thyroid"
# target_dim = "pendigits"
# target_dim = "optdigits"
# target_dim = "pima"
# target_dim = "vowels"
# target_dim = "ionosphere"
# target_dim = 2

# x_id_train = pickled_dataset[target_dim][0]["x_id_train"]
# x_id_train = pickled_dataset[target_dim][0]["x_id_test"]
# x_id_train = pickled_dataset[target_dim][0]["x_ood_test"]


def get_calibration_stats(x_data):
    total_features = np.product(x_data.shape[1:])
    flattened_x_id = flatten_np(x_data)
    ece_features = []
    p_corrs = []
    shapiro_vals = []
    for i_feature in range(total_features):
        dt_samples = flattened_x_id[:, i_feature]

        mean, std = norm.fit(dt_samples)
        ecdf_f = ECDF(dt_samples)
        y_ecdf = ecdf_f(dt_samples)
        y_cdf = norm.cdf(dt_samples, mean, std)
        calibration_error = np.mean(np.abs(y_ecdf - y_cdf))
        p_corr = pearsonr(y_cdf, y_ecdf)[0]
        shapiro_val = shapiro(dt_samples)
        # shapiro_val = anderson(dt_samples)
        shapiro_vals.append(shapiro_val)
        if not np.isnan(calibration_error) and not np.isnan(p_corr):
            ece_features.append(calibration_error)
            p_corrs.append(p_corr)
    ece_features = np.array(ece_features)
    p_corrs = np.array(p_corrs)
    high_corr_perc = len(np.argwhere(np.array(p_corrs) > 0.9)) / total_features

    return {
        "ece": ece_features,
        "p_corr": p_corrs,
        "high_corr": high_corr_perc,
        "shapiro": shapiro_vals,
    }


# calculate ECE for all
# target_dim = "thyroid"
# target_dim = "pendigits"
# target_dim = "optdigits"
# target_dim = "pima"
# target_dim = "vowels"
# target_dim = "ionosphere"
target_dim = 1

all_ece_diff = []
all_pcorr_diff = []
num_splits = len(pickled_dataset[target_dim])
for split_i in range(num_splits):
    # id_stats = get_calibration_stats(pickled_dataset[target_dim][split_i]["x_id_train"])
    id_stats = get_calibration_stats(
        np.concatenate(
            (
                pickled_dataset[target_dim][split_i]["x_id_train"],
                pickled_dataset[target_dim][split_i]["x_id_test"],
            )
        )
    )
    ood_stats = get_calibration_stats(
        pickled_dataset[target_dim][split_i]["x_ood_test"]
    )

    ece_diffs = id_stats["ece"] - ood_stats["ece"]
    pcorr_diffs = id_stats["p_corr"] - ood_stats["p_corr"]

    ece_diff_mean = np.mean(np.abs(ece_diffs))
    pcorr_diff_mean = np.mean(np.abs(pcorr_diffs))

    # ece_diff_mean = np.mean(ece_diffs)
    # pcorr_diff_mean = np.mean(pcorr_diffs)

    # print("CALIB ERROR: " + str(ece_diff_mean))
    # print("MEAN CORR: " + str(pcorr_diff_mean))
    all_ece_diff.append(ece_diff_mean)
    all_pcorr_diff.append(pcorr_diff_mean)

all_ece_diff = np.array(all_ece_diff)
all_pcorr_diff = np.array(all_pcorr_diff)

print("CALIB ERROR: " + str(np.mean(all_ece_diff) * 100))
print("MEAN CORR: " + str(np.mean(all_pcorr_diff)))

# =======================
# flatten_x_id = flatten_np(x_id_train)
#
# shapiro_test = shapiro(pickled_dataset[target_dim][split_i]["x_id_train"][:, 0, 0])

# target_dim = "thyroid"
# target_dim = "pendigits"
# target_dim = "optdigits"
# target_dim = "pima"
# target_dim = "vowels"
# target_dim = "ionosphere"
target_dim = 3

all_shapiros = {"id": [], "ood": []}

num_splits = len(pickled_dataset[target_dim])
for split_i in range(num_splits):
    id_stats = get_calibration_stats(
        np.concatenate(
            (
                pickled_dataset[target_dim][split_i]["x_id_train"],
                pickled_dataset[target_dim][split_i]["x_id_test"],
            )
        )
    )
    ood_stats = get_calibration_stats(
        pickled_dataset[target_dim][split_i]["x_ood_test"]
    )
    all_shapiros["id"].append(np.copy(id_stats["shapiro"]))
    all_shapiros["ood"].append(np.copy(ood_stats["shapiro"]))

shapiros_id = np.array(all_shapiros["id"]).mean(0)[:, 0]
shapiros_ood = np.array(all_shapiros["ood"]).mean(0)[:, 0]

div_ = np.nanmean(shapiros_ood / shapiros_id)

print(np.nanmean(shapiros_id))
print(np.nanmean(shapiros_ood))
print(np.nanmean(div_))

# =======================


# total_features = np.product(x_id_train.shape[1:])
# flattened_x_id = flatten_np(x_id_train)
# ece_features = []
# p_corrs = []
# for i_feature in range(total_features):
#     dt_samples = flattened_x_id[:, i_feature]
#
#     mean, std = norm.fit(dt_samples)
#     ecdf_f = ECDF(dt_samples)
#     y_ecdf = ecdf_f(dt_samples)
#     y_cdf = norm.cdf(dt_samples, mean, std)
#     calibration_error = np.mean(np.abs(y_ecdf - y_cdf))
#     p_corr = pearsonr(y_cdf, y_ecdf)[0]
#     if not np.isnan(calibration_error):
#         ece_features.append(calibration_error)
#         p_corrs.append(p_corr)
# ece_features = np.array(ece_features)
# p_corrs = np.array(p_corrs)
# # mean_ece = np.mean(ece_features)
# # mean_corr = np.mean(p_corrs)
# # mean_ece = np.mean(ece_features)
# # mean_corr = np.mean(p_corrs)
# high_corr_perc = len(np.argwhere(np.array(p_corrs) > 0.9)) / total_features
#

# print("CALIB ERROR: " + str(mean_ece))
# print("MEAN CORR: " + str(mean_corr))
# print(len(np.argwhere(np.array(p_corrs) > 0.9)) / total_features)


# plt.figure()
# plt.plot(y_ecdf, y_ecdf)
# plt.scatter(y_ecdf, y_cdf)


# plt.figure()
# plt.plot(x, y_ecdf)
# plt.plot(x, y_cdf)
# plt.plot(x, y_cdf_1)


# plt.figure()
# plt.scatter(y_cdf, y_ecdf)
# plt.plot(y_cdf, y_cdf)
# plt.plot(y_cdf, y_cdf_1)

plt.figure()
plt.plot(y_ecdf, y_ecdf)
plt.scatter(y_ecdf, y_cdf)
plt.scatter(y_ecdf, y_cdf_1)
