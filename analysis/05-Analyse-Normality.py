# Plot PP Plot and calculate their mean abs error
# To show that the residuals are of Gaussian
# Models that performed better with the Heteroscedestic Gaussian

import pickle
import matplotlib

# matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm, sem, shapiro, iqr, ks_2samp, multivariate_normal
from statsmodels.distributions import ECDF
import seaborn as sns
from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.models_v2.base_layer import flatten_np
import numpy as np
import statsmodels.api as sm
from thesis_experiments.util_analyse import (
    get_pp_stats,
    get_random_args,
    get_mae_stats,
    get_top_whisker,
    get_low_high_quantile,
)
from pingouin import multivariate_normality

# dataset = "ZEMA"
# dataset = "STRATH"
dataset = "ODDS"
# dataset = "Images"

# y_pred_dict = pickle.load(open("recons/" + dataset + "_20220426.p", "rb"))
# y_pred_dict = pickle.load(open("recons/ZEMA_20220426.p", "rb"))
# y_pred_dict = pickle.load(open("recons/ODDS_20220426.p", "rb"))
y_pred_dict = pickle.load(open("recons/ZEMA_STANDARDISE_20220426.p", "rb"))

# y_pred_dict = pickle.load(open("recons/ODDS_Y_RECON_XTEST.p", "rb"))
# y_pred_dict = pickle.load(open("recons/ODDS_Y_RECON_XTRAIN.p", "rb"))
# y_pred_dict = pickle.load(open("recons/CIFAR_Y_RECON_XTEST_20220425.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_SVHN_XTEST_20220425.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_FMNIST_XTEST_20220425.p", "rb"))


all_mae_dfs = {}
# ====FLATTEN ERR=====
for target_dim in y_pred_dict.keys():

    x_id = y_pred_dict[target_dim]["x_id"]
    x_ood = y_pred_dict[target_dim]["x_ood"]
    y_id = y_pred_dict[target_dim]["y_id"]
    y_ood = y_pred_dict[target_dim]["y_ood"]
    bae_noise = y_pred_dict[target_dim]["bae_gauss_noise"]

    # collect random seeds
    total_random_seeds = len(y_pred_dict[target_dim]["x_id"])

    # calculate results for each random seed
    res_id_seeds = []
    res_ood_seeds = []

    for random_seed_i in range(total_random_seeds):
        # flatten_x and y
        flatten_x_id = flatten_np(y_pred_dict[target_dim]["x_id"][random_seed_i])
        flatten_x_ood = flatten_np(y_pred_dict[target_dim]["x_ood"][random_seed_i])
        flatten_y_id = flatten_np(
            y_pred_dict[target_dim]["y_id"][random_seed_i].mean(0)
        )
        flatten_y_ood = flatten_np(
            y_pred_dict[target_dim]["y_ood"][random_seed_i].mean(0)
        )

        # calculate residuals (true - predicted)
        feature_err_id = flatten_x_id - flatten_y_id
        feature_err_ood = flatten_x_ood - flatten_y_ood

        # total features
        total_features = feature_err_id.shape[-1]

        # get MAE of PP plots
        res_id = [
            get_mae_stats(feature_err_id[:, feature_i])["mae"]
            for feature_i in range(total_features)
        ]
        res_ood = [
            get_mae_stats(feature_err_ood[:, feature_i])["mae"]
            for feature_i in range(total_features)
        ]

        # collect results for each rd seed
        res_id_seeds.append(res_id)
        res_ood_seeds.append(res_ood)
    all_mae_dfs.update({target_dim: {"id": res_id_seeds, "ood": res_ood_seeds}})

    # =====GET RESULTS====
    round_deci = 2
    mae_id = np.array(all_mae_dfs[target_dim]["id"])
    mae_ood = np.array(all_mae_dfs[target_dim]["ood"])

    low_q, high_q = 25, 75
    # low_q, high_q = 5, 95
    mae_lohi_id = np.array(
        [get_low_high_quantile(mae, high_q=high_q, low_q=low_q) for mae in mae_id]
    )
    mae_lohi_ood = np.array(
        [get_low_high_quantile(mae, high_q=high_q, low_q=low_q) for mae in mae_ood]
    )

    mae_id_max = mae_lohi_id[:, 1]
    mae_ood_max = mae_lohi_ood[:, 1]
    # mae_id_max = np.percentile(mae_id, 95, axis=1)
    # mae_ood_max = np.percentile(mae_ood, 95, axis=1)

    mae_id_max_mean = (np.mean(mae_id_max)).round(round_deci)
    mae_id_max_sem = (sem(mae_id_max) * 1).round(round_deci)
    mae_ood_max_mean = (np.mean(mae_ood_max)).round(round_deci)
    mae_ood_max_sem = (sem(mae_ood_max) * 1).round(round_deci)

    print("TARGET DIM:" + str(target_dim))
    print("MAX-MAE ID:" + str(mae_id_max_mean) + "+-" + str(mae_id_max_sem))
    print("MAX-MAE OOD:" + str(mae_ood_max_mean) + "+-" + str(mae_ood_max_sem))

    # ====== plot ====
    plt.figure()
    plt.boxplot([mae_id.mean(0), mae_ood.mean(0)])
    # plt.boxplot([mae_id.reshape(-1), mae_ood.reshape(-1)])
    # plt.boxplot([mae_id.reshape(-1), mae_ood.reshape(-1)], showfliers=False)
    plt.title(str(target_dim))

    # plt.figure()
    # plt.hist(mae_id.mean(0), density=True)
    # plt.hist(mae_ood.mean(0), density=True)
    # plt.title(str(target_dim))

    mae_id_mean = mae_id.mean(0)
    mae_ood_mean = mae_ood.mean(0)

    low_q, high_q = 5, 95
    # low_q, high_q = 25, 75
    mae_low_id, mae_high_id = get_low_high_quantile(
        mae_id_mean, high_q=high_q, low_q=low_q, axis=0
    )
    mae_low_ood, mae_high_ood = get_low_high_quantile(
        mae_ood_mean, high_q=high_q, low_q=low_q, axis=0
    )

    plt.figure()
    plt.barh(
        y=1,
        width=mae_high_id,
        left=mae_low_id,
    )
    plt.barh(
        y=0,
        width=mae_high_ood,
        left=mae_low_ood,
    )
    plt.title(str(target_dim))

    # =========BAE NOISE=========
    bae_noise_mean = bae_noise.mean(0).mean(0)
    bae_noise_mean = bae_noise[0].mean(0)

    # # plt.figure()
    # # plt.plot(bae_noise_mean)
    # #
    # # plt.figure()
    # # plt.plot(feature_err_id.mean(0))
    # # plt.plot(feature_err_ood.mean(0))
    #
    # var_id = flatten_np(x_id.mean(0)).var(0)
    # var_ood = flatten_np(x_ood.mean(0)).var(0)
    #
    # rescale_id = (var_id - np.min(var_id)) / (np.max(var_id) - np.min(var_id))
    # rescale_ood = (var_ood - np.min(var_ood)) / (np.max(var_ood) - np.min(var_ood))
    #
    # plt.figure()
    # plt.plot(rescale_id)
    # plt.plot(rescale_ood)
    #
    # # plt.figure()
    # # plt.plot(flatten_np(x_id.mean(0)).var(0))
    # # plt.plot(flatten_np(x_ood.mean(0)).var(0))

    # plt.figure()
    # plt.scatter(x_id[0].var(0), bae_noise_mean)
    # ============================

    # x_id = y_pred_dict[target_dim]["x_id"]
    # x_ood = y_pred_dict[target_dim]["x_ood"]
    # y_id = y_pred_dict[target_dim]["y_id"]
    # y_ood = y_pred_dict[target_dim]["y_ood"]
    nll_aurocs = []
    mse_aurocs = []
    for random_seed_i in range(total_random_seeds):
        # random_seed_i = -1
        mse_id = (x_id[random_seed_i] - y_id[random_seed_i].mean(0)) ** 2
        mse_ood = (x_ood[random_seed_i] - y_ood[random_seed_i].mean(0)) ** 2
        mse_auroc = calc_auroc(
            flatten_np(mse_id).mean(-1), flatten_np(mse_ood).mean(-1)
        )

        # sigma = 0.005 - bae_noise[random_seed_i].mean(0)
        sigma = bae_noise[random_seed_i].mean(0) * 10
        nll_id = (flatten_np(mse_id) / (2 * sigma)) + np.log(sigma) / 2
        nll_ood = (flatten_np(mse_ood) / (2 * sigma)) + np.log(sigma) / 2
        nll_auroc = calc_auroc(
            flatten_np(nll_id).mean(-1), flatten_np(nll_ood).mean(-1)
        )

        #
        # custom_sigma = np.abs(
        #     flatten_np(mse_id).mean(0) - flatten_np(mse_ood).mean(0)
        # )  # how to make terbalik?
        # custom_sigma = (np.max(custom_sigma) - custom_sigma) + 1e-6
        # # custom_sigma = (custom_sigma - np.min(custom_sigma)) / (
        # #     np.max(custom_sigma) - np.min(custom_sigma)
        # # ) + 1e-6
        #
        # # sigma = flatten_np(x_id[random_seed_i]).var(0) + 1e-11
        # # sigma = custom_sigma
        # nll_id = (flatten_np(mse_id) / (2 * sigma)) + np.log(sigma) / 2
        # nll_ood = (flatten_np(mse_ood) / (2 * sigma)) + np.log(sigma) / 2
        # nll_auroc = calc_auroc(
        #     flatten_np(nll_id).mean(-1), flatten_np(nll_ood).mean(-1)
        # )

        # sigma using x_id?
        # sigma = flatten_np(x_id[random_seed_i]).var(0)
        sigma = bae_noise[random_seed_i].mean(0)
        nll_id = (flatten_np(mse_id) / (2 * sigma)) + np.log(sigma) / 2

        mse_aurocs.append(mse_auroc)
        nll_aurocs.append(nll_auroc)
    print("TGT DIM:" + str(target_dim))
    print("MSE:" + str(np.mean(mse_aurocs)))
    print("NLL:" + str(np.mean(nll_aurocs)))

    # plt.figure()
    # plt.plot(mae_id_mean)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(flatten_np(mse_id).mean(0))
    ax1.plot(flatten_np(mse_ood).mean(0))
    ax2.plot(bae_noise.mean(0).mean(0))
    # ax2.plot(custom_sigma)
    # ax2.plot(mae_id_mean)
    # ax2.plot(mae_ood_mean)

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(np.abs(nll_id.mean(0) - nll_ood.mean(0)))
    # ax2.plot(bae_noise.mean(0).mean(0))

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(np.abs(flatten_np(mse_id).mean(0) - flatten_np(mse_ood).mean(0)))
    # ax2.plot(bae_noise.mean(0).mean(0))
    #
    # custom_sigma = np.abs(nll_id.mean(0) - nll_ood.mean(0))  # how to make terbalik?
    # custom_sigma = (np.max(custom_sigma) - custom_sigma) + 1e-6
    # custom_sigma = (custom_sigma - np.min(custom_sigma)) / (
    #     np.max(custom_sigma) - np.min(custom_sigma)
    # ) + 1e-6
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(np.abs(nll_id.mean(0) - nll_ood.mean(0)))
    # ax2.plot(custom_sigma)

    # ============================
