# Plot PP Plot and calculate their mean abs error
# To show that the residuals are of Gaussian
# Models that performed better with the Heteroscedestic Gaussian

import pickle
import matplotlib

# matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm, sem, shapiro, iqr
from statsmodels.distributions import ECDF
import seaborn as sns
from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.models_v2.base_layer import flatten_np
import numpy as np
import statsmodels.api as sm
from thesis_experiments.util_analyse import get_pp_stats, get_random_args

dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"

y_pred_dict = pickle.load(open("recons/" + dataset + "_Y_RECON.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_Y_RECON_XTEST.p", "rb"))
# y_pred_dict = pickle.load(open("recons/ODDS_Y_RECON_XTRAIN_MSE.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_Y_RECON_XTRAIN_CIFAR.p", "rb"))

# y_pred_dict = pickle.load(open("recons/ODDS_Y_RECON_XTEST.p", "rb"))
# y_pred_dict = pickle.load(open("recons/ODDS_Y_RECON_XTRAIN.p", "rb"))
# y_pred_dict = pickle.load(open("recons/CIFAR_Y_RECON_XTEST_20220425.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_SVHN_XTEST_20220425.p", "rb"))
# y_pred_dict = pickle.load(open("recons/Images_FMNIST_XTEST_20220425.p", "rb"))


all_mae_dfs = []
# ====FLATTEN ERR=====
for target_dim in y_pred_dict.keys():
    x_id = y_pred_dict[target_dim]["x_id"]
    x_ood = y_pred_dict[target_dim]["x_ood"]
    y_id = y_pred_dict[target_dim]["y_id"]
    y_ood = y_pred_dict[target_dim]["y_ood"]
    bae_noise = y_pred_dict[target_dim]["bae_gauss_noise"]

    # for each random seed
    sample_mae_id = []
    sample_mae_ood = []
    plot_samples_id = []
    plot_samples_ood = []
    all_res_samples = []
    for random_seed_i in range(y_id.shape[0]):
        err_ids = []
        err_oods = []

        # =============GET MEAN OF ENSEMBLE PREDICTION============
        err_id_i = y_id[random_seed_i].mean(0) - x_id[random_seed_i]
        err_ood_i = y_ood[random_seed_i].mean(0) - x_ood[random_seed_i]

        serr_id = (y_id[random_seed_i].mean(0) - x_id[random_seed_i]) ** 2
        serr_ood = (y_ood[random_seed_i].mean(0) - x_ood[random_seed_i]) ** 2

        # analyse raw inputs instead of residuals?..
        # err_id_i = x_id[random_seed_i]
        # err_ood_i = x_ood[random_seed_i]

        err_ids.append(err_id_i)
        err_oods.append(err_ood_i)

        # ========================================================
        err_ids = np.concatenate(err_ids)
        err_oods = np.concatenate(err_oods)

        flatten_err_id = flatten_np(err_ids)
        flatten_err_ood = flatten_np(err_oods)
        total_features = flatten_err_id.shape[-1]

        # EXP: Calc AUROC as feature importance map
        flatten_serr_id = flatten_np(serr_id)
        flatten_serr_ood = flatten_np(serr_ood)
        feature_imp = np.array(
            [
                calc_auroc(
                    np.abs(flatten_serr_id[:, feature_i]),
                    np.abs(flatten_serr_ood[:, feature_i]),
                )
                for feature_i in range(total_features)
            ]
        )

        # collect samples for plotting
        plot_samples_id.append(flatten_err_id)
        plot_samples_ood.append(flatten_err_ood)

        # calculate results of all
        all_res = []
        for feature_i in range(total_features):
            feature_err_id_i = flatten_err_id[:, feature_i]
            feature_err_ood_i = flatten_err_ood[:, feature_i]
            res_ = get_pp_stats(feature_err_id_i, feature_err_ood_i)
            all_res.append(res_)
        all_res_samples.append(all_res)

        # get mae for each feature
        mae_feat_id = np.array([feature["id"]["mae"] for feature in all_res])
        mae_feat_ood = np.array([feature["ood"]["mae"] for feature in all_res])

        # Experimental
        shapiro_feat_id = np.array([feature["id"]["shapiro"] for feature in all_res])
        shapiro_feat_ood = np.array([feature["ood"]["shapiro"] for feature in all_res])
        weighted_mae_id = mae_feat_id * feature_imp
        weighted_mae_ood = mae_feat_ood * feature_imp
        maeXpcorr_feat_id = np.array(
            [feature["id"]["maeXpcorr"] for feature in all_res]
        )
        maeXpcorr_feat_ood = np.array(
            [feature["ood"]["maeXpcorr"] for feature in all_res]
        )
        var_feat_id = np.array([feature["id"]["var"] for feature in all_res])
        var_feat_ood = np.array([feature["ood"]["var"] for feature in all_res])

        perc_higher_mae = (
            len(
                np.argwhere(((mae_feat_id - mae_feat_ood) <= -3) & (mae_feat_ood >= 5))[
                    :, 0
                ]
            )
            / len(mae_feat_ood)
            * 100
        )

        # collect mae for each random seed
        sample_mae_id.append(mae_feat_id)
        sample_mae_ood.append(mae_feat_ood)
    print("SHAPIRO ID:" + str(shapiro_feat_id.mean()))
    print("SHAPIRO OOD:" + str(shapiro_feat_ood.mean()))

    print("NOT WEIGHTED MAE ID:" + str(mae_feat_id.mean()))
    print("NOT WEIGHTED MAE OOD:" + str(mae_feat_ood.mean()))

    print("WEIGHTED MAE ID:" + str(maeXpcorr_feat_id.mean()))
    print("WEIGHTED MAE OOD:" + str(maeXpcorr_feat_ood.mean()))

    print("VAR ID:" + str(var_feat_id.mean()))
    print("VAR OOD:" + str(var_feat_ood.mean()))

    print("MAE X PCORR ID:" + str(weighted_mae_id.mean()))
    print("MAE X PCORR OOD:" + str(weighted_mae_ood.mean()))

    print("HIGH PERC FEATS MAE:" + str(perc_higher_mae))
    print(mae_feat_id - mae_feat_ood)
    # convert into np array
    sample_mae_id = np.array(sample_mae_id)
    sample_mae_ood = np.array(sample_mae_ood)

    # 1. Compute MAE-Mean over features
    # 2. Calculate MAE statistics Mean+-SEM
    mae_id_mean = np.mean(sample_mae_id.mean(1))
    mae_id_sem = sem(sample_mae_id.mean(1))
    mae_ood_mean = np.mean(sample_mae_ood.mean(1))
    mae_ood_sem = sem(sample_mae_ood.mean(1))

    # PRINT RESULTS
    round_deci = 2
    print("TARGET DIM: " + str(target_dim))
    print(
        "MAE ID:"
        + str(mae_id_mean.round(round_deci))
        + "+-"
        + str(mae_id_sem.round(round_deci))
    )
    print(
        "MAE OOD:"
        + str(mae_ood_mean.round(round_deci))
        + "+-"
        + str(mae_ood_sem.round(round_deci))
    )

    # =========PREPARE FOR PLOTTING============
    # Mean over random seeds to get MAE mean for each feature
    mae_feat_id_mean = sample_mae_id.mean(0)
    mae_feat_ood_mean = sample_mae_ood.mean(0)

    # # Plot the most distinctive difference
    # found_feature = False
    # u_bound = 90
    # l_bound = 10
    # # u_bound = mae_ood_mean
    # # l_bound = mae_id_mean
    # while not found_feature:
    #     u_bound -= 1
    #     l_bound += 1
    #     # u_bound -= 0.1
    #     # l_bound += 0.1
    #     feature_plot_i = np.argwhere(
    #         (mae_feat_ood_mean >= np.percentile(mae_feat_ood_mean, u_bound))
    #         & (mae_feat_id_mean <= np.percentile(mae_feat_id_mean, l_bound))
    #     )
    #     # feature_plot_i = np.argwhere(
    #     #     (mae_feat_ood_mean >= u_bound) & (mae_feat_id_mean <= l_bound)
    #     # )
    #     if len(feature_plot_i) > 0:
    #         feature_plot_i = feature_plot_i.flatten()[-1]
    #         found_feature = True
    #         print("Chosen feature number:" + str(feature_plot_i))
    #     if u_bound <= 50 or l_bound >= 50:
    #         raise ValueError("Can't find best feature to plot")

    # more_than = np.argwhere(mae_feat_ood_mean >= mae_feat_id_mean)[:, 0]
    # less_than = np.argwhere(mae_feat_ood_mean < mae_feat_id_mean)[:, 0]
    # feature_list = less_than if target_dim == 1 else more_than
    # # feature_plot_i =0
    #
    # for feature_plot_i in feature_list:
    feature_map = {"ZEMA": {0: 11, 1: 38, 2: 39, 3: 23}, "STRATH": {2: 84}}
    if dataset in feature_map.keys():
        feature_plot_i = feature_map[dataset][target_dim]
    else:
        feature_plot_i = 1
    print("Chosen feature number:" + str(feature_plot_i))

    plot_single_sample = True

    # =============PP PLOT=====================
    diagonal_line = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(1, 1)
    ax.plot(diagonal_line, diagonal_line, "--", color="black")
    color_id = "tab:blue"
    color_ood = "tab:red"
    print("MAE PLOT ID:" + str(mae_feat_id_mean[feature_plot_i]))
    print("MAE PLOT OOD:" + str(mae_feat_ood_mean[feature_plot_i]))

    # Reduce number of plotting points
    # too many points may lag in plotting
    id_reduce_factor = 5 if dataset == "ZEMA" else 2
    ood_reduce_factor = 5 if dataset == "ZEMA" else 1

    # plot single samples or more
    if plot_single_sample:
        total_samples = 1
    else:
        total_samples = len(all_res_samples)
    for random_seed_i in range(total_samples):
        # Plot on top for each random seed
        empirical_cdf_id = all_res_samples[random_seed_i][feature_plot_i]["id"][
            "empirical"
        ]
        theoretical_cdf_id = all_res_samples[random_seed_i][feature_plot_i]["id"][
            "theoretical"
        ]
        empirical_cdf_ood = all_res_samples[random_seed_i][feature_plot_i]["ood"][
            "empirical"
        ]
        theoretical_cdf_ood = all_res_samples[random_seed_i][feature_plot_i]["ood"][
            "theoretical"
        ]
        id_args = get_random_args(
            empirical_cdf_id, reduce_factor=id_reduce_factor, replace=False
        )
        ood_args = get_random_args(
            empirical_cdf_ood, reduce_factor=ood_reduce_factor, replace=False
        )

        # START plotting
        ax.scatter(
            theoretical_cdf_id[id_args], empirical_cdf_id[id_args], color=color_id
        )
        ax.scatter(
            theoretical_cdf_ood[ood_args],
            empirical_cdf_ood[ood_args],
            color=color_ood,
        )
    ax.set_title(
        str(target_dim)
        + "  MAE-ID:"
        + str(mae_feat_id_mean[feature_plot_i])
        + "  MAE-OOD:"
        + str(mae_feat_ood_mean[feature_plot_i])
    )
    fig.savefig(
        "plots/dummy/"
        + dataset
        + "-"
        + str(target_dim)
        + "-"
        + str(feature_plot_i)
        + ".png"
    )
    # plt.close("all")

    # ===========PLOT RECON SAMPLE===============
    # x_id = y_pred_dict[target_dim]["x_id"]
    # x_ood = y_pred_dict[target_dim]["x_ood"]
    # y_id = y_pred_dict[target_dim]["y_id"]
    # y_ood = y_pred_dict[target_dim]["y_ood"]
    # bae_noise = y_pred_dict[target_dim]["bae_gauss_noise"]
    #
    # # ====FLATTEN ERR=====
    #
    # sample_i = 0
    # bae_sample_i = 0
    # bae_noise_i = bae_noise[sample_i][bae_sample_i]
    # err_id_i = y_id[sample_i][bae_sample_i] - x_id[sample_i]
    # err_ood_i = y_ood[sample_i][bae_sample_i] - x_ood[sample_i]

    # ===========================================
    # be abit more ambitious?.. NOT WORKING
    # interpolate

    # collect all features?
    total_features = len(all_res_samples[0])
    total_seeds = len(all_res_samples)

    seed_range = range(1)
    feature_range = range(total_features)

    empirical_cdf_ids = np.concatenate(
        [
            all_res_samples[random_seed_i][feature_plot_i]["id"]["empirical"]
            for random_seed_i in seed_range
            for feature_plot_i in feature_range
        ]
    )
    theoretical_cdf_ids = np.concatenate(
        [
            all_res_samples[random_seed_i][feature_plot_i]["id"]["theoretical"]
            for random_seed_i in seed_range
            for feature_plot_i in feature_range
        ]
    )

    empirical_cdf_oods = np.concatenate(
        [
            all_res_samples[random_seed_i][feature_plot_i]["ood"]["empirical"]
            for random_seed_i in seed_range
            for feature_plot_i in feature_range
        ]
    )
    theoretical_cdf_oods = np.concatenate(
        [
            all_res_samples[random_seed_i][feature_plot_i]["ood"]["theoretical"]
            for random_seed_i in seed_range
            for feature_plot_i in feature_range
        ]
    )

    x_eval = np.linspace(0, 1, num=100)
    interpolated_ecdf_id = interp1d(
        theoretical_cdf_ids, empirical_cdf_ids, fill_value="extrapolate"
    )(x_eval)
    interpolated_ecdf_ood = interp1d(
        theoretical_cdf_oods, empirical_cdf_oods, fill_value="extrapolate"
    )(x_eval)

    fig, ax = plt.subplots(1, 1)
    ax.plot(diagonal_line, diagonal_line, "--", color="black")

    alpha = 0.35
    ax.scatter(x_eval, interpolated_ecdf_id, alpha=alpha)
    ax.scatter(x_eval, interpolated_ecdf_ood, alpha=alpha)

    # ax.plot(x_eval, interpolated_ecdf_id)
    # ax.plot(x_eval, interpolated_ecdf_ood)

    mae_id = np.nanmean(np.abs(x_eval - interpolated_ecdf_id)) * 100
    mae_ood = np.nanmean(np.abs(x_eval - interpolated_ecdf_ood)) * 100

    print("Interpolated MAE:")
    print(mae_id)
    print(mae_ood)

    fig, ax = plt.subplots(1, 1)
    # plt.boxplot([mae_feat_id, mae_feat_ood])
    # plt.boxplot([sample_mae_id.mean(0), sample_mae_ood.mean(0)])
    # plt.boxplot([sample_mae_id.mean(0), sample_mae_ood.mean(0)])
    # plt.boxplot([sample_mae_id.reshape(-1), sample_mae_ood.reshape(-1)])
    temp_plot_df = pd.DataFrame(
        {"id": sample_mae_id.mean(0), "ood": sample_mae_ood.mean(0)}
    )
    # sns.violinplot(data=temp_plot_df)
    sns.boxplot(data=temp_plot_df, showfliers=False)
    all_mae_dfs.append(temp_plot_df)
    # sns.boxplot(data=temp_plot_df, showfliers=True)
    # bplot = sns.barplot(
    #     data=temp_plot_df,
    #     capsize=0.1,
    #     ax=ax,
    #     errwidth=1.5,
    #     ci=95,
    #     hatch="///",
    # )
    plt.title(str(target_dim))
    # plt.axhline(5.0, linestyle="--", color="black")
    # range of MAEs
    factor = 1.5
    sample_mae_id_mean = sample_mae_id.mean(0)
    sample_mae_ood_mean = sample_mae_ood.mean(0)
    # sample_mae_id_mean = sample_mae_id.reshape(-1)
    # sample_mae_ood_mean = sample_mae_ood.reshape(-1)
    hp = 75
    lp = 25
    try:
        median_id = np.percentile(sample_mae_id_mean, 50).round(2)
        ubound_id = np.percentile(sample_mae_id_mean, hp) + factor * iqr(
            sample_mae_id_mean
        )
        lbound_id = np.percentile(sample_mae_id_mean, lp) - factor * iqr(
            sample_mae_id_mean
        )
        ubound_id = sample_mae_id_mean[sample_mae_id_mean <= ubound_id].max().round(2)
        lbound_id = sample_mae_id_mean[sample_mae_id_mean >= lbound_id].min().round(2)

        median_ood = np.percentile(sample_mae_ood_mean, 50).round(2)
        ubound_ood = np.percentile(sample_mae_ood_mean, hp) + factor * iqr(
            sample_mae_ood_mean
        )
        lbound_ood = np.percentile(sample_mae_ood_mean, lp) - factor * iqr(
            sample_mae_ood_mean
        )
        ubound_ood = (
            sample_mae_ood_mean[sample_mae_ood_mean <= ubound_ood].max().round(2)
        )
        lbound_ood = (
            sample_mae_ood_mean[sample_mae_ood_mean >= lbound_ood].min().round(2)
        )

        # ======DBM / OVS?=======
        dbm = np.abs(median_id - median_ood)
        ovs = np.abs(np.max([ubound_id, ubound_ood]) - np.min([lbound_id, lbound_ood]))
        dbm_ovs = (dbm / ovs * 100).round(2)
        print("DBM-OVS: " + str(dbm_ovs))
        # =======================

        # ubound_id = (sample_mae_id_mean.mean() + sample_mae_id_mean.std() * factor).round(2)
        # lbound_id = (sample_mae_id_mean.mean() - sample_mae_id_mean.std() * factor).round(2)
        #
        # ubound_ood = (
        #     sample_mae_ood_mean.mean() + sample_mae_ood_mean.std() * factor
        # ).round(2)
        # lbound_ood = (
        #     sample_mae_ood_mean.mean() - sample_mae_ood_mean.std() * factor
        # ).round(2)

        print("FEATURE BOUND TEST:" + str(target_dim))
        print("ID: " + str(lbound_id) + "-" + str(median_id) + "-" + str(ubound_id))
        print("OOD: " + str(lbound_ood) + "-" + str(median_ood) + "-" + str(ubound_ood))
        print("=======================")
    except Exception as e:
        print(e)

    # # test hetero?
    # y_id_flat = flatten_np(y_id[random_seed_i].mean(0))
    # x_id_flat = flatten_np(x_id[random_seed_i])
    # y_ood_flat = flatten_np(y_ood[random_seed_i].mean(0))
    # x_ood_flat = flatten_np(x_ood[random_seed_i])
    # total_features = x_id_flat.shape[-1]
    # print("CHG OF VAR:" + str((y_id_flat.std(0) - y_ood_flat.std(0)).mean()))
    # test_id = np.array(
    #     [
    #         het_goldfeldquandt(
    #             y_id_flat[:, feature_i],
    #             x_id_flat,
    #             # np.expand_dims(x_id_flat[:, feature_i], 1),
    #         )
    #         for feature_i in range(total_features)
    #     ]
    # )[:, 1].astype(float)
    #
    # test_ood = np.array(
    #     [
    #         het_goldfeldquandt(
    #             y_ood_flat[:, feature_i],
    #             x_ood_flat,
    #             # np.expand_dims(x_ood_flat[:, feature_i], 1),
    #         )
    #         for feature_i in range(total_features)
    #     ]
    # )[:, 1].astype(float)
    #
    # print("HETERO: " + str(target_dim))
    # print(len(np.argwhere(test_id < 0.05)) / len(test_id) * 100)
    # print(len(np.argwhere(test_ood < 0.05)) / len(test_ood) * 100)
    # print(test_id.mean())
    # print(test_ood.mean())
# ===========================================
# plt.figure()
# plt.hist(shap_feat_id, density=True, alpha=0.7)
# plt.hist(shap_feat_ood, density=True, alpha=0.7)

# ===============================
all_mae_dfs_temp = pd.concat(all_mae_dfs)
plt.figure()
sns.boxplot(data=all_mae_dfs_temp, showfliers=True)
plt.axhline(5.0, linestyle="--", color="black")

print("Mean:")
# print(all_mae_dfs_temp.quantile(0.5))
print(all_mae_dfs_temp.mean())
# print("MAX:")
# print(all_mae_dfs_temp.quantile(0.99))
print("MAX:")
print(all_mae_dfs_temp.max())

# ======variances========
for target_dim in y_pred_dict.keys():
    x_id = y_pred_dict[target_dim]["x_id"]
    x_ood = y_pred_dict[target_dim]["x_ood"]

    plt.figure()
    # plt.boxplot([flatten_np(x_id[0]).var(0), flatten_np(x_ood[0]).var(0)])
    plt.boxplot([flatten_np(x_id[0]).var(0) - flatten_np(x_ood[0]).var(0)])
    plt.title(str(target_dim))

# =======================
#
# # =================
# z_res_id = (feature_err_id_i - feature_err_id_i.mean()) / feature_err_id_i.std()
# z_res_ood = (feature_err_ood_i - feature_err_ood_i.mean()) / feature_err_ood_i.std()
#
# # DIY
# count_data = len(feature_err_id_i)
# argsort = np.argsort(feature_err_id_i)
# ranks = np.arange(len(feature_err_id_i[argsort])) + 1
# percentiles = (ranks - 0.5) / count_data
# theoretical_z_scores = norm.ppf(percentiles)
# empirical_z_scores = z_res_id[argsort]
#
# plt.figure()
# plt.scatter(theoretical_z_scores, empirical_z_scores)
# plt.plot(theoretical_z_scores, theoretical_z_scores, "--", color="black")
#
# fig, ax = plt.subplots(1, 1)
# sm.qqplot(z_res_id, line="45", ax=ax, color="tab:blue")
# sm.qqplot(z_res_ood, line="45", ax=ax, color="tab:orange")
# plt.show()

# =====
# y_id_flat = flatten_np(y_id[random_seed_i].mean(0))
# x_id_flat = flatten_np(x_id[random_seed_i])
# y_ood_flat = flatten_np(y_ood[random_seed_i].mean(0))
# x_ood_flat = flatten_np(x_ood[random_seed_i])
# total_features = x_id_flat.shape[-1]
#
# test_id = np.array(
#     [
#         het_goldfeldquandt(
#             y_id_flat[:, feature_i],
#             np.expand_dims(x_id_flat[:, feature_i], 1),
#         )
#         for feature_i in range(total_features)
#     ]
# )[:, 1].astype(float)
#
# test_ood = np.array(
#     [
#         het_goldfeldquandt(
#             y_ood_flat[:, feature_i],
#             np.expand_dims(x_ood_flat[:, feature_i], 1),
#         )
#         for feature_i in range(total_features)
#     ]
# )[:, 1].astype(float)
# =======
