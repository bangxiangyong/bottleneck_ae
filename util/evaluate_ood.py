import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

from baetorch.baetorch.evaluation import concat_ood_score, calc_auroc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from util.exp_manager import ExperimentManager
from statsmodels.stats.stattools import medcouple


def plot_histogram_ood(
    nll_inliers_train, nll_inliers_test, nll_outliers_test, fig=None, ax=None, bins=20
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    train_plot = ax.hist(
        nll_inliers_train, density=True, color="tab:blue", alpha=0.85, bins=bins
    )
    test_plot = ax.hist(
        nll_inliers_test, density=True, color="tab:green", alpha=0.85, bins=bins
    )
    outliers_plot = ax.hist(
        nll_outliers_test, density=True, color="tab:orange", alpha=0.85, bins=bins
    )

    handles = [
        mpatches.Patch(facecolor="tab:blue", label="Train"),
        mpatches.Patch(facecolor="tab:green", label="Test (inlier)"),
        mpatches.Patch(facecolor="tab:orange", label="Test (OOD)"),
    ]
    ax.legend(handles=handles)


def plot_kde_ood(
    nll_inliers_train, nll_inliers_test, nll_outliers_test, fig=None, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    train_plot = sns.kdeplot(nll_inliers_train, ax=ax, color="tab:blue")
    test_plot = sns.kdeplot(nll_inliers_test, ax=ax, color="tab:green")
    outliers_plot = sns.kdeplot(nll_outliers_test, ax=ax, color="tab:orange")

    handles = [
        mpatches.Patch(facecolor="tab:blue", label="Train"),
        mpatches.Patch(facecolor="tab:green", label="Test (inlier)"),
        mpatches.Patch(facecolor="tab:orange", label="Test (OOD)"),
    ]
    ax.legend(handles=handles)


def calc_tukey_fence(x, level=1.5):
    """
    Calculates lower and upper bound of Tukey's fence.
    """
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    return (Q1 - level * IQR, Q3 + level * IQR)


def calc_adjusted_tukey(x, level=1.5):
    """
    Calculates lower and upper bound of Tukey's fence.
    """
    MC = medcouple(x)
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    if MC >= 0:
        LB = Q1 - level * np.exp(-3.5 * MC) * IQR
        UB = Q3 + level * np.exp(4 * MC) * IQR
    elif MC <= 0:
        LB = Q1 - level * np.exp(-4 * MC) * IQR
        UB = Q3 + level * np.exp(3.5 * MC) * IQR

    return (LB, UB)


def flag_tukey_fence(x, level=1.5, adjusted=False):
    """
    Puts 1 for entries above or below the upper/lower bound respectively. i.e outside of tukey fence
    and 0 for those within the bounds.
    """
    if adjusted:
        lb, ub = calc_adjusted_tukey(x, level=level)
    else:
        lb, ub = calc_tukey_fence(x, level=level)
    flags = np.argwhere((x > ub) | (x < lb))[:, 0]
    res = np.zeros_like(x)
    res[flags] = 1
    return res


def evaluate_bce_se(bae_model, x_id_test, x_ood_test, aggregate=False):
    if (
        bae_model.likelihood == "gaussian" and not bae_model.twin_output
    ) or bae_model.likelihood == "bernoulli":
        bae_id_pred = bae_model.predict(
            x_id_test, select_keys=["se", "bce"], aggregate=aggregate
        )
        bae_ood_pred = bae_model.predict(
            x_ood_test, select_keys=["se", "bce"], aggregate=aggregate
        )
        if aggregate:
            # get ood scores
            e_se_id = bae_id_pred["se_mean"].mean(-1)
            e_se_ood = bae_ood_pred["se_mean"].mean(-1)
            var_se_id = bae_id_pred["se_var"].mean(-1)
            var_se_ood = bae_ood_pred["se_var"].mean(-1)

            e_bce_id = bae_id_pred["bce_mean"].mean(-1)
            e_bce_ood = bae_ood_pred["bce_mean"].mean(-1)
            var_bce_id = bae_id_pred["bce_var"].mean(-1)
            var_bce_ood = bae_ood_pred["bce_var"].mean(-1)

        if not aggregate:
            # get ood scores
            e_se_id = flatten_nll(bae_id_pred["se"]).mean(0)
            e_se_ood = flatten_nll(bae_ood_pred["se"]).mean(0)
            var_se_id = flatten_nll(bae_id_pred["se"]).var(0)
            var_se_ood = flatten_nll(bae_ood_pred["se"]).var(0)

            e_bce_id = flatten_nll(bae_id_pred["bce"]).mean(0)
            e_bce_ood = flatten_nll(bae_ood_pred["bce"]).mean(0)
            var_bce_id = flatten_nll(bae_id_pred["bce"]).var(0)
            var_bce_ood = flatten_nll(bae_ood_pred["bce"]).var(0)

        eval_res = {
            "E_SE_AUROC": calc_auroc(e_se_id, e_se_ood),
            "V_SE_AUROC": calc_auroc(var_se_id, var_se_ood),
            "E_BCE_AUROC": calc_auroc(e_bce_id, e_bce_ood),
            "V_BCE_AUROC": calc_auroc(var_bce_id, var_bce_ood),
        }

        return eval_res
