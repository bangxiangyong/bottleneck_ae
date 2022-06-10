# UTILITY TO ANALYSE OUTPUT RESULT FILES
import itertools

import numpy as np
import pandas as pd
import os
import pickle

from scipy.stats import (
    norm,
    shapiro,
    spearmanr,
    pearsonr,
    linregress,
    kstest,
    anderson,
    iqr,
)
from statsmodels.distributions import ECDF


def concat_csv_files(results_folder, key_word="AUROC.csv", drop_duplicated=True):
    # concatenate output files from HPC nodes

    all_files = os.listdir(results_folder)

    filtered_files = [file for file in all_files if key_word in file]
    new_df = pd.concat(
        [pd.read_csv(os.path.join(results_folder, file)) for file in filtered_files]
    ).reset_index(drop=True)

    if drop_duplicated:
        duplicated_rows = new_df.iloc[:, 1:].duplicated()
        # print("LEN DUPLICATED:" + str(len(duplicated_rows)))
        new_df = new_df[~duplicated_rows].reset_index(drop=True)
    return new_df


def get_topk_grid(sensor_ranks=[3, 13, 10]):
    """
    Provide sensor ranks (highest on left): [3,13,10]
    Returns top k sensor for grid evaluation: [[3], [3,13], [3,13,10]]
    """
    topk_sensor_ranks = []
    for k, sensor_k in enumerate(sensor_ranks):
        topk_sensors = sensor_ranks[: k + 1]
        topk_sensor_ranks.append(topk_sensors)
    return topk_sensor_ranks


def save_pickle_grid(grid_keys, grid_list, grid_filename="grid.p"):
    pickle_grid = {"grid_keys": grid_keys, "grid_list": grid_list}
    pickle.dump(pickle_grid, open(grid_filename, "wb"))
    print("Saving new grid list as " + grid_filename)


def add_df_cols(df, cols):
    # CREATE A COLUMN BY ADDING TWO COLUMNS
    # get model+ll column identifiers
    # add multiple columns of a given df into a single col
    new_df_col = None
    for i, fixed_param in enumerate(cols):
        if i == 0 and new_df_col is None:
            new_df_col = df.loc[:, fixed_param].astype(str).copy()
        else:
            new_df_col += df.loc[:, fixed_param].astype(str).copy()
    return new_df_col


def apply_optim_df(
    raw_df,
    fixed_params=["bae_type", "full_likelihood"],
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
    target_dim_col="target_dim",
    return_groupby=False,
):
    # SELECT OPTIMISED HYPERPARAMETERS
    # iterate through target dim
    all_targets = raw_df[target_dim_col].unique()
    all_max_params = []

    # drop columns which are not present in optim params
    optim_params_temp_ = [param for param in optim_params if param in raw_df.columns]
    for target_dim in all_targets:
        res_df_target_dim = raw_df[raw_df[target_dim_col] == target_dim].copy()
        all_params = fixed_params + optim_params_temp_

        # mean over random seeds, while fixing these params
        res_df_target_dim_groupby = (
            res_df_target_dim.groupby(all_params).mean()[perf_key].reset_index()
        )
        res_df_target_dim_groupby["fixed_params"] = add_df_cols(
            res_df_target_dim_groupby, fixed_params
        )

        # for each combination of fixed params
        for fixed_param in res_df_target_dim_groupby["fixed_params"].unique():
            temp_df = res_df_target_dim_groupby[
                res_df_target_dim_groupby["fixed_params"] == fixed_param
            ]
            argmax_params = temp_df[perf_key].argmax()

            # get the optimised params
            max_params = temp_df.iloc[argmax_params].loc[all_params]
            max_params.loc[target_dim_col] = target_dim
            all_max_params.append(max_params)

    # result of optimised params to be merged with main raw df (for filtering)
    all_max_params_ = pd.concat(
        all_max_params, axis=1
    ).T  # concat results from the loop
    optim_df = raw_df.merge(all_max_params_).copy()  # filter raw df by max params

    # return grouped by
    if return_groupby:
        groupdf = optim_df.groupby(fixed_params + [target_dim_col])
        return groupdf
    return optim_df


def apply_optim_df_v2(
    raw_df,
    fixed_params=["bae_type", "full_likelihood"],
    optim_params=["current_epoch"],
    perf_key="E_AUROC",
):
    # SELECT OPTIMISED HYPERPARAMETERS
    # iterate through target dim

    all_max_params = []

    # drop columns which are not present in optim params
    optim_params_temp_ = [param for param in optim_params if param in raw_df.columns]
    res_df_target_dim = raw_df.copy()
    all_params = fixed_params + optim_params_temp_

    # mean over random seeds, while fixing these params
    res_df_target_dim_groupby = (
        res_df_target_dim.groupby(all_params).mean()[perf_key].reset_index()
    )
    res_df_target_dim_groupby["fixed_params"] = add_df_cols(
        res_df_target_dim_groupby, fixed_params
    )

    # for each combination of fixed params
    for fixed_param in res_df_target_dim_groupby["fixed_params"].unique():
        temp_df = res_df_target_dim_groupby[
            res_df_target_dim_groupby["fixed_params"] == fixed_param
        ]
        argmax_params = temp_df[perf_key].argmax()

        # get the optimised params
        max_params = temp_df.iloc[argmax_params].loc[all_params]
        all_max_params.append(max_params)

    # result of optimised params to be merged with main raw df (for filtering)
    all_max_params_ = pd.concat(
        all_max_params, axis=1
    ).T  # concat results from the loop
    optim_df = raw_df.merge(all_max_params_).copy()  # filter raw df by max params

    return optim_df


def append_mean_ate_rows(pivot_df, label_col="bae_type", baseline_col="A"):
    """
    Creates two new rows: Mean and ATE
    """
    new_pivot_df = pivot_df.copy()

    # SUMMARY ROW: Mean and ATE of tasks
    mean_row = new_pivot_df.mean(0, numeric_only=True)
    ATE_row = mean_row - mean_row[baseline_col]
    mean_row[label_col] = "Mean"  # add column label to Mean row
    ATE_row[label_col] = "ATE"  # add column label to ATE row

    new_rows = pd.DataFrame([mean_row, ATE_row])

    new_pivot_df = pd.concat([new_pivot_df, new_rows], axis=0).reset_index(
        drop=True
    )  # append both Mean and ATE rows

    return new_pivot_df


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def grid_keyval_product(grid):
    """
    Applies product to a grid into a list of dictionary of all possible combinations
    """
    return [
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    ]


def replace_df_label_maps(df, label_map, col=None):
    """
    Replaces all labels in df with the provided mapping
    Example of a label_map: {"ae": "Det. AE"}
    Can specify specific col of df to be replaced, otherwise all cols are considered.
    """
    new_df = df.copy()
    for key, val in label_map.items():
        if col is None:
            new_df = new_df.replace(key, val)
        else:
            new_df.loc[:, col] = new_df.loc[:, col].replace(key, val)
    return new_df


def calc_qq_plot(data):

    count_data = len(data)
    argsort = np.argsort(data)
    ranks = np.arange(len(data[argsort])) + 1
    percentiles = (ranks - 0.5) / count_data
    theoretical_z_scores = norm.ppf(percentiles, loc=0, scale=1)
    empirical_z_scores = ((data - data.mean()) / data.std())[argsort]

    return theoretical_z_scores, empirical_z_scores


def get_pp_stats(feature_err_id_i, feature_err_ood_i):
    """
    1. Calculates the empirical and theoretical N(0,1) CDFs for examples of features from id and ood
    2. Gets the MAE between the empirical and theoretical CDFs
    3. Returns in nested dict {"id":{"empirical":[], "theoretical":[], "mae": float}
    """

    # standardise
    z_res_id = (feature_err_id_i - feature_err_id_i.mean()) / feature_err_id_i.std()
    z_res_ood = (feature_err_ood_i - feature_err_ood_i.mean()) / feature_err_ood_i.std()

    # Compute ECDF vs theoretical CDF
    empirical_cdf_id = ECDF(z_res_id)(z_res_id)
    theoretical_cdf_id = norm.cdf(z_res_id, loc=0, scale=1)
    empirical_cdf_ood = ECDF(z_res_ood)(z_res_ood)
    theoretical_cdf_ood = norm.cdf(z_res_ood, loc=0, scale=1)

    # Exp:Calculate QQ plot
    # DIY
    # theoretical_cdf_id, empirical_cdf_id = calc_qq_plot(feature_err_id_i)
    # theoretical_cdf_ood, empirical_cdf_ood = calc_qq_plot(feature_err_ood_i)

    # Calculate MAE
    mae_id = (np.abs(theoretical_cdf_id - empirical_cdf_id)).mean() * 100
    mae_ood = (np.abs(theoretical_cdf_ood - empirical_cdf_ood)).mean() * 100

    # mae_id = (np.abs(theoretical_cdf_id - empirical_cdf_id)).mean() * 100
    # mae_ood = (np.abs(theoretical_cdf_ood - empirical_cdf_ood)).mean() * 100

    # replace with ks test?
    # mae_id = kstest(z_res_id, "norm")[0] * 10
    # mae_ood = kstest(z_res_ood, "norm")[0] * 10

    # mae_id = anderson(feature_err_id_i, "norm")[0]
    # mae_ood = anderson(feature_err_ood_i, "norm")[0]

    # temp_anderson_id = anderson(feature_err_id_i, "norm")
    # temp_anderson_ood = anderson(feature_err_ood_i, "norm")
    # mae_id = 1 if temp_anderson_id[0] > temp_anderson_id[1][-3] else 0
    # mae_ood = 1 if temp_anderson_ood[0] > temp_anderson_ood[1][-3] else 0

    # mae_ood = temp_anderson_ood[2][
    #     np.argwhere(temp_anderson_ood[1] <= temp_anderson_ood[0])[:, 0][-1]
    # ]

    # calculate shapiro
    # shapiro_id = shapiro(feature_err_id_i)[0]
    # shapiro_ood = shapiro(feature_err_ood_i)[0]
    # shapiro_id = np.log(shapiro(feature_err_id_i)[1])
    # shapiro_ood = np.log(shapiro(feature_err_ood_i)[1])
    shapiro_id = shapiro(feature_err_id_i)[1]
    shapiro_ood = shapiro(feature_err_ood_i)[1]
    # shapiro_id = anderson(feature_err_id_i, "norm")[1]
    # shapiro_ood = anderson(feature_err_ood_i, "norm")[1]

    # calculate pcorr
    pcorr_id = spearmanr(theoretical_cdf_id, empirical_cdf_id)[0]
    pcorr_ood = spearmanr(theoretical_cdf_ood, empirical_cdf_ood)[0]

    # variance
    var_id = feature_err_id_i.std()
    var_ood = feature_err_ood_i.std()
    slope, intercept, r_value_id, p_value, std_err = linregress(
        theoretical_cdf_id, empirical_cdf_id
    )
    slope, intercept, r_value_ood, p_value, std_err = linregress(
        theoretical_cdf_ood, empirical_cdf_ood
    )

    return {
        "id": {
            "empirical": empirical_cdf_id,
            "theoretical": theoretical_cdf_id,
            "mae": mae_id,
            "shapiro": shapiro_id,
            "pcorr": pcorr_id,
            "maeXpcorr": mae_id / pcorr_id,
            "var": r_value_id,
        },
        "ood": {
            "empirical": empirical_cdf_ood,
            "theoretical": theoretical_cdf_ood,
            "mae": mae_ood,
            "shapiro": shapiro_ood,
            "pcorr": pcorr_ood,
            "maeXpcorr": mae_ood / pcorr_ood,
            "var": r_value_ood,
        },
    }


def get_random_args(arr, reduce_factor=1, replace=False):
    """
    Returns randomly picked args from an array with length divided by reduce_factor
    Can be used to reduce sample size for plotting or training purposes.
    """
    random_args = np.random.choice(
        np.arange(len(arr)),
        size=len(arr) // reduce_factor,
        replace=replace,
    )
    return random_args


def get_mae_stats(feature_err_id_i):
    """
    1. Calculates the empirical and theoretical N(0,1) CDFs for examples of features from id and ood
    2. Gets the MAE between the empirical and theoretical CDFs
    3. Returns in dict of {"tcdf":[], "ecdf":[], "mae":float}
    """

    # standardise
    z_res_id = (feature_err_id_i - feature_err_id_i.mean()) / feature_err_id_i.std()

    # Compute ECDF vs theoretical CDF
    empirical_cdf_id = ECDF(z_res_id)(z_res_id)
    theoretical_cdf_id = norm.cdf(z_res_id, loc=0, scale=1)

    # Calculate MAE
    mae_id = (np.abs(theoretical_cdf_id - empirical_cdf_id)).mean() * 100

    return {"tcdf": theoretical_cdf_id, "ecdf": empirical_cdf_id, "mae": mae_id}


def get_top_whisker(data, axis=0):
    top_fence = np.percentile(data, 75, axis=axis) + 1.5 * iqr(data, axis=axis)

    top_whisker = data[np.argwhere(data <= top_fence)[:, 0]].max(axis=axis)
    return top_whisker


def get_low_high_quantile(data, high_q=75, low_q=25, axis=0):
    top_fence = np.percentile(data, high_q, axis=axis)
    low_fence = np.percentile(data, low_q, axis=axis)

    top_data = data[np.argwhere(data <= top_fence)[:, 0]].max(axis=axis)
    low_data = data[np.argwhere(data >= low_fence)[:, 0]].min(axis=axis)

    return low_data, top_data


def rearrange_df(df, col: str, labels: list):
    """
    Rearranges a df according to a given labels of a column.
    e.g. col = "bae_type", labels = ["ae","ens"]
    Labels not stated in the column will be dropped.
    """
    rearrranged_df = []
    for label in labels:
        rearrranged_df.append(df[(df[col] == label)])
    rearrranged_df = pd.concat(rearrranged_df).reset_index(drop=True)
    return rearrranged_df


def get_mean_sem(df, conditions: dict = {}, groupby: list = [], minus_baseline=True):
    """
    Get Mean and Sem from a raw df, with options to filter by condition and specify 'groupby' list.
    """
    # groupby list
    temp_df = df.copy()

    # combine list of conditions
    # conditions are dicts of {"col": value}
    if len(conditions) > 0:
        for i, key in enumerate(conditions.keys()):
            val = conditions[key]
            if i == 0:
                and_conds = df[key] == val
            else:
                and_conds = and_conds & (df[key] == val)
        temp_df = temp_df[and_conds]

    # apply groupby and mean/sem
    groupby_df = temp_df.groupby(groupby)
    df_mean = groupby_df.mean().reset_index()
    df_sem = groupby_df.sem().reset_index()

    if minus_baseline:
        df_mean["E_AUROC"] = df_mean["E_AUROC"] - df_mean["E_AUROC"][0]

    return df_mean, df_sem
