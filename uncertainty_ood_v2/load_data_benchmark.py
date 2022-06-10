import copy
import os

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    auc,
)
from sklearn.model_selection import train_test_split
from typing import Union

from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import (
    calc_auroc,
    calc_avgprc,
    calc_auprc,
    calc_avgprc_perf,
    evaluate_misclas_detection,
    concat_ood_score,
    evaluate_retained_unc,
    evaluate_random_retained_unc,
    retained_top_unc_indices,
    retained_random_indices,
)
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed

# from uncertainty_ood.exceed import ExCeeD
from uncertainty_ood_v2.util.get_predictions import (
    calc_e_nll,
    calc_var_nll,
    flatten_nll,
    # calc_exceed,
)
from uncertainty_ood_v2.util.prepare_anomaly_datasets import (
    list_anomaly_dataset,
    load_anomaly_dataset,
)

import matplotlib.pyplot as plt
import numpy as np
import pickle

# ===EXPERIMENT PARAMETERS===
# fixed variables
from util.uncertainty import (
    convert_cdf,
    calc_outlier_unc,
    convert_hard_pred,
    plot_unc_tptnfpfn,
    get_indices_error,
)


base_folder = "anomaly_datasets"
train_size = 0.8
# valid_size = 0.1
valid_size = 0.0

# manipulated variables
random_seed = 123
dataset_id = 5

# set random seed
bae_set_seed(123)

# ===DATASET PREPARATION===
# anomaly_datasets = list_anomaly_dataset()

n_random_seeds = 10
random_seeds = np.random.randint(0, 1000, n_random_seeds)

anomaly_datasets = [
    dt + ".mat"
    for dt in [
        "cardio",
        "lympho",
        "optdigits",
        "pendigits",
        "thyroid",
        "ionosphere",
        "pima",
        "vowels",
    ]
]


final_dt = []
for random_seed in random_seeds:
    for dataset in anomaly_datasets:

        x_id_train, x_id_test, x_ood_test = load_anomaly_dataset(
            mat_file_id=dataset,
            base_folder=base_folder,
            train_size=train_size,
            random_seed=random_seed,
        )
        print(dataset)
        print(x_id_train.shape)
        print(x_id_test.shape)
        print(x_ood_test.shape)

        final_dt.append(
            {
                "dataset": dataset,
                "random_seed": random_seed,
                "x_id_train": x_id_train,
                "x_id_test": x_id_test,
                "x_ood_test": x_ood_test,
            }
        )

# save pickle
# pickle.dump(final_dt, open("ad_benchmark.p", "wb"))














