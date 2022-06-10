# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
from scipy import stats

from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.benchmarks import Params_ODDS, Params_Images

bae_set_seed(100)

import pandas as pd
import os
import copy
import itertools
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch.cuda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4, run_auto_lr_range_v5
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method

from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from uncertainty_ood_v2.util.get_predictions import flatten_nll
from uncertainty_ood_v2.util.sensor_preproc import (
    MinMaxSensor,
    FFT_Sensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence
from util.exp_manager import ExperimentManager
import torch

# =========IMPORT Params for each dataset=========
from case_study import Params_STRATH
from case_study import Params_ZEMA

# ================================================

# GLOBAL experiment parameters
set_scheduler = False  # use cyclic learning scheduler or not
use_auto_lr = True
show_auto_lr_plot = False
check_row_exists = False  # for continuation of grid, in case interrupted
eval_ood_unc = False
eval_test = True
autolr_window_size = 5  # 5 # 1
mean_prior_loss = False
use_cuda = torch.cuda.is_available()
bias = False

# total_mini_epochs = 0  # every mini epoch will evaluate AUROC
# min_epochs = 100  # warm start

total_mini_epochs = 1  # every mini epoch will evaluate AUROC
min_epochs = 100  # minimum epochs to start evaluating mini epochs; mini epochs are counted only after min_epochs

load_custom_grid = False
# load_custom_grid = "grids/STRATH-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ZEMA-reboot-rk-grid-20220418.p"
# load_custom_grid = "grids/ODDS_exll_errgrid_20220418.p"
# load_custom_grid = "grids/images_ll_incomp_20220420.p"

# Specify selected datasets and exp names
dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "CIFAR"
# dataset = "FashionMNIST"
# dataset = "Images"

exp_names = {
    "ZEMA": "ZEMA_TEST11_",
    "STRATH": "STRATH_TEST11_",
    "ODDS": "ODDS_EXLL_ERREPAIR_",
    "CIFAR": "CIFAR_",
    "FashionMNIST": "FashionMNIST_",
    "SVHN": "SVHN_",
    "MNIST": "MNIST_",
    "Images": "Images_LL_INCOMP_",
}

# =================PREPARE DATASETS============

if dataset == "ZEMA":
    zema_data = Params_ZEMA.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "STRATH":
    strath_data = Params_STRATH.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "ODDS":
    odds_data = Params_ODDS.prepare_data(
        pickle_path=os.path.join("benchmarks", "pickles")
    )

# ============================================
grid_ZEMA = {
    "random_seed": [53],
    "apply_fft": [False],
    # fmt: off
    "ss_id": [[10,3,14]],
    # to be replaced via Best TOP-K Script
    # fmt: on
    "target_dim": [3],
    "resample_factor": ["Hz_1"],
}
grid_STRATH = {
    "random_seed": [10],
    "apply_fft": [False],
    "ss_id": [[9]],
    "target_dim": [2],
    "mode": ["forging"],
    "resample_factor": [50],
}

# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
grids_datasets = {
    "ZEMA": grid_ZEMA,
    "STRATH": Params_STRATH.grid_STRATH,
    "ODDS": Params_ODDS.grid_ODDS,
    "CIFAR": Params_Images.grid_Images,
    "FashionMNIST": Params_Images.grid_Images,
    "SVHN": Params_Images.grid_Images,
    "MNIST": Params_Images.grid_Images,
    "Images": Params_Images.grid_Images,
}  # grids for each dataset
grid = grids_datasets[dataset]  # select grid based on dataset
grid_keys = grid.keys()
grid_list = list(itertools.product(*grid.values()))

# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))
# =====================================================
from scipy.stats import truncnorm, pearsonr, shapiro, norm, anderson, levene, bartlett

exp_params = dict(zip(grid_keys, grid_list[0]))
if dataset == "ZEMA":
    x_id_train, x_id_test, x_ood_test = Params_ZEMA.get_x_splits(
        zema_data, exp_params, min_max_clip=True, train_size=0.70
    )
elif dataset == "STRATH":
    x_id_train, x_id_test, x_ood_test = Params_STRATH.get_x_splits(
        strath_data, exp_params, min_max_clip=True, train_size=0.70
    )

x_id_train = np.concatenate((x_id_train, x_id_test))
flatten_x_id = flatten_np(x_id_train)
flatten_x_ood = flatten_np(x_ood_test)

# ============TEST============
# def normality_test(x):
#     return shapiro(x)[0]
def normality_test(x):
    return np.var(x)


test_id = np.apply_along_axis(normality_test, axis=0, arr=flatten_x_id)
test_ood = np.apply_along_axis(normality_test, axis=0, arr=flatten_x_ood)
# feature_ratio = np.nanmean(test_id / test_ood)
variance_diff = test_id - test_ood

print("ID: " + str(np.nanmean(test_id) ** 0.5))
print("OOD: " + str(np.nanmean(test_ood) ** 0.5))
print("DIFF: " + str(np.nanmean(variance_diff) ** 0.5))

# =================================
flatten_x_id_T = np.moveaxis(flatten_x_id, 0, 1)
flatten_x_ood_T = np.moveaxis(flatten_x_ood, 0, 1)

flatten_x_id_T = flatten_x_id_T[np.argwhere(flatten_x_id_T.var(1) > 0)[:, 0]]
flatten_x_ood_T = flatten_x_ood_T[np.argwhere(flatten_x_ood_T.var(1) > 0)[:, 0]]

test_stats = []
for x_id_feat, x_ood_feat in zip(flatten_x_id_T, flatten_x_ood_T):
    test_stats.append(levene(x_id_feat, x_ood_feat)[1])
    # test_stats.append(bartlett(x_id_feat, x_ood_feat)[1])
test_stats = np.array(test_stats)

print("MEAN LEVENE P VALUE:" + str(np.mean(test_stats)))
print("NUM EQUAL:" + str(len(np.argwhere(test_stats > 0.05)) / len(test_stats) * 100))
