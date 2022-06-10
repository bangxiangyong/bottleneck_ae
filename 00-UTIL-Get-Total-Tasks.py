# THIS SHOULD COMBINE EXPERIMENTAL SETUP FOR ALL DATASETS UNDER ONE PIPELINE
# 1. SPECIFY GRID
# 2. PREPARE DATASET (using grid params)
# 3. PREPARE MODEL   (using grid params)
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
set_scheduler = False
use_auto_lr = True
show_auto_lr_plot = False
check_row_exists = False  # for continuation of grid, in case interrupted
eval_ood_unc = False
eval_test = False
autolr_window_size = 5  # 5 # 1
use_cuda = torch.cuda.is_available()

# Specify selected datasets and exp names
dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "CIFAR"
# dataset = "FashionMNIST"
# dataset = "Images"

exp_names = {
    "ZEMA": "ZEMA_HYD_NEW00_",
    "STRATH": "STRATH_NEW_",
    "ODDS": "ODDS_NEW_",
    "CIFAR": "CIFAR_",
    "FashionMNIST": "FashionMNIST_",
    "SVHN": "SVHN_",
    "MNIST": "MNIST_",
    "Images": "Images_",
}


# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
grids_datasets = {
    "ZEMA": Params_ZEMA.grid_ZEMA,
    "STRATH": Params_STRATH.grid_STRATH,
    "ODDS": Params_ODDS.grid_ODDS,
    "CIFAR": Params_Images.grid_Images,
    "FashionMNIST": Params_Images.grid_Images,
    "SVHN": Params_Images.grid_Images,
    "MNIST": Params_Images.grid_Images,
    "Images": Params_Images.grid_Images,
}  # grids for each dataset
grid = grids_datasets[dataset]  # select grid based on dataset
# ==================COUNT GRID SIZE====================
grid_list = list(itertools.product(*grid.values()))
print("TOTAL TASKS:" + str(len(grid_list)))
# =====================================================
