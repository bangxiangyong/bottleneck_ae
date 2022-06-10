# PLOT SAMPLES FROM THE ZEMA OR STRATH DATASETS
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from baetorch.baetorch.util.seed import bae_set_seed
from uncertainty_ood_v2.util.sensor_preproc import Resample_Sensor
from util.evaluate_ood import flag_tukey_fence
import seaborn as sns
import numpy as np
from pprint import pprint

bae_set_seed(7897)

# data preproc. hyper params
# resample_factor = 10
mode = "forging"
# mode = "heating"
pickle_path = "pickles"

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

zema_data = pickle.load(open(pickle_path + "/" + "zema_hyd_inputs_outputs.p", "rb"))

sensor_data_Hz1 = zema_data["Hz_1"]
target_dim = 0
target_labels = zema_data["id_target"]


# ======================================
# ===== PLOT SENSOR 1-HZ SAMPLES ============
# ======================================
chosen_sensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
chosen_sensors_filtered = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])

sample_i = 1500
column_names = zema_data["sensor_metadata"]["sensor_id"]

# Prepare sensor units and names for labelling
sensor_units = zema_data["sensor_metadata"]["unit"]
sensor_full_names = (
    zema_data["sensor_metadata"]["sensor_name"]
    + "("
    + zema_data["sensor_metadata"]["sensor_id"]
    + ")"
)
sensor_names = zema_data["sensor_metadata"]["sensor_id"]


# =======================================
# ===========VIEW== PERCENTAGE===========
# =======================================


def calc_perc_minmax(trace, upper_threshold=0.9, lower_threshold=0.1):
    trace_ = MinMaxScaler().fit_transform(trace.reshape(-1, 1)).flatten()
    perc_minmax = len(
        np.argwhere((trace_ >= upper_threshold) | (trace_ <= lower_threshold))
    ) / len(trace_)
    return perc_minmax


sensor_i = 0
target_dim = 1

percs = np.array([calc_perc_minmax(trace[:, sensor_i]) for trace in sensor_data_Hz1])
percs_id = percs[np.array(zema_data["id_target"][target_dim])]
percs_ood = percs[np.array(zema_data["ood_target"][target_dim])]

# =======
plt.figure()
# plt.hist(percs_id, density=True, alpha=0.9)
# plt.hist(percs_ood, density=True, alpha=0.9)

sns.kdeplot(percs_id)
sns.kdeplot(percs_ood)

# trace = sensor_data_Hz1[sample_i, :, sensor_i]
# trace = MinMaxScaler().fit_transform(trace.reshape(-1, 1)).flatten()
# upper_threshold = 0.9
# lower_threshold = 0.1
# perc_minmax = len(
#     np.argwhere((trace >= upper_threshold) | (trace <= lower_threshold))
# ) / len(trace)


plt.figure()


plt.figure()
plt.hist(trace)
