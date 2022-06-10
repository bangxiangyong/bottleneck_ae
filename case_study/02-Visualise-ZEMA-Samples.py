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

# plt.rcParams.update({"font.size": 15})

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


# Prepare matplotlib figures plot samples
figsize = (12, 10)
fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)
axes = axes.flatten()

fig.delaxes(axes[14])
fig.delaxes(axes[15])

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = ["Hz_1"]

# sensor_traces = zema_data["Hz_1"]
# sensor_traces = zema_data["Hz_10"]
max_length = 60
for resample_i in resample_factors:
    sensor_traces = zema_data[resample_i]

    for i, sensor_i in enumerate(chosen_sensors_filtered):

        # resample sensor
        trace_i = sensor_traces[sample_i][:, sensor_i]

        # get appropriate x_indices
        x_indices = np.arange(0, max_length)

        # actually plot
        axes[i].plot(x_indices, trace_i)

        axes[i].set_ylabel(sensor_units[i], fontsize="small")
        axes[i].set_title(sensor_full_names[i], fontsize="small")
        # axes[i].set_xlabel("Time " + r"($1\times{10}$ms)")
        axes[i].set_xlabel("Time " + r"(s)")
        axes[i].xaxis.set_tick_params(labelbottom=True)

fig.tight_layout()

fig.savefig("zema-sample-sensors-1Hz.png", dpi=500)

# ========================================
# ===== PLOT SENSOR RESAMPLED ============
# ========================================
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

# Prepare matplotlib figures plot samples
figsize = (12, 10)
fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)
axes = axes.flatten()

fig.delaxes(axes[14])
fig.delaxes(axes[15])

resample_factors = ["Hz_100", "Hz_10", "Hz_1"]
# resample_factors = ["Hz_1", "Hz_10", "Hz_100"]
# colors = ["tab:red", "tab:blue", "tab:orange"]
colors = ["tab:blue", "tab:orange", "tab:green"]
all_lines = []
max_length = 6000
for resample_i, resample_factor in enumerate(resample_factors):
    sensor_traces = zema_data[resample_factor]

    for i, sensor_i in enumerate(chosen_sensors_filtered):

        # resample sensor
        trace_i = sensor_traces[sample_i][:, sensor_i]

        # get appropriate x_indices
        x_indices = np.arange(0, max_length, max_length // len(trace_i))

        # actually plot
        (line,) = axes[i].plot(x_indices, trace_i, alpha=0.6, color=colors[resample_i])

        if i == 3:
            all_lines.append(line)

        axes[i].set_ylabel(sensor_units[i], fontsize="small")
        axes[i].set_title(sensor_full_names[i], fontsize="small")
        axes[i].set_xlabel("Time " + r"($1\times{10}$ms)")
        axes[i].xaxis.set_tick_params(labelbottom=True)

# legend
axes[3].legend(all_lines, ["1Hz", "10Hz", "100Hz"])

fig.tight_layout()

fig.savefig("zema-resampled-sensors.png", dpi=500)

# =============================================================
# ===============PLOT HISTOGRAM TO SHOW EXTREME VALUES=========
# =============================================================

# Prepare matplotlib figures plot samples
figsize = (12, 5)
fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)
axes = axes.flatten()

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = [100]

for resample_i in resample_factors:
    for i, sensor_i in enumerate(chosen_sensors_filtered):

        # resample sensor
        trace_i = sensor_data_Hz1[sample_i][:, sensor_i]

        # get appropriate x_indices
        x_indices = np.arange(0, len(trace_i))

        # actually plot
        axes[i].hist(MinMaxScaler().fit_transform(trace_i.reshape(-1, 1)), density=True)

        axes[i].set_ylabel("Density", fontsize="small")
        axes[i].set_title(sensor_names[i], fontsize="small")

fig.tight_layout()

# fig.savefig("strath-sample-sensors.png", dpi=550)


# ====================================
# ====================================
# ====================================


def perturb_sensor(sensor_trace, mode="bias", level=0.1):
    if mode == "bias":
        add_bias = level
        perturbed_sensor = np.clip(sensor_trace + add_bias, 0, 1)
    elif mode == "noise-uniform":
        add_uniform_noise = np.random.uniform(0, level, size=sensor_trace.shape)
        perturbed_sensor = np.clip(sensor_trace + add_uniform_noise, 0, 1)
    elif mode == "stuck":
        add_stuck_fault = np.random.uniform(0.95, 1, size=sensor_trace.shape)
        perturbed_sensor = np.clip(sensor_trace + add_stuck_fault, 0, 1)

    # clip sensor trace
    perturbed_sensor = np.clip(sensor_trace + add_stuck_fault, 0, 1)
    perturbed_sensor = add_stuck_fault


# ===============================================
# =============SHOW CORRELATION PLOT=============
# ===============================================
# reduce redundancy in data
sensor_Hz1_corr = [
    pd.DataFrame(sensor_data_Hz1[i]).corr().values
    for i in range(sensor_data_Hz1.shape[0])
]
sensor_Hz1_abs_corr = np.abs(np.nanmean(sensor_Hz1_corr, 0))

figsize = (12, 9)
plt.figure(figsize=figsize)
sns.heatmap(
    sensor_Hz1_abs_corr,
    annot=True,
    fmt=".2f",
    yticklabels=sensor_names,
    xticklabels=sensor_names,
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Correlation-ZEMA.png", dpi=500)

# plt.tick_params(axis="x", which="major", pad=n)
# plt.tight_layout()

# find out correlated pairs
high_corr_pairs = []
high_corr_pairs_names = []
corr_threshold = 0.95
for i in range(len(sensor_Hz1_abs_corr)):
    corr_targets = np.argwhere(sensor_Hz1_abs_corr[i] >= corr_threshold).flatten()
    for corr_target in corr_targets:
        if corr_target != i:
            high_corr_pairs.append((i, corr_target))
            high_corr_pairs_names.append((sensor_names[i], sensor_names[corr_target]))

pprint(high_corr_pairs)
pprint(high_corr_pairs_names)

# ======================================
# ===== PLOT A FEW SAMPLES ============
# ======================================
chosen_sensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# chosen_sensors_filtered = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
chosen_sensors_filtered = np.array([1, 8, 10])

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
sensor_quantities = zema_data["sensor_metadata"]["sensor_name"]

sensor_names = zema_data["sensor_metadata"]["sensor_id"]


# Prepare matplotlib figures plot samples
# figsize = (12, 6.5)
figsize = (12, 6.5)
fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
axes = axes.flatten()

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = ["Hz_100"]

# sensor_traces = zema_data["Hz_1"]
# sensor_traces = zema_data["Hz_10"]
max_length = 6000
for resample_i in resample_factors:
    sensor_traces = zema_data[resample_i]

    for i, sensor_i in enumerate(chosen_sensors_filtered):

        # resample sensor
        trace_i = sensor_traces[sample_i][:, sensor_i]

        # get appropriate x_indices
        x_indices = np.arange(0, max_length)

        # actually plot
        axes[i].plot(x_indices, trace_i)

        axes[i].set_ylabel(
            sensor_quantities[sensor_i] + " (" + sensor_units[sensor_i] + ")",
        )
        axes[i].set_xlabel("Time " + r"(s)")
        axes[i].xaxis.set_tick_params(labelbottom=True)
        axes[i].set_xticks(
            list(x_indices[::1000]) + [max_length],
        )
        axes[i].set_xticklabels(
            np.array(list(x_indices[::1000]) + [max_length]) // 100,
        )
fig.tight_layout()
fig.savefig("zema-sample-sensors-100Hz.png", dpi=500)


# ========================================================
# ==================4X1 PLOT==============================
# ========================================================


chosen_sensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# chosen_sensors_filtered = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
chosen_sensors_filtered = np.array([1, 8, 10])

sample_i = 1500
column_names = zema_data["sensor_metadata"]["sensor_id"]

# Prepare sensor units and names for labelling
sensor_units = zema_data["sensor_metadata"]["unit"]
sensor_units = sensor_units.replace("C", "°C")
sensor_full_names = (
    zema_data["sensor_metadata"]["sensor_name"]
    + "("
    + zema_data["sensor_metadata"]["sensor_id"]
    + ")"
)
sensor_quantities = zema_data["sensor_metadata"]["sensor_name"]

sensor_names = zema_data["sensor_metadata"]["sensor_id"]

plt.rcParams.update({"font.size": 15})

# Prepare matplotlib figures plot samples
# figsize = (12, 6.5)
figsize = (10, 2.5)
fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
axes = axes.flatten()

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = ["Hz_100"]

# sensor_traces = zema_data["Hz_1"]
# sensor_traces = zema_data["Hz_10"]
max_length = 6000
for resample_i in resample_factors:
    sensor_traces = zema_data[resample_i]

    for i, sensor_i in enumerate(chosen_sensors_filtered):

        # resample sensor
        trace_i = sensor_traces[sample_i][:, sensor_i]

        # get appropriate x_indices
        x_indices = np.arange(0, max_length)

        # actually plot
        axes[i].plot(x_indices, trace_i)

        axes[i].set_ylabel(
            sensor_quantities[sensor_i] + " (" + sensor_units[sensor_i] + ")",
        )
        axes[i].set_xlabel("Time " + r"(s)")
        axes[i].xaxis.set_tick_params(labelbottom=True)
        axes[i].set_xticks(
            list(x_indices[::1000]) + [max_length],
        )
        axes[i].set_xticklabels(
            np.array(list(x_indices[::1000]) + [max_length]) // 100,
        )
fig.tight_layout()
fig.subplots_adjust(
    top=0.955, bottom=0.231, left=0.089, right=0.995, hspace=0.2, wspace=0.45
)
fig.savefig("plots/zema-sample-sensors.png", dpi=600)
