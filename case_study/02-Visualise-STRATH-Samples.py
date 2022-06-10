# PLOT SAMPLES FROM THE ZEMA OR STRATH DATSETS
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
from scipy.stats import iqr

plt.rcParams.update({"font.size": 12})

bae_set_seed(7897)

# data preproc. hyper params
# resample_factor = 10
# mode = "forging"
mode = "heating"
pickle_path = "pickles"

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

# strath_data = pickle.load(open(pickle_path + "/" + "strath_inputs_outputs.p", "rb"))
strath_data = pickle.load(
    open(pickle_path + "/" + "strath_inputs_outputs_bxy20.p", "rb")
)

heating_traces = strath_data["heating"]
forging_traces = strath_data["forging"]
column_names = strath_data["sensor_names"]
cmm_data_abs_err = strath_data["cmm_data_abs_err"]
cmm_data = strath_data["cmm_data"]
cmm_data_header = strath_data["cmm_header"]

# fmt: off
heating_sensors_all = np.array([63, 64, 70, 71, 79, 76, 77, 78]) - 1
heating_sensors_pairs = np.array([[63, 70], [64, 71]]) - 1
forging_sensors_all = np.array(
    [3, 4, 6, 7, 10, 18, 26, 34, 42, 50, 66, 12, 28, 20, 36, 44, 52, 68, 14, 22, 30, 38, 46, 54, 15, 23, 31, 39, 47, 55,
     16, 24, 32, 40, 57, 65, 72, 80, 82, 83]) - 1
forging_sensor_pairs = np.array([[10, 34], [26, 50], [12, 36], [28, 52], [22, 46], [23, 47]]) - 1
# fmt: on
# forging_sensors_all = np.array([2, 25, 11, 13, 54, 71, 82, 72])
forging_sensors_all = np.array([2, 9, 11, 13, 54, 71, 82, 3])

# Names of sensors
if mode == "forging":
    chosen_sensors = forging_sensors_all
elif mode == "heating":
    chosen_sensors = heating_sensors_all

# ======================================
# ===== PLOT SENSOR SAMPLES ============
# ======================================

sample_i = 0

# Prepare sensor units and names for labelling
sensor_units = []
sensor_names = []
for sensor_full_name in np.array(column_names)[chosen_sensors]:
    temp_ = sensor_full_name.split("[")
    if len(temp_) > 1:
        sensor_units.append(temp_[1][:-1])
    else:
        sensor_units.append("")
    sensor_names.append(temp_[0].strip().replace("_", "-").replace(" ", "-"))

# Prepare matplotlib figures plot samples
figsize = (12, 5)
fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
axes = axes.flatten()

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = [100]

for resample_i in resample_factors:
    for i, sensor_i in enumerate(chosen_sensors):

        # resample sensor
        if mode == "forging":
            trace_i = forging_traces[sample_i][:, sensor_i]
        elif mode == "heating":
            trace_i = heating_traces[sample_i][:, sensor_i]

        trace_i_resampled = Resample_Sensor().transform(
            x=trace_i, n=resample_i, mode="down"
        )

        # get appropriate x_indices
        x_indices = np.arange(0, len(trace_i), resample_i)

        # actually plot
        axes[i].plot(x_indices, trace_i_resampled)

        axes[i].set_ylabel(sensor_units[i], fontsize="small")
        axes[i].set_title(sensor_names[i], fontsize="small")
        axes[i].set_xlabel("Time " + r"($1\times{10}$ms)")
        axes[i].xaxis.set_ticks([0, 2000, 4000, 5720])
        axes[i].xaxis.set_tick_params(labelbottom=True)

fig.tight_layout()
fig.savefig("strath-sample-sensors.png", dpi=500)

# ======================================
# ===== PLOT BOX PLOT TARGET CMM========
# ======================================
cmm_data_abs_err = strath_data["cmm_data_abs_err"]
cmm_data = strath_data["cmm_data"]

plt.figure()
plt.boxplot(cmm_data_abs_err)

# =============================================================
# ===============PLOT HISTOGRAM TO SHOW EXTREME VALUES=========
# =============================================================

# Prepare matplotlib figures plot samples
figsize = (12, 5)
fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
axes = axes.flatten()

# resample_factors = [1, 10, 100, 500]
# resample_factors = [10]
resample_factors = [100]

for resample_i in resample_factors:
    for i, sensor_i in enumerate(chosen_sensors):

        # resample sensor
        if mode == "forging":
            trace_i = forging_traces[sample_i][:, sensor_i]
        elif mode == "heating":
            trace_i = heating_traces[sample_i][:, sensor_i]

        trace_i_resampled = Resample_Sensor().transform(
            x=trace_i, n=resample_i, mode="down"
        )

        # get appropriate x_indices
        x_indices = np.arange(0, len(trace_i), resample_i)

        # actually plot
        axes[i].hist(MinMaxScaler().fit_transform(trace_i.reshape(-1, 1)), density=True)

        axes[i].set_ylabel("Density", fontsize="small")
        axes[i].set_title(sensor_names[i], fontsize="small")

fig.tight_layout()
# fig.savefig("strath-sample-sensors.png", dpi=550)

# ===================================================
# =============CORRELATION ANALYSIS===================
# ===================================================
# resample sensor
if mode == "forging":
    sensor_data = forging_traces[:, :, forging_sensors_all]
elif mode == "heating":
    sensor_data = heating_traces[:, :, heating_sensors_all]

# reduce redundancy in data
sensor_data_corr = [
    pd.DataFrame(sensor_data[i]).corr().values for i in range(sensor_data.shape[0])
]
sensor_abs_corr = np.abs(np.nanmean(sensor_data_corr, 0))

figsize = (12, 9)
plt.figure(figsize=figsize)
sns.heatmap(
    sensor_abs_corr,
    annot=True,
    fmt=".2f",
    yticklabels=sensor_names,
    xticklabels=sensor_names,
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Correlation-STRATH-" + str(mode) + ".png", dpi=500)

# plt.tick_params(axis="x", which="major", pad=n)
# plt.tight_layout()

# find out correlated pairs
high_corr_pairs = []
high_corr_pairs_names = []
corr_threshold = 0.95
for i in range(len(sensor_abs_corr)):
    corr_targets = np.argwhere(sensor_abs_corr[i] >= corr_threshold).flatten()
    for corr_target in corr_targets:
        if corr_target != i:
            high_corr_pairs.append((i, corr_target))
            high_corr_pairs_names.append((sensor_names[i], sensor_names[corr_target]))

pprint(high_corr_pairs)
pprint(high_corr_pairs_names)


# ====================================
# ====================================
# ====================================

# add-bias
sensor_trace = forging_traces[:, :, forging_sensors_all[0]][0]
sensor_trace = MinMaxScaler().fit_transform(sensor_trace.reshape(-1, 1))


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


# ===========================================
# VIEW TARGET DIM 2
# TUKEY ABS

std_threshold_factor = 1.25
target_dim = 2
# dt_temp = cmm_data[:, target_dim]
dt_temp = cmm_data_abs_err[:, target_dim]

# drop a few
drop_indices = [0, 1]
dt_temp = dt_temp[[i for i in range(len(dt_temp)) if i not in drop_indices]]

Q1 = np.percentile(dt_temp, 25)
Q3 = np.percentile(dt_temp, 75)
IQR = Q3 - Q1
ucl = Q3 + IQR * std_threshold_factor
lcl = Q1 - IQR * std_threshold_factor
cl = np.median(dt_temp)  # central line
plt.figure()
plt.plot(np.arange(len(dt_temp)), dt_temp)
plt.scatter(np.arange(len(dt_temp)), dt_temp)
cl_pattern = "--"
cl_color = "tab:red"
plt.axhline(ucl, linestyle=cl_pattern, color=cl_color)
plt.axhline(lcl, linestyle=cl_pattern, color=cl_color)
plt.axhline(cl, linestyle=cl_pattern, color="black")
print("Num outliers:")
print(len(np.argwhere(dt_temp >= ucl)))

# CTRL CHART - ABS
std_threshold_factor = 2
target_dim = 2
dt_temp = cmm_data_abs_err[:, target_dim]
ucl = dt_temp.mean() + dt_temp.std() * std_threshold_factor
lcl = dt_temp.mean() - dt_temp.std() * std_threshold_factor
cl = dt_temp.mean()  # central line
plt.figure()
plt.plot(dt_temp)
cl_pattern = "--"
cl_color = "tab:red"
plt.axhline(ucl, linestyle=cl_pattern, color=cl_color)
plt.axhline(lcl, linestyle=cl_pattern, color=cl_color)
plt.axhline(cl, linestyle=cl_pattern, color="black")

# CTRL CHART - ABS
dt_temp = cmm_data[:, target_dim]
ucl = dt_temp.mean() + dt_temp.std() * std_threshold_factor
lcl = dt_temp.mean() - dt_temp.std() * std_threshold_factor
cl = dt_temp.mean()  # central line
plt.figure()
plt.plot(cmm_data[:, target_dim])
cl_pattern = "--"
cl_color = "tab:red"
plt.axhline(ucl, linestyle=cl_pattern, color=cl_color)
plt.axhline(lcl, linestyle=cl_pattern, color=cl_color)
plt.axhline(cl, linestyle=cl_pattern, color="black")

# =================CONCAT TRACES PHASES====================

# resample sensor
forging_trace_i = forging_traces[sample_i][:, 3]  # force

if "transfer" in strath_data.keys():
    transfer_traces = strath_data["transfer"]
    heating_trace_i = np.concatenate(
        (heating_traces[sample_i][:, 76], transfer_traces[sample_i][:, 76])
    )  # pyrometer
else:
    heating_trace_i = heating_traces[sample_i][:, 76]  # pyrometer

concat_trace = np.concatenate((heating_trace_i, forging_trace_i))
len_xticks = np.arange(len(concat_trace))
# figsize = (12,6.5)
figsize = (6, 3)
# plt.rcParams.update({"font.size": 20})

fig, ax = plt.subplots(1, 1, figsize=figsize)
heating_color = "tab:red"
forging_color = "tab:blue"
ax_twin = ax.twinx()
ax.plot(
    np.concatenate(
        (heating_trace_i, np.zeros_like(forging_trace_i) + np.min(heating_trace_i))
    ),
    color=heating_color,
    alpha=0.7,
)
ax_twin.plot(
    np.concatenate(
        (np.zeros_like(heating_trace_i) + np.min(forging_trace_i), forging_trace_i)
    ),
    color="tab:blue",
    alpha=0.7,
)
ax.set_ylabel("Temperature (°C)", color=heating_color)
ax_twin.set_ylabel("Force (kN)", color="tab:blue")

ax.spines["left"].set_color(heating_color)
ax.spines["right"].set_color(forging_color)
ax.yaxis.label.set_color(heating_color)
ax.tick_params(axis="y", colors=heating_color)

ax_twin.spines["left"].set_color(heating_color)
ax_twin.spines["right"].set_color(forging_color)
ax_twin.yaxis.label.set_color(forging_color)
ax_twin.tick_params(axis="y", colors=forging_color)

x_ticks = np.array([0, 5000, 10000, 15000, 20000])
ax.set_xticks(ticks=x_ticks)
ax.set_xticklabels(labels=x_ticks // 100)
ax.set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("STRATH-cycle.png", dpi=500)
# =============================================================================
# ===============VISUALISE PHASES V2===================================
# ======================================================================
plt.rcParams.update({"font.size": 15})

# resample sensor
forging_trace_i = forging_traces[sample_i][:, 3]  # force

if "transfer" in strath_data.keys():
    transfer_traces = strath_data["transfer"]
    heating_trace_i = np.concatenate(
        (heating_traces[sample_i][:, 76], transfer_traces[sample_i][:, 76])
    )  # pyrometer
else:
    heating_trace_i = heating_traces[sample_i][:, 76]  # pyrometer

concat_trace = np.concatenate((heating_trace_i, forging_trace_i))
len_xticks = np.arange(len(concat_trace))
# figsize = (12,6.5)
figsize = (6, 4.5)
# plt.rcParams.update({"font.size": 20})

fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
heating_color = "tab:red"
forging_color = "tab:blue"
ax_twin = ax.twinx()
ax.plot(
    np.concatenate(
        (heating_trace_i, np.zeros_like(forging_trace_i) + np.min(heating_trace_i))
    ),
    color=heating_color,
    alpha=0.7,
)
ax_twin.plot(
    np.concatenate(
        (np.zeros_like(heating_trace_i) + np.min(forging_trace_i), forging_trace_i)
    ),
    color="tab:blue",
    alpha=0.7,
)
ax.set_ylabel("Temperature (°C)", color=heating_color)
ax_twin.set_ylabel("Force (kN)", color="tab:blue")

ax.spines["left"].set_color(heating_color)
ax.spines["right"].set_color(forging_color)
ax.yaxis.label.set_color(heating_color)
ax.tick_params(axis="y", colors=heating_color)

ax_twin.spines["left"].set_color(heating_color)
ax_twin.spines["right"].set_color(forging_color)
ax_twin.yaxis.label.set_color(forging_color)
ax_twin.tick_params(axis="y", colors=forging_color)

x_ticks = np.array([0, 5000, 10000, 15000, 20000])
ax.set_xticks(ticks=x_ticks)
ax.set_xticklabels(labels=x_ticks // 100)
# ax.set_xlabel("Time (s)")

# AX2: A-ACTpos
forging_trace_i = forging_traces[sample_i][:, 9]  # force
ax2_twin = ax2.twinx()
ax2_twin.plot(
    np.concatenate(
        (
            heating_traces[sample_i][:, 9],
            transfer_traces[sample_i][:, 9],
            forging_trace_i,
        )
    ),
    color="tab:blue",
    alpha=0.7,
)
ax2_twin.spines["right"].set_color(forging_color)
ax2_twin.yaxis.label.set_color(forging_color)
ax2_twin.tick_params(axis="y", colors=forging_color)
ax2_twin.set_ylabel("A-ACTpos (mm)", color=forging_color)
ax2.set_xlabel("Time (s)")

ax2.set_yticks([])
ax2.set_yticklabels([])
fig.tight_layout()
fig.subplots_adjust(
    top=0.975, bottom=0.133, left=0.163, right=0.815, hspace=0.212, wspace=0.185
)
fig.savefig("plots/STRATH-cycle-v2.png", dpi=600)

# =============================================================================
