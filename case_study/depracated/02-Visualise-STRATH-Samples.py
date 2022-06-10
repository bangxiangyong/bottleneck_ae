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

bae_set_seed(7897)

# data preproc. hyper params
# resample_factor = 10
# mode = "forging"
mode = "heating"
pickle_path = "pickles"

tukey_threshold = 1.5

target_dims_all = [1, 2, 7, 9, 12, 17]

strath_data = pickle.load(open(pickle_path + "/" + "strath_inputs_outputs.p", "rb"))
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
forging_sensors_all = np.array([2, 25, 11, 13, 54, 71, 82, 72])

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
for sensor_full_name in column_names[chosen_sensors]:
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
    # legend
    # fig.legend(resample_factors)
fig.tight_layout()
fig.savefig("strath-sample-sensors.png", dpi=500)

# ======================================
# ===== PLOT BOX PLOT TARGET CMM========
# ======================================

from statsmodels.graphics.tsaplots import plot_acf

cmm_data_abs_err = strath_data["cmm_data_abs_err"]
cmm_data = strath_data["cmm_data"]

plt.figure()
plt.boxplot(cmm_data)

plt.figure()
plt.boxplot(cmm_data_abs_err)

# count number of outliers
plt.figure()
plt.boxplot(MinMaxScaler().fit_transform(cmm_data))

plt.figure()
plt.boxplot(MinMaxScaler().fit_transform(cmm_data_abs_err))

# plt.figure()
# plt.plot(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, 0])
#
# plt.figure()
# plt.plot(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, 3])
#
# plt.figure()
# plot_acf(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, 3])
#
# plt.figure()
# plot_acf(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, -1])
#
# plt.figure()
# plot_acf(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, 4])
#
# plt.figure()
# plt.plot(MinMaxScaler().fit_transform(cmm_data_abs_err)[:, 4])

# ood and id
tukey_flags = np.apply_along_axis(
    flag_tukey_fence, arr=cmm_data[:, 2:14], axis=0, level=tukey_threshold
)
# tukey_flags_filtered = (
#     tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
# )
ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])


# tukey_flags[3:14]
# '38 dia @200', '42 dia @140', '42 dia @80',
#        'Base angle F', 'Base angle BR', 'Base angle BL', '162mm taper F',
#        '162mm taper BR', '162mm taper BL', '40.5mm taper F', '40.5mm taper BR',
#        '40.5mm taper BL'

# correlation in outputs
# plt.figure()
# sns.pairplot(pd.DataFrame(cmm_data))

cmm_df = pd.DataFrame(cmm_data).astype(float)
plt.figure(figsize=(7, 5))
# define the mask to set the values in the upper triangle to True

# mask
mask = np.triu(np.ones_like(cmm_df, dtype=np.bool_))
# adjust mask and df
mask = mask[1:, :-1]
corr = cmm_df.iloc[1:, :-1].copy()
# plot heatmap
sns.heatmap(
    corr,
    # mask=mask,
    annot=True,
    fmt=".2f",
    cmap="BrBG",
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.8},
    yticklabels=cmm_data_header[1:],
    xticklabels=cmm_data_header[:-1],
)

plt.figure()
sns.heatmap(
    cmm_df.corr().abs(),
    yticklabels=cmm_data_header,
    xticklabels=cmm_data_header,
    annot=True,
    fmt=".2f",
)
plt.tight_layout()
