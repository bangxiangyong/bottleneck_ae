#!/usr/bin/env python
# coding: utf-8

import pickle as pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io

import pywt
import requests
import zipfile

from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from scipy.signal import find_peaks
from tslearn.preprocessing import TimeSeriesResampler

# # Load Data
#

# We download the AFRC radial forge data from the link specified.
#
# Credits to Christos Tachtatzis for the code to download & extract.

# In[2]:

# afrc_data_url = "https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1"
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.models_v2.base_layer import flatten_np
from uncertainty_ood_v2.util.sensor_preproc import (
    crest_factor,
    MinMaxSensor,
    Resample_Sensor,
)
from util.evaluate_ood import flag_tukey_fence


n_doe = 0

data_path = (
    ("AFRC Radial Forge DOE 1 and 2 - Zenodoo Upload v1/Data")  # folder for dataset
    if n_doe > 0
    else "Data_v2"
)


# In[3]:
#
# def download_and_extract(url, destination, force=True):
#     if not os.path.exists(data_path):
#         os.mkdir(data_path)
#
#     response = requests.get(url)
#     zipDocument = zipfile.ZipFile(io.BytesIO(response.content))
#
#     # Attempt to see if we are going to overwrite anything
#     if not force:
#         abort = False
#         for file in zipDocument.filelist:
#             if os.path.isfile(os.path.join(destination, file.filename)):
#                 print(
#                     file.filename,
#                     "already exists. If you want to overwrite the file call the method with force=True",
#                 )
#                 abort = True
#         if abort:
#             print("Zip file was not extracted.")
#             return
#
#     zipDocument.extractall(destination)


# In[4]:

# download_and_extract(afrc_data_url, data_path)


# The data is downloaded into the folder 'Data' , now we transform the data into a list of dataframes.
#
# Each dataframe in list represents the time-series measurements of all sensors for a part.


# ## Load sensor data into dataframes
dt_subpath = (
    "ScopeTraces_DOE" + str(n_doe)
    if n_doe > 0
    else "STRATH radial forge dataset 11Sep19"
)
data_inputs_list = []
all_files = [
    file
    for file in os.listdir(os.path.join(data_path, dt_subpath))
    if ("Scope" in file and "csv" in file)
]
all_files.sort()

# load each part's data as a dataframe to a list
for filename in all_files:
    if "Scope" in filename and "csv" in filename:
        file_csv = pd.read_csv(
            os.path.join(data_path, dt_subpath, filename),
            encoding="cp1252",
        )
        data_inputs_list.append(file_csv)

len(data_inputs_list)

# ## Load CMM data into dataframe
#
# 1. Read data
# 2. Subtract the CMM measurements from the "base value"
# 3. Save into a dataframe

# In[4]:
path_ = (
    os.path.join(
        data_path,
        "CMM_DOE" + str(n_doe) + ".xlsx",
    )
    if n_doe > 0
    else os.path.join(data_path, "STRATH radial forge dataset 11Sep19", "CMMData.xlsx")
)
output_pd = pd.read_excel(path_)


# # extract necessary output values
if n_doe > 0:
    output_headers = output_pd.columns[1:]
    base_val = output_pd.values[0, 1:]
    output_val = output_pd.values[4:, 1:]
else:
    output_headers = output_pd.columns[4:]
    base_val = output_pd.values[0, 4:]
    output_val = output_pd.values[3:, 4:]

np_data_outputs = np.copy(output_val)
np_data_abs_err = np.copy(np_data_outputs)

# extract abs error from expected base values
for output in range(np_data_abs_err.shape[1]):
    np_data_abs_err[:, output] -= base_val[output]
np_data_abs_err = np.abs(np_data_abs_err)


# In[5]:
output_df = {}
for i, value in enumerate(output_headers):
    new_df = {value: np_data_outputs[:, i]}
    output_df.update(new_df)
output_df = pd.DataFrame(output_df)

output_df_abs_err = {}
for i, value in enumerate(output_headers):
    new_df = {value: np_data_abs_err[:, i]}
    output_df_abs_err.update(new_df)
output_df_abs_err = pd.DataFrame(output_df_abs_err)

## Preproc STRATH

# In[3]:
sensor_data = data_inputs_list

# split into forging, heating, transfer phases
stitched_data = sensor_data[0:]

stitched_data = np.concatenate(stitched_data, axis=0)

column_names = sensor_data[0].columns

# segment based on digital signals of Heat and Force
heating_ssid = np.argwhere(
    column_names
    == ("$U_GH_HEATON_2 (U26S0)" if n_doe > 0 else "$U_GH_HEATON_1 (U25S0).1")
).flatten()[0]
forging_ssid = np.argwhere(column_names == "Force [kN]").flatten()[0]
stitched_data[:, forging_ssid][-1] = 0

digital_heat = np.diff(stitched_data[:, heating_ssid])
digital_forge = np.diff((stitched_data[:, forging_ssid] > 0).astype("int"))

# print(np.argwhere(column_names == "$U_GH_HEATON_2 (U26S0)"))
# print(np.argwhere(column_names == "Force [kN]"))


if n_doe > 0:
    digital_heat_start_index = np.argwhere(digital_heat == 1)
    digital_heat_end_index = np.argwhere(digital_heat == -1)
    heating_traces = [
        stitched_data[digital_heat_start[0] : digital_heat_end[0]]
        for digital_heat_start, digital_heat_end in zip(
            digital_heat_start_index, digital_heat_end_index
        )
    ]
else:
    digital_heat_diff_index = np.argwhere(digital_heat > 0)
    # for
    heating_traces = [
        stitched_data[digital_heat_diff_index[i][0] : digital_heat_diff_index[i + 1][0]]
        for i in range(digital_heat_diff_index.shape[0])
        if i < (digital_heat_diff_index.shape[0] - 1)
    ]

digital_forge_start_index = np.argwhere(digital_forge == 1)
digital_forge_end_index = np.argwhere(digital_forge == -1)
forging_traces = [
    stitched_data[digital_forge_start[0] : digital_forge_end[0]]
    for digital_forge_start, digital_forge_end in zip(
        digital_forge_start_index, digital_forge_end_index
    )
]

# verify the number of parts segmented
if len(heating_traces) != len(sensor_data):
    print("STITCHING ERROR IN HEATING PHASE")
if len(forging_traces) != len(sensor_data):
    print("STITCHING ERROR IN FORGING PHASE")

# plt.figure()
# plt.plot(sensor_data[-1].values[:, 99])
# plt.plot(np.diff(sensor_data[-1].values[:, 99]))

# for trace in heating_traces:
#     print(len(trace))

# =============PICKLE THEM========
pickle_path = "pickles"

if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# truncate to shortest length
# cut to smallest trace length
min_heat_length = np.array([len(trace) for trace in heating_traces]).min()
min_forge_length = np.array([len(trace) for trace in forging_traces]).min()
max_heat_length = np.array([len(trace) for trace in heating_traces]).max()
max_forge_length = np.array([len(trace) for trace in forging_traces]).max()

x_heating = np.array(
    [heating_trace[:min_heat_length] for heating_trace in heating_traces]
)
x_forging = np.array(
    [forging_trace[:min_forge_length] for forging_trace in forging_traces]
)


# ==========EXPERIMENTAL: RESIZE SERIES===============
reshaped_forge_trace = []
for trace in forging_traces:
    trace_temp = np.moveaxis(trace, 0, 1)
    new_ts = [
        np.squeeze(TimeSeriesResampler(sz=max_forge_length).fit_transform(x_))
        for x_ in trace_temp
    ]
    reshaped_forge_trace.append(np.moveaxis(new_ts, 0, 1))
reshaped_forge_trace = np.array(reshaped_forge_trace)

reshaped_heat_trace = []
for trace in heating_traces:
    trace_temp = np.moveaxis(trace, 0, 1)
    new_ts = [
        np.squeeze(TimeSeriesResampler(sz=max_heat_length).fit_transform(x_))
        for x_ in trace_temp
    ]
    reshaped_heat_trace.append(np.moveaxis(new_ts, 0, 1))
reshaped_heat_trace = np.array(reshaped_heat_trace)
# ===========================================

# build sensor metadata
# Prepare sensor units and names for labelling
sensor_units = []
sensor_names = []
for sensor_full_name in column_names:
    temp_ = sensor_full_name.split("[")
    if len(temp_) > 1:
        sensor_units.append(temp_[1][:-1])
    else:
        sensor_units.append("")
    sensor_names.append(temp_[0].strip().replace("_", "-").replace(" ", "-"))
sensor_metadata = pd.DataFrame(
    {
        "sensor_id": sensor_names,
        "sensor_name": sensor_names,
        "unit": sensor_units,
        "freq": 100,
    }
)

# save into pickle file
final_dict = {
    "heating": reshaped_heat_trace,
    "forging": reshaped_forge_trace,
    "cmm_data": np_data_outputs,
    "cmm_data_abs_err": np_data_abs_err,
    "sensor_names": sensor_names,
    "cmm_header": output_headers,
    "sensor_metadata": sensor_metadata,
}
pickle.dump(
    final_dict,
    open(pickle_path + "/" + "strath_DOE" + str(n_doe) + "_UPSAMPLED.p", "wb"),
)


# =======================SEGMENT FORGE===========================
sensor_data = data_inputs_list

x_ = [ss.values for ss in sensor_data]

# plt.figure()
# for x in x_:
#     plt.plot(x[:, 33])
#
# digital_forge = np.diff((stitched_data[:, forging_ssid] > 0).astype("int"))
# print(np.argwhere(column_names == "$U_GH_HEATON_2 (U26S0)"))
# print(np.argwhere(column_names == "Force [kN]"))
# digital_heat_start_index = np.argwhere(digital_heat == 1)
# digital_heat_end_index = np.argwhere(digital_heat == -1)
# digital_forge_start_index = np.argwhere(digital_forge == 1)
# digital_forge_end_index = np.argwhere(digital_forge == -1)
#

RNompos_id = 30
ANompos_id = 33
RNompos_id_absdiff = [np.abs(np.diff(x[:, RNompos_id], axis=-1)) for x in x_]
ANompos_id_absdiff = [np.abs(np.diff(x[:, ANompos_id], axis=-1)) for x in x_]

peaksR = [find_peaks(x, height=2)[0] for x in RNompos_id_absdiff]
peaksA = [find_peaks(x, height=10)[0] for x in ANompos_id_absdiff]

all_traces = []

for sample_i in range(len(x_)):
    if n_doe > 0:
        peak_start = peaksR[sample_i][np.argwhere(peaksR[sample_i] >= 15000)[0, 0]]
    else:
        peak_start = peaksR[sample_i][0]
    peak_end = peaksA[sample_i][-1]
    segmented_x_ = x_[sample_i][peak_start:peak_end]
    all_traces.append(np.moveaxis(segmented_x_, 0, 1))

# ==========RESIZE SERIES===============
from tslearn.preprocessing import TimeSeriesResampler

all_lens = [x_.shape[-1] for x_ in all_traces]
max_len = np.max(all_lens)
# max_len = 100

reshaped_trace = []
for trace in all_traces:
    new_ts = [
        np.squeeze(TimeSeriesResampler(sz=max_len).fit_transform(x_)) for x_ in trace
    ]
    reshaped_trace.append(new_ts)
reshaped_trace = np.array(reshaped_trace)
all_traces = reshaped_trace


# ===========SANITY CHECK BY CHECKING SAMPLES=================
#  plot samples
plt.figure()
for trace in all_traces:
    plt.plot(trace[3], color="tab:blue", alpha=0.05)

# ===========FEATURE EXTRACTION==============
def get_feats(i):
    feats = [
        np.abs(i).mean(),
        i.var(),
        kurtosis(i),
        skew(i),
        i.max(),
        i.min(),
        np.sqrt(np.sum(i ** 2)) / len(i),  # energy for power
        crest_factor(i),
    ]
    return np.array(feats)


# DOE 2: DB5 LV5
def apply_dwt(trace, wt_type="haar", wt_level=3):
    res = pywt.wavedec(trace, wt_type, level=wt_level)
    # res = np.array([get_feats(i) for i in res]).flatten()
    res = np.concatenate(res)
    return res


# def apply_dwt(trace, wt_type="haar", wt_level=5):
#     res = pywt.wavedec(trace, wt_type, level=wt_level)
#     res = np.array([get_feats(i) for i in res]).flatten()
#     return res


# DOE 1: DB3 LV10
# DOE 1: FEATS ONLY
# def apply_dwt(trace, wt_type="db3", wt_level=10):
#     # res = pywt.wavedec(trace, wt_type, level=wt_level)[:-2]
#     res = pywt.wavedec(trace, wt_type, level=wt_level)
#     res = np.array([get_feats(i) for i in res]).flatten()
#     return res

resample_factor_ = 50
if resample_factor_ > 1:
    downsample_type = "decimate"
    all_traces = Resample_Sensor().transform(
        all_traces.copy(), n=resample_factor_, downsample_type=downsample_type
    )

all_feats = []
for trace in all_traces:
    # all_feats.append([get_feats(trace_) for trace_ in trace])
    all_feats.append([apply_dwt(trace_) for trace_ in trace])
all_feats = np.array(all_feats)

select_sensors = [9, 13, 25, 3]
# select_sensors = [9]
select_feats = all_feats[:, select_sensors].copy()


# split x
# split ood
# get ood and id
target_dim = [1 if n_doe > 0 else 2]
tukey_threshold = 1.0
tukey_flags = np.apply_along_axis(
    flag_tukey_fence,
    arr=np_data_outputs,
    # arr=np_data_abs_err,
    axis=0,
    level=tukey_threshold,
)
tukey_flags = tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])

train_size = 0.9
random_seed = 999
id_train_args, id_test_args = train_test_split(
    id_args, train_size=train_size, shuffle=True, random_state=random_seed
)
x_id_train = select_feats[id_train_args].copy()
x_id_test = select_feats[id_test_args].copy()
x_ood_test = select_feats[ood_args].copy()

sensor_scaler = MinMaxSensor(num_sensors=x_id_train.shape[1], axis=1, clip=True)
x_id_train = sensor_scaler.fit_transform(x_id_train)
x_id_test = sensor_scaler.transform(x_id_test)
x_ood_test = sensor_scaler.transform(x_ood_test)

# sensor_scaler = MinMaxScaler()
# x_id_train = sensor_scaler.fit_transform(flatten_np(x_id_train))
# x_id_test = sensor_scaler.transform(flatten_np(x_id_test))
# x_ood_test = sensor_scaler.transform(flatten_np(x_ood_test))
# x_id_train = np.clip(x_id_train, 0, 1)
# x_id_test = np.clip(x_id_test, 0, 1)
# x_ood_test = np.clip(x_ood_test, 0, 1)

# FIT CLF
res_clf = {}
for base_model in [ABOD, KNN, IsolationForest, OCSVM, PCA]:
    clf_pyod = base_model().fit(flatten_np(x_id_train))
    ascore_id = clf_pyod.decision_function(flatten_np(x_id_test))
    ascore_ood = clf_pyod.decision_function(flatten_np(x_ood_test))

    auroc_clf_pyod = calc_auroc(ascore_id, ascore_ood)
    res_clf.update({clf_pyod.__class__.__name__.split(".")[-1]: auroc_clf_pyod})

print("NUM OUTLIERS:" + str(len(ood_args)))
pprint("AUROC PYOD:" + str(res_clf))

# ===========================================

# plt.figure()
# plt.plot(all_traces[-1])
# plt.plot(all_traces[0])
#
# sample_i = -3
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.plot(x_[sample_i][:, 3], color="black")
# ax2.plot(RNompos_id_absdiff[sample_i], color="black")
# ax3.plot(ANompos_id_absdiff[sample_i], color="black")
#
# for peak_r in peaksR[sample_i]:
#     ax1.axvline(peak_r, linestyle="--", color="red", alpha=0.5)
#     ax2.axvline(peak_r, linestyle="--", color="red", alpha=0.5)
# for peak_a in peaksA[sample_i]:
#     ax1.axvline(peak_a, linestyle="--", color="blue", alpha=0.5)
#     ax3.axvline(peak_a, linestyle="--", color="blue", alpha=0.5)
#
# all_means = [np.mean(trace) for trace in all_traces]
#
# plt.figure()
# plt.boxplot(all_means)

# extract features from sequence?


# def get_peaks(arr_, threshold=1):
#     peaks, _ = find_peaks(arr_, height=threshold)
#     return peaks
# peaksR = np.apply_along_axis(
#     func1d=get_peaks,
#     axis=1,
#     arr=RNompos_id_absdiff,
# )
# peaksA = np.apply_along_axis(
#     func1d=get_peaks, axis=1, arr=ANompos_id_absdiff, threshold=10
# )


# =========================================
