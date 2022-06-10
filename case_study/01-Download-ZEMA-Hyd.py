import os, requests, zipfile, io
import numpy as np
import pickle
import pandas as pd

from uncertainty_ood_v2.util.sensor_preproc import Resample_Sensor

data_url = "https://zenodo.org/record/1323611/files/data.zip?download=1"


def download_and_extract(url, destination, force=False):
    response = requests.get(url)
    zipDocument = zipfile.ZipFile(io.BytesIO(response.content))
    # Attempt to see if we are going to overwrite anything
    if not force:
        abort = False
        for file in zipDocument.filelist:
            if os.path.isfile(os.path.join(destination, file.filename)):
                print(
                    file.filename,
                    "already exists. If you want to overwrite the file call the method with force=True",
                )
                abort = True
        if abort:
            print("Zip file was not extracted")
            return

    zipDocument.extractall(destination)


download_and_extract(data_url, "dataset/ZEMA_Hydraulic/")

data_path = "dataset/ZEMA_Hydraulic/"

filenames_input_data_1Hz = ["ts1", "ts2", "ts3", "ts4", "vs1", "se", "ce", "cp"]
filenames_input_data_1Hz = [file.upper() + ".txt" for file in filenames_input_data_1Hz]

filenames_input_data_10Hz = ["fs1", "fs2"]
filenames_input_data_10Hz = [
    file.upper() + ".txt" for file in filenames_input_data_10Hz
]

filenames_input_data_100Hz = ["ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "eps1"]
filenames_input_data_100Hz = [
    file.upper() + ".txt" for file in filenames_input_data_100Hz
]

data_input_data_1Hz = np.zeros((2205, 60, len(filenames_input_data_1Hz)))
data_input_data_10Hz = np.zeros((2205, 600, len(filenames_input_data_10Hz)))
data_input_data_100Hz = np.zeros((2205, 6000, len(filenames_input_data_100Hz)))

for id_, file_name in enumerate(filenames_input_data_1Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_1Hz[:, :, id_] = input_data.copy()

for id_, file_name in enumerate(filenames_input_data_10Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_10Hz[:, :, id_] = input_data.copy()

for id_, file_name in enumerate(filenames_input_data_100Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_100Hz[:, :, id_] = input_data.copy()

# move axis
data_input_data_1Hz = np.moveaxis(data_input_data_1Hz, 1, 2)
data_input_data_10Hz = np.moveaxis(data_input_data_10Hz, 1, 2)
data_input_data_100Hz = np.moveaxis(data_input_data_100Hz, 1, 2)

# deal with output data now
filename_target_data = "profile.txt"
data_path = "dataset/ZEMA_Hydraulic/"
sensor_metadata = pd.read_csv("zema_hydraulic_info.csv")

targets_data = np.loadtxt(data_path + filename_target_data, delimiter="\t").astype(int)[
    :, :-1
]
target_dim_maps = {
    "cooler": [100, 20, 3],
    "valve": [100, 90, 80, 73],
    "pump": [0, 1, 2],
    "acc": [130, 115, 100, 90],
}

coded_targets_data = np.copy(targets_data)

for dim_i in range(targets_data.shape[1]):
    maps = list(target_dim_maps.values())[dim_i]
    for pos_i, map_i in enumerate(maps):
        coded_targets_data[
            np.argwhere(targets_data[:, dim_i] == map_i)[:, 0], dim_i
        ] = pos_i

all_tensor_output = np.copy(coded_targets_data)

# Apply resampling
# Build HZ1
resampled_Hz1 = []
resampled_Hz1.append(data_input_data_1Hz)
resampled_Hz1.append(
    Resample_Sensor().transform(x=data_input_data_10Hz, n=10, mode="down",downsample_type="decimate")
)
resampled_Hz1.append(
    Resample_Sensor().transform(x=data_input_data_100Hz, n=100, mode="down",downsample_type="decimate")
)

resampled_Hz1 = np.concatenate(resampled_Hz1, axis=1)

# Build HZ10
resampled_Hz10 = []
resampled_Hz10.append(
    Resample_Sensor().transform(x=data_input_data_1Hz, n=10, mode="up")
)
resampled_Hz10.append(data_input_data_10Hz)
resampled_Hz10.append(
    Resample_Sensor().transform(x=data_input_data_100Hz, n=10, mode="down",downsample_type="decimate")
)
resampled_Hz10 = np.concatenate(resampled_Hz10, axis=1)

# Build HZ100
resampled_Hz100 = []
resampled_Hz100.append(
    Resample_Sensor().transform(x=data_input_data_1Hz, n=100, mode="up")
)
resampled_Hz100.append(
    Resample_Sensor().transform(x=data_input_data_10Hz, n=10, mode="up")
)
resampled_Hz100.append(data_input_data_100Hz)
resampled_Hz100 = np.concatenate(resampled_Hz100, axis=1)

# Build binary targets
y_arg_oods = []
y_arg_oods_full = []
y_arg_ids = []
y_target = np.copy(all_tensor_output)
for target_dim in range(4):
    y_arg_ood = np.argwhere(
        (y_target[:, target_dim] > 0)
        & (y_target[:, [i for i in range(4) if i != target_dim]].sum(1) == 0)
    )[:, 0]
    y_arg_ood_full = np.argwhere((y_target[:, target_dim] > 0))[:, 0]
    y_arg_oods.append(np.copy(y_arg_ood))
    y_arg_oods_full.append(np.copy(y_arg_ood_full))
    y_arg_ids.append(np.copy(np.argwhere(y_target[:, target_dim] == 0)[:, 0]))

# ===========PICKLE DATA==================

# save raw data into dict
raw_data = {
    "Hz_1": resampled_Hz1,
    "Hz_10": resampled_Hz10,
    "Hz_100": resampled_Hz100,
    "raw_target": all_tensor_output,
    "id_target": y_arg_ids,
    "ood_target": y_arg_oods,
    "ood_target_full": y_arg_oods_full,
    "sensor_metadata": sensor_metadata,
}

# Move Axis
for id_, key in enumerate(["Hz_1", "Hz_10", "Hz_100"]):
    raw_data[key] = np.moveaxis(raw_data[key], 1, 2)

pickle_folder = "pickles"

if os.path.exists(pickle_folder) == False:
    os.mkdir(pickle_folder)

# Pickle them
pickle.dump(raw_data, open(pickle_folder + "/zema_hyd_inputs_outputs.p", "wb"))
