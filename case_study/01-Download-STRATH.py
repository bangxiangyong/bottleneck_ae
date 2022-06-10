#!/usr/bin/env python
# coding: utf-8

import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io
import requests
import zipfile

# # Load Data
#

# We download the AFRC radial forge data from the link specified.
#
# Credits to Christos Tachtatzis for the code to download & extract.

# In[2]:

afrc_data_url = "https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1"

data_path = "Data_v2"  # folder for dataset

# In[3]:


def download_and_extract(url, destination, force=True):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

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
            print("Zip file was not extracted.")
            return

    zipDocument.extractall(destination)


# In[4]:

download_and_extract(afrc_data_url, data_path)


# The data is downloaded into the folder 'Data' , now we transform the data into a list of dataframes.
#
# Each dataframe in list represents the time-series measurements of all sensors for a part.


# ## Load sensor data into dataframes

data_inputs_list = []
all_files = [
    file
    for file in os.listdir(
        os.path.join(data_path, "STRATH radial forge dataset 11Sep19")
    )
    if ("Scope" in file and "csv" in file)
]
all_files.sort()

# load each part's data as a dataframe to a list
for filename in all_files:
    if "Scope" in filename and "csv" in filename:
        file_csv = pd.read_csv(
            os.path.join(data_path, "STRATH radial forge dataset 11Sep19", filename),
            encoding="cp1252",
        )
        data_inputs_list.append(file_csv)

# data_inputs_list = []
#
# # load each part's data as a dataframe to a list
# for filename in os.listdir(
#     os.path.join(data_path, "STRATH radial forge dataset 11Sep19")
# ):
#     if "Scope" in filename and "csv" in filename:
#         file_csv = pd.read_csv(
#             os.path.join(data_path, "STRATH radial forge dataset 11Sep19", filename),
#             encoding="ISO-8859-1",
#             # encoding='cp1252'
#         )
#
#         data_inputs_list.append(file_csv)


# In[8]:


len(data_inputs_list)


### Load CMM data into dataframe
#
# 1. Read data
# 2. Subtract the CMM measurements from the "base value"
# 3. Save into a dataframe

# In[4]:


data_path = "Data_v2"  # folder for dataset
output_pd = pd.read_excel(
    os.path.join(data_path, "STRATH radial forge dataset 11Sep19", "CMMData.xlsx")
)

# extract necessary output values
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


# ## Pickle Data
#
# Pickle the input & output data for ease of future use

# In[13]:

pickle_path = "pickles"
input_file_name = "strath_inputs.p"
output_file_name = "strath_outputs.p"
output_file_name_abs_err = "strath_outputs_abs_err.p"


if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# save into pickle file
pickle.dump(data_inputs_list, open(pickle_path + "/" + input_file_name, "wb"))
pickle.dump(output_df, open(pickle_path + "/" + output_file_name, "wb"))
pickle.dump(output_df_abs_err, open(pickle_path + "/" + output_file_name_abs_err, "wb"))

print(
    "Data preparation from Zenodo completed as "
    + input_file_name
    + " and "
    + output_file_name
)

## Preproc STRATH

# In[3]:
sensor_data = data_inputs_list

# split into forging, heating, transfer phases
stitched_data = sensor_data[0:]

stitched_data = np.concatenate(stitched_data, axis=0)

column_names = sensor_data[0].columns

# segment based on digital signals of Heat and Force
digital_heat = np.diff(stitched_data[:, -1])
digital_forge = np.diff((stitched_data[:, 3] > 0).astype("int"))

print(np.argwhere(column_names == "$U_GH_HEATON_1 (U25S0)"))
print(np.argwhere(column_names == "Force [kN]"))

digital_heat_diff_index = np.argwhere(digital_heat > 0)
digital_forge_start_index = np.argwhere(digital_forge == 1)
digital_forge_end_index = np.argwhere(digital_forge == -1)

# for
heating_traces = [
    stitched_data[digital_heat_diff_index[i][0] : digital_heat_diff_index[i + 1][0]]
    for i in range(digital_heat_diff_index.shape[0])
    if i < (digital_heat_diff_index.shape[0] - 1)
]
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

# =============PICKLE THEM========
pickle_path = "pickles"

if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# truncate to shortest length
# cut to smallest trace length
min_heat_length = np.array([len(trace) for trace in heating_traces]).min()
min_forge_length = np.array([len(trace) for trace in forging_traces]).min()

x_heating = np.array(
    [heating_trace[:min_heat_length] for heating_trace in heating_traces]
)
x_forging = np.array(
    [forging_trace[:min_forge_length] for forging_trace in forging_traces]
)

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
    "heating": x_heating,
    "forging": x_forging,
    "cmm_data": np_data_outputs,
    "cmm_data_abs_err": np_data_abs_err,
    "sensor_names": sensor_names,
    "cmm_header": output_headers,
    "sensor_metadata": sensor_metadata,
}
pickle.dump(final_dict, open(pickle_path + "/" + "strath_inputs_outputs.p", "wb"))
