import pickle as pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy as copy

afrc_data_url = "https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1"
data_path = "Data_v2"  # folder for dataset

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 120

pickle_path = "pickles"

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
column_names = data_inputs_list[0].columns
# ==========APPENDED SENSOR DATA=========
sensor_data = data_inputs_list
# appended_sensor_data = []
# if os.path.isfile(pickle_path + "/" + "appended_sensor_data.p") == False:
#     for sensor_number in range(sensor_data[0].values.shape[1]):
#         each_sensor_data = []
#         for part in range(len(sensor_data)):
#             part_data = sensor_data[part].values[:, sensor_number].tolist()
#             each_sensor_data = each_sensor_data + part_data
#         appended_sensor_data = appended_sensor_data + [np.array(each_sensor_data)]
#     all_data_temp = {}
#     for index, sensor_name in enumerate(sensor_data[0].columns):
#         all_data_temp.update({sensor_name: appended_sensor_data[index]})
#     appended_sensor_data = pd.DataFrame(all_data_temp)
#
#     pickle.dump(
#         appended_sensor_data, open(pickle_path + "/appended_sensor_data.p", "wb")
#     )
# else:
#     appended_sensor_data = pickle.load(
#         open(pickle_path + "/appended_sensor_data.p", "rb")
#     )

appended_sensor_data = pd.concat(data_inputs_list, axis=0).reset_index(drop=True)

# appended_sensor_data_v2 = pd.concat(data_inputs_list, axis=0).reset_index(drop=True)

# =========================================

digital_heating_sensor_name = "$U_GH_HEATON_1 (U25S0)"
digital_forging_sensor_name = "Force [kN]"

# use digital signal to trim off
# for heating stage
digital_sensor_heating = appended_sensor_data[digital_heating_sensor_name].values
trigger_val = 0.3
heating_start = (
    np.flatnonzero(
        (digital_sensor_heating[:-1] < trigger_val)
        & (digital_sensor_heating[1:] > trigger_val)
    )
    + 1
)
heating_start = heating_start[:-1]
heating_stop = (
    np.flatnonzero(
        (digital_sensor_heating[:-1] > trigger_val)
        & (digital_sensor_heating[1:] < trigger_val)
    )
    + 1
)

# use digital signal to trim off
# for forging stage
digital_sensor_forging = appended_sensor_data[digital_forging_sensor_name].values
trigger_val = 0.3
forging_start = (
    np.flatnonzero(
        (digital_sensor_forging[:-1] < trigger_val)
        & (digital_sensor_forging[1:] > trigger_val)
    )
    + 1
)
# forging_start = forging_start[:-1]
forging_stop = (
    np.flatnonzero(
        (digital_sensor_forging[:-1] > trigger_val)
        & (digital_sensor_forging[1:] < trigger_val)
    )
    + 1
)

# now get the segmentation points into a dataframe
segmentation_points = pd.DataFrame(
    {
        "heating_start": heating_start,
        "heating_stop": heating_stop,
        "forging_start": forging_start,
        "forging_stop": forging_stop,
    }
)

# there should be a heating & forging start-stop for each part
print(len(heating_start), len(heating_stop), len(forging_start), len(forging_stop))
# # ======SEGMENT TO HEAT-TRANSFER-FORGE======

data_heating_phase = []
data_transfer_phase = []
data_forging_phase = []
data_full_phase = []

for part_index, part_row in segmentation_points.iterrows():
    # slice into segments
    heat_temp_dt = copy.copy(
        appended_sensor_data.loc[part_row["heating_start"] : part_row["heating_stop"]]
    )
    transfer_temp_dt = copy.copy(
        appended_sensor_data.loc[part_row["heating_stop"] : part_row["forging_start"]]
    )
    forge_temp_dt = copy.copy(
        appended_sensor_data.loc[part_row["forging_start"] : part_row["forging_stop"]]
    )
    full_temp_dt = copy.copy(
        appended_sensor_data.loc[part_row["heating_start"] : part_row["forging_stop"]]
    )

    # drop index column
    heat_temp_dt = heat_temp_dt.reset_index(drop=True)
    transfer_temp_dt = transfer_temp_dt.reset_index(drop=True)
    forge_temp_dt = forge_temp_dt.reset_index(drop=True)
    full_temp_dt = full_temp_dt.reset_index(drop=True)

    # append into list
    data_heating_phase.append(heat_temp_dt)
    data_transfer_phase.append(transfer_temp_dt)
    data_forging_phase.append(forge_temp_dt)
    data_full_phase.append(full_temp_dt)

###===========Load CMM data into dataframe===================

# 1. Read data
# 2. Subtract the CMM measurements from the "base value"
# 3. Save into a dataframe

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

# ===============PICKLE THEM=====================

pickle_path = "pickles"

if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# truncate to shortest length
# cut to smallest trace length
min_heat_length = np.array([len(trace) for trace in data_heating_phase]).min()
min_transfer_length = np.array([len(trace) for trace in data_transfer_phase]).min()
min_forge_length = np.array([len(trace) for trace in data_forging_phase]).min()
min_full_length = np.array([len(trace) for trace in data_full_phase]).min()

x_heating = np.array(
    [heating_trace.iloc[:min_heat_length] for heating_trace in data_heating_phase]
)
x_transfer = np.array(
    [
        transfer_trace.iloc[:min_transfer_length]
        for transfer_trace in data_transfer_phase
    ]
)
x_forging = np.array(
    [forging_trace.iloc[:min_forge_length] for forging_trace in data_forging_phase]
)
x_full = np.array([full_trace.iloc[:min_full_length] for full_trace in data_full_phase])

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
    "transfer": x_transfer,
    # "full": x_full,
    "cmm_data": np_data_outputs,
    "cmm_data_abs_err": np_data_abs_err,
    "sensor_names": sensor_names,
    "cmm_header": output_headers,
    "sensor_metadata": sensor_metadata,
}

pickle.dump(final_dict, open(pickle_path + "/" + "strath_inputs_outputs_bxy20.p", "wb"))

# ==============SANITY CHECK ON THE SEQUENCE LENGTHS=====================

# len_heating = [len(trace_) for trace_ in data_heating_phase]
# len_transfer = [len(trace_) for trace_ in data_transfer_phase]
# len_forging = [len(trace_) for trace_ in data_forging_phase]
# len_full = [len(trace_) for trace_ in data_full_phase]
#
# sensor_id = 76
# column_names[sensor_id]

# Plot some samples
# plt.figure()
# plt.plot(data_heating_phase[1][column_names[sensor_id]])
# plt.plot(data_transfer_phase[6][column_names[sensor_id]])
# plt.plot(data_forging_phase[1][column_names[sensor_id]])
# plt.plot(data_full_phase[1][column_names[sensor_id]])
# plt.title(column_names[sensor_id])
