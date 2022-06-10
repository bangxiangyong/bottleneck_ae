import numpy as np
import pandas as pd

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

# ======================
np.argwhere(coded_targets_data[:, 1] > 0)

len(np.argwhere(coded_targets_data[:, 1] > 0))


# ======Build binary targets========
y_arg_oods = []
y_arg_ids = []
y_target = np.copy(all_tensor_output)
for target_dim in range(4):
    y_arg_ood = np.argwhere(
        (y_target[:, target_dim] > 0)
        & (y_target[:, [i for i in range(4) if i != target_dim]].sum(1) == 0)
    )[:, 0]
    y_arg_oods.append(np.copy(y_arg_ood))
    y_arg_ids.append(np.copy(np.argwhere(y_target[:, target_dim] == 0)[:, 0]))

# (ISOLATE BY SUBSYSTEM)
# SUBSYSTEM UNHEALTHY WHILE OTHERS ARE HEALTHY
