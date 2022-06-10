# Prepare datasets in numpy array format for BAE-Infinite convenience


from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.benchmarks import Params_ODDS, Params_Images

bae_set_seed(100)

import os
import itertools
import pickle as pickle
import numpy as np
from tqdm import tqdm

# =========IMPORT Params for each dataset=========
from case_study import Params_STRATH
from case_study import Params_ZEMA

# ================================================

save_pickle = True
# save_pickle = False

# Specify selected datasets and exp names
dataset = "ZEMA"
# dataset = "STRATH"
# dataset = "ODDS"
# dataset = "Images"

# =================PREPARE DATASETS============
if dataset == "ZEMA":
    zema_data = Params_ZEMA.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "STRATH":
    strath_data = Params_STRATH.prepare_data(
        pickle_path=os.path.join("case_study", "pickles")
    )
elif dataset == "ODDS":
    odds_data = Params_ODDS.prepare_data(
        pickle_path=os.path.join("benchmarks", "pickles")
    )
# ============================================
# Grids
grid_Images = {
    "random_seed": [930, 717, 10, 5477, 510],
    "id_dataset": [
        "FashionMNIST",
        "CIFAR",
    ],
}

grid_ODDS = {
    "random_seed": [510, 365, 382, 322, 988, 98, 742, 17, 595, 106],
    "dataset": [
        "cardio",
        "lympho",
        "optdigits",
        "pendigits",
        "thyroid",
        "ionosphere",
        "pima",
        "vowels",
    ],
}

grid_STRATH = {
    "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
    "apply_fft": [False],
    "ss_id": [[-1]],
    "target_dim": [2],
    "mode": ["forging"],
    "resample_factor": [50],
}

grid_ZEMA = {
    # "random_seed": [53, 802, 866, 752, 228, 655, 280, 738, 526, 578],
    "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
    "apply_fft": [False],
    "ss_id": [[-1]],
    "target_dim": [0, 1, 2, 3],
    "resample_factor": ["Hz_1"],
}
map_ZEMA_target_ssid = {0: [6, 14], 1: [10], 2: [5, 8], 3: [10, 3]}
map_STRATH_target_ssid = {2: [9]}

tasks_col_map = {
    "ZEMA": "target_dim",
    "STRATH": "target_dim",
    "ODDS": "dataset",
    "Images": "id_dataset",
}
tasks_col_name = tasks_col_map[dataset]  ## enabled only if aggregate_all_tasks is False
# ================PREPARE GRID================
# Hyperparameter grids for running experiment.
# Uncomment and run for the required results in subsequent analysis.
# For each grid completion, please move the results into respective subfolder (`sensors`,`resampling`,`latent`,`likelihood`) inside a `results` folder.
grids_datasets = {
    "ZEMA": grid_ZEMA,
    "STRATH": grid_STRATH,
    "ODDS": grid_ODDS,
    "Images": grid_Images,
}  # grids for each dataset
grid = grids_datasets[dataset]  # select grid based on dataset
grid_keys = grid.keys()
grid_list = list(itertools.product(*grid.values()))

# ==================COUNT GRID SIZE====================
print("TOTAL TASKS:")
print(len(grid_list))
# =====================================================

# Loop over all grid search combinations
data_dict = {}
for values in grid_list:

    # setup the grid
    exp_params = dict(zip(grid_keys, values))
    print(exp_params)

    # unpack exp params
    random_seed = exp_params["random_seed"]
    target_dim = exp_params[tasks_col_name]
    bae_set_seed(random_seed)

    # remap sensors to target
    if dataset == "ZEMA" or dataset == "STRATH":
        if dataset == "ZEMA":
            exp_params.update({"ss_id": map_ZEMA_target_ssid[target_dim]})
        elif dataset == "STRATH":
            exp_params.update({"ss_id": map_STRATH_target_ssid[target_dim]})

    # ==============PREPARE X_ID AND X_OOD=============
    if dataset == "ZEMA":
        x_id_train, x_id_test, x_ood_test = Params_ZEMA.get_x_splits(
            zema_data, exp_params, min_max_clip=True, train_size=0.70
        )
    elif dataset == "STRATH":
        x_id_train, x_id_test, x_ood_test = Params_STRATH.get_x_splits(
            strath_data, exp_params, min_max_clip=True, train_size=0.70
        )
    elif dataset == "ODDS":
        x_id_train, x_id_test, x_ood_test = Params_ODDS.get_x_splits(
            odds_data, exp_params
        )
    elif dataset == "Images":
        id_train_loader, id_test_loader, ood_test_loader = Params_Images.get_x_splits(
            exp_params
        )

        # iterate data
        for dt_loader, x_id_label in zip(
            [id_train_loader, id_test_loader, ood_test_loader],
            ["x_id_train", "x_id_test", "x_ood_test"],
        ):
            # iterate dataloader
            new_data = []
            for batch_idx, (data, target) in tqdm(enumerate(dt_loader)):
                new_data.append(data.cpu().detach().numpy())
            new_data = np.concatenate(new_data, axis=0)

            if x_id_label == "x_id_train":
                # randomly select training batch size 80% from data
                # to reduce memory size
                total_examples = len(new_data)
                x_id_train = np.copy(new_data)[
                    np.random.choice(
                        np.arange(total_examples), size=int(total_examples * 0.8)
                    )
                ]
            elif x_id_label == "x_id_test":
                x_id_test = np.copy(new_data)
            if x_id_label == "x_ood_test":
                x_ood_test = np.copy(new_data)

    # store into dict
    # Images have a different format of nesting
    if dataset == "Images":
        new_entry = {
            "random_seed": random_seed,
            "x_id_train": x_id_train,
        }
        if target_dim not in data_dict.keys():  # if not existed, create new
            data_dict.update(
                {
                    target_dim: {
                        "train": [new_entry.copy()],
                        "x_id_test": x_id_test.copy(),
                        "x_ood_test": x_ood_test.copy(),
                    }
                }
            )
        else:
            data_dict[target_dim]["train"].append(new_entry.copy())  # otherwise, append

    else:
        new_entry = {
            "random_seed": random_seed,
            "x_id_train": x_id_train,
            "x_id_test": x_id_test,
            "x_ood_test": x_ood_test,
        }
        if target_dim not in data_dict.keys():  # if not existed, create new
            data_dict.update({target_dim: [new_entry.copy()]})
        else:
            data_dict[target_dim].append(new_entry.copy())  # otherwise, append

for key in data_dict.keys():
    print("Target dim: " + str(key))
    print("Num repetitions:" + str(len(data_dict[key])))
    if dataset == "Images":
        print("X ID SHAPE:" + str(data_dict[key]["train"][0]["x_id_train"].shape))
    else:
        print("X ID SHAPE:" + str(data_dict[key][0]["x_id_train"].shape))

# save pickle
if save_pickle:
    pickle_name = dataset + "_np_data_20220526.p"
    pickle.dump(data_dict, open(pickle_name, "wb"))
    print("Pickling data into " + str(pickle_name))
