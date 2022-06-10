from pprint import pprint

import pandas as pd

from case_study import Params_ZEMA, Params_STRATH
from thesis_experiments.benchmarks import Params_ODDS, Params_Images
from thesis_experiments.util_analyse import (
    concat_csv_files,
    save_pickle_grid,
    grid_keyval_product,
)

## =====SPECIFY LOADING PARAMS HERE========
# results_folder = "results/odds-reboot-ll-20220419"  # load results here
# incomplete_grid_filename = "grids/ODDS_incomplete_ll_20220419.p"  # save grid file
# full_grid = Params_ODDS.grid_ODDS

# results_folder = "results/images-btneck-20220420"  # load results here
# incomplete_grid_filename = "grids/images_btneck_incomp_20220420.p"  # save grid file

results_folder = "results/odds-btneck-246-20220527"  # load results from here
incomplete_grid_filename = "grids/odds-btneck.p"  # save grid file
save_grid = False
# save_grid = True
optim_sensors = True

# dataset = "STRATH"
# dataset = "ZEMA"
# dataset = "Images"
dataset = "ODDS"

# results_folder = "results/images-ll-20220420"  # load results here
# incomplete_grid_filename = "grids/images_ll_incomp_20220420.p"  # save grid file

grids = {
    "Images": Params_Images.grid_Images,
    "ZEMA": Params_ZEMA.grid_ZEMA,
    "STRATH": Params_STRATH.grid_STRATH,
    "ODDS": Params_ODDS.grid_ODDS,
}
full_grid = grids[dataset]

## ============LOAD BY CONCAT CSV RESULTS=================
current_results_df = concat_csv_files(
    results_folder=results_folder, key_word="AUROC.csv"
)

## =========HANDLE ERRONEOUS GRID PARAMS======

## select columns
drop_cols = ["current_epoch", "k_sens"]
select_cols = [col for col in current_results_df.columns[1:-4] if col not in drop_cols]

## get current and targeted full grid params
full_grid_df = pd.DataFrame(grid_keyval_product(full_grid))
current_grid_df = (
    current_results_df.loc[:, select_cols].drop_duplicates().reset_index(drop=True)
)

# apply optim sensors?
if optim_sensors and (dataset == "ZEMA" or dataset == "STRATH"):
    current_grid_df_copy = full_grid_df.copy()
    zema_map_ssid = {
        0: {"bernoulli": [3], "mse": [6, 14]},
        1: {"bernoulli": [5, 8], "mse": [10]},
        2: {"bernoulli": [5], "mse": [5, 8]},
        3: {"bernoulli": [9], "mse": [10, 3]},
    }
    strath_map_ssid = {2: {"bernoulli": [9], "mse": [9]}}
    ss_id_map = {"ZEMA": zema_map_ssid, "STRATH": strath_map_ssid}

    # replace grid entry with optimised sensor selection
    new_grid = []
    for i, entry in current_grid_df_copy.iterrows():
        # select target dim
        target_dim = entry["target_dim"]
        likelihood = entry["full_likelihood"]

        # update selected sensors
        entry["ss_id"] = ss_id_map[dataset][target_dim][likelihood]
        new_grid.append(entry)
    full_grid_df = pd.concat(new_grid, 1).T

## merge intersection to find the incomplete ones
# incomplete_df = pd.concat([full_grid_df, current_grid_df]).drop_duplicates(keep=False)
incomplete_df = pd.concat([full_grid_df, current_grid_df])
duplicated_rows = ~incomplete_df.astype(str).duplicated(keep=False)
incomplete_df = incomplete_df[duplicated_rows]

## print validation
incomplete_grid_params = incomplete_df.to_dict("records")
diff_len = len(full_grid_df) - len(current_grid_df)
if diff_len != len(incomplete_df):
    raise Exception("LENGTH OF TASKS NOT EXPECTED AS THE DIFFERENCE!")

## save pickle grid
if len(incomplete_grid_params) > 0:
    grid_keys = list(incomplete_grid_params[0].keys())
    grid_list = [list(params.values()) for params in incomplete_grid_params]
    ## print results
    pprint(grid_keys)
    pprint(grid_list)

    if save_grid:
        save_pickle_grid(grid_keys, grid_list, incomplete_grid_filename)
        print("SAVED GRID: " + incomplete_grid_filename)
else:
    print("NO MISSING TASKS FOUND. ALL TASKS HAVE BEEN COMPLETED!")


print("ORIGINAL TASKS:" + str(len(full_grid_df)))
print("CURRENT TASKS:" + str(len(current_grid_df)))

if len(incomplete_grid_params) > 0:
    print("INCOMPLETE TASKS:" + str(len(grid_list)))
