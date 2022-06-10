import pandas as pd
import os
from thesis_experiments.util_analyse import concat_csv_files, save_pickle_grid
import pickle
from pprint import pprint

# results_folder = "results/images-ll-20220413"
#
# all_files = os.listdir(results_folder)
#
# error_files = [file for file in all_files if "ERROR.csv" in file]
# error_df = pd.concat(
#     [pd.read_csv(os.path.join(results_folder, file)) for file in error_files]
# ).reset_index()

# results_folder = "results/images-ll-20220413"
# results_folder = "results/zema-sensors-rank-20220414"
# results_folder = "results/odds-ll-20220414"
results_folder = "results/odds-ex-ll-20220416"

error_df = concat_csv_files(results_folder=results_folder, key_word="ERROR.csv")

auroc_df = concat_csv_files(results_folder=results_folder, key_word="AUROC.csv")

# V_AUROC = (
#     auroc_df.groupby(["bae_type", "full_likelihood", "id_dataset", "layer_norm"])
#     .mean()["V_AUROC"]
#     .reset_index()
# )
#
# E_AUROC = auroc_df[
#     (auroc_df["layer_norm"] == "none") & (auroc_df["id_dataset"] == "CIFAR")
# ]
# E_AUROC = (
#     E_AUROC.groupby(["bae_type", "full_likelihood", "id_dataset", "layer_norm"])
#     .mean()["E_AUROC"]
#     .reset_index()
# )

# =========HANDLE ERRONEOUS GRID PARAMS======

# get erroneous grid params
error_grid_params = error_df.iloc[:, 1:11].to_dict("records")

# save pickle grid
grid_keys = list(error_grid_params[0].keys())
grid_list = [list(params.values()) for params in error_grid_params]
error_grid_filename = "grids/ODDS_exll_errgrid_20220418.p"
save_pickle_grid(grid_keys, grid_list, error_grid_filename)

pprint(grid_list)

# def load_error_grid(filename):
#     error_grid = pickle.load(open(filename, "rb"))
#
#     for grid_ in error_grid:
#


# ===========================================
