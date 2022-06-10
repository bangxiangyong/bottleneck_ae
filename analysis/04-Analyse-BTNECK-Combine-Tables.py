# REQUIREMENT: Run 04-Analyse-BTNECK-V2022-V2.py first to create the csvs

import pandas as pd
import os
import numpy as np

from thesis_experiments.util_analyse import (
    concat_csv_files,
    rearrange_df,
    replace_df_label_maps,
)

folder = "tables/btneck-agg/"
csv_files = os.listdir(folder)

# res_df = pd.read_csv(folder + csv_files[0])
# res_df = res_df[(res_df["bae_type"] != "Mean") & (res_df["bae_type"] != "ATE")]

all_pds = concat_csv_files(folder, key_word="-BTNECK-RAW.csv")
all_pds = all_pds[(all_pds["bae_type"] != "Mean") & (all_pds["bae_type"] != "ATE")]


all_pds_mean = all_pds.mean(0)
all_pds_sem = all_pds.sem(0)
all_pds_ATE = all_pds_mean - all_pds_mean[0]

print(str(all_pds))
print("MEAN:" + str(all_pds_mean))
print("SEM:" + str(all_pds_sem))
print("ATE:" + str(all_pds_ATE))

round_deci = 3
csv_table = all_pds_mean.copy()
display_format = "{:.0" + str(3) + "f}"
csv_table = (
    all_pds_mean.apply(display_format.format)
    + "$\pm$"
    + all_pds_sem.apply(display_format.format)
)
all_pds_ATE = all_pds_ATE.apply(display_format.format)

# ==========RE-ARRANGE=============
# bae_type_order = ["ae", "vae", "mcd", "vi", "ens", "bae_inf"]
#
# bae_type_map = {
#     "ae": "Deterministic AE",
#     "ens": "BAE-Ensemble",
#     "mcd": "BAE-MCD",
#     "vi": "BAE-BBB",
#     "sghmc": "BAE-SGHMC",
#     "vae": "VAE",
#     "bae_inf": "BAE-" + "\scalebox{1.25}{$\infty$}",
# }
# all_pds = rearrange_df(
#     all_pds,
#     "bae_type",
#     labels=["ae", "vae", "mcd", "vi", "ens", "bae_inf"],
# )
#
# all_pds = replace_df_label_maps(all_pds, bae_type_map, col="bae_type")
#
# print(all_pds)
