# BTNECK results and LL results have been saved in the same csvs
# need to separate them

import pandas as pd
import os

from thesis_experiments.util_analyse import concat_csv_files

results_folder = "results/images-ll-reboot-20220422-broken"
metric_keyword = "AVGPRC"
# metric_keyword = "AUROC"
all_df = concat_csv_files(results_folder, key_word=metric_keyword + ".csv")

# FOR LL DF
ll_df = all_df[(all_df["full_likelihood"] != "mse")]
ll_mse_df = all_df[
    (all_df["full_likelihood"] == "mse")
    & (all_df["latent_factor"] == 0.1)
    & (all_df["skip"] == False)
    & (all_df["layer_norm"] == "none")
    & (all_df["id_dataset"] != "SVHN")
]
duplicated_rows = ll_mse_df.iloc[:, 1:].duplicated()
ll_mse_df = ll_mse_df[~duplicated_rows]
ll_combined_df = pd.concat((ll_df, ll_mse_df))
ll_combined_df = ll_combined_df.reset_index(drop=True)
ll_combined_df.to_csv("Images-LL-reboot-" + metric_keyword + ".csv", index=False)

# FOR BTNECK DF
btneck_df = all_df[(all_df["full_likelihood"] == "mse")]
duplicated_rows = btneck_df.iloc[:, 1:].duplicated()
btneck_df_clean = btneck_df[~duplicated_rows]
