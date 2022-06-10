## COLUMNS IN AVGPRC HAS BEEN NAMED INCORRECTLY AS AUROC, NEED TO RENAME THEM

import pandas as pd
import os

folder = "results"

subfolders = os.listdir(folder)

remap_col = {
    "E_AUROC": "E_AVGPRC",
    "V_AUROC": "V_AVGPRC",
    "WAIC_AUROC": "WAIC_AVGPRC",
    "VX_AUROC": "VX_AVGPRC",
}

# Drop first col on index if it is not needed anymore
drop_first_col = False

for subfolder in subfolders:
    for file in os.listdir(os.path.join(folder, subfolder)):
        if "AVGPRC" in file:
            this_filepath = os.path.join(folder, subfolder, file)
            res_csv = pd.read_csv(this_filepath)

            if drop_first_col:
                res_csv = res_csv.iloc[:, 1:]

            new_cols = [
                remap_col[col] if col in remap_col.keys() else col
                for col in res_csv.columns
            ]
            new_csv = res_csv.copy()
            new_csv.columns = new_cols
            new_csv.to_csv(this_filepath, index=False)
            print("Previous col:" + str(res_csv.columns))
            print("New col:" + str(new_csv.columns))
            print("Renamed: " + this_filepath)
