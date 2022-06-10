import pickle
import pandas as pd

dt = pickle.load(open("ad_benchmark.p", "rb"))
name_map = {""}
dt_details = []
for dt_ in dt:
    name = dt_["dataset"].split(".")[0].capitalize()
    train_n = len(dt_["x_id_train"])
    id_test_n = len(dt_["x_id_test"])
    ood_test_n = len(dt_["x_ood_test"])
    features = dt_["x_id_train"].shape[-1]
    dt_details.append({"name":name,
                       "train_length":train_n,
                       "id_test_n": id_test_n,
                       "ood_test_n": ood_test_n,
                       "features":features
                       })
dt_details = pd.DataFrame(dt_details).drop_duplicates()
dt_details.to_csv("ODDS_details.csv")

