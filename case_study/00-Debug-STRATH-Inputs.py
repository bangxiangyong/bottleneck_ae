import matplotlib.pyplot as plt
import pandas as pd
import os

csv_file_windows = "Data_v2/STRATH radial forge dataset 11Sep19/Scope0001.csv"
csv_file_unix = "Data_v2/STRATH radial forge dataset 11Sep19/Scope0001-UNIX.csv"

# csv_file_windows = "Data_v2/STRATH radial forge dataset 11Sep19/Scope0081.csv"
# csv_file_unix = "Data_v2/STRATH radial forge dataset 11Sep19/Scope0081-UNIX.csv"

pd_windows = pd.read_csv(
    csv_file_windows,
    encoding="ISO-8859-1",
)
pd_unix = pd.read_csv(
    csv_file_unix,
    encoding="ISO-8859-1",
)

plt.figure()
plt.plot(pd_windows.values[:, 3])

plt.figure()
plt.plot(pd_unix.values[:, 3])


print(pd_windows.values[:, 3])
print(pd_unix.values[:, 3])

print(pd_windows.values[:, 3].sum())
