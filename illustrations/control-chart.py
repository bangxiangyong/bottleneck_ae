import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

np.random.seed(123)

n_samples = 20
figsize = (5,4)
data = np.random.normal(10,2,n_samples)

anomaly_index =15
data[anomaly_index] *=3

ucl = data.mean()+data.std()*3
lcl = data.mean()-data.std()*3
cl = data.mean() #central line


fig, ax = plt.subplots(1,1, figsize=figsize)

cl_pattern = "--"
cl_color = "tab:red"

ax.axhline(ucl, linestyle=cl_pattern, color=cl_color)
ax.axhline(lcl, linestyle=cl_pattern, color=cl_color)
ax.axhline(cl, linestyle=cl_pattern, color="black")

ax.plot(np.arange(len(data)),data, linestyle="-",marker=".", zorder=1)
# ax.plot(np.arange(len(data)),data, linestyle="-")

trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)

# ax.text(0,ucl, "Upper control limit", color="red", ha="right", va="center", transform=trans)
# ax.text(0,lcl, "Lower control limit", color="red", ha="right", va="center", transform=trans)
# ax.text(0,cl, "Central line", color="black", ha="right", va="center", transform=trans)

y_offset = 0.5
x_offset = -4

ax.text(x_offset,ucl+y_offset, "Upper control limit", color="red")
ax.text(x_offset,lcl+y_offset, "Lower control limit", color="red")
ax.text(x_offset,cl+y_offset, "Central line", color="black")


ax.set_yticks([])

# cross overlimit
ax.scatter(anomaly_index, data[anomaly_index], marker="x", color=cl_color, s= 50, zorder=2, linewidths=2)
ax.text(anomaly_index+0.5,data[anomaly_index]+0.1, "Anomaly", color="black")

ax.set_xticks(np.arange(len(data)))
# ax.set_xticklabels(np.arange(len(data))+1)
ax.set_xticklabels([])

ax.set_xlabel("Sequence of samples")
ax.set_ylabel("Measurement of quality (unit)")
ax.set_xlim(-5, len(data)+1)
fig.tight_layout()

fig.savefig("SPC.png",dpi=500)