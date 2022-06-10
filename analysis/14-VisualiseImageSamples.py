import pickle
import matplotlib.pyplot as plt
import numpy as np

image_pickle = "../np_datasets/Images_np_data_20220526.p"
image_np = pickle.load(open(image_pickle, "rb"))

dataset = "FashionMNIST"
# dataset = "CIFAR"
# show_text = True
show_text = False

# ========================================
plot_samples = 5
size_factor = 0.7
fig, axes = plt.subplots(
    2,
    plot_samples,
    gridspec_kw={"wspace": 0, "hspace": 0},
    squeeze=True,
    figsize=(
        size_factor * plot_samples,
        (size_factor - (0.15 if show_text else 0)) * 2,
    ),
)
np.random.seed(200)
for plot_sample in range(plot_samples):
    id_data = image_np[dataset]["x_id_test"]
    ood_data = image_np[dataset]["x_ood_test"]

    image_id = id_data[np.random.randint(0, len(id_data)) + plot_sample]
    image_id = np.moveaxis(image_id, 0, 2)

    image_ood = ood_data[np.random.randint(0, len(ood_data)) + plot_sample]
    image_ood = np.moveaxis(image_ood, 0, 2)

    # id
    axes[0][plot_sample].axis("off")
    axes[0][plot_sample].imshow(image_id, cmap="viridis", aspect="auto")

    # ood
    axes[1][plot_sample].axis("off")
    axes[1][plot_sample].imshow(image_ood, cmap="viridis", aspect="auto")

if show_text:
    axes[0][-1].text(image_id.shape[1] + 2, image_id.shape[1] // 2, "Inliers")
    axes[1][-1].text(image_id.shape[1] + 2, image_id.shape[1] // 2, "Anomalies")

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0, wspace=0)

figname = "plots/" + dataset + "-samples.png"
plt.savefig(figname, dpi=600)
