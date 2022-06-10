import matplotlib.pyplot as plt
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.utils.data import get_outliers_inliers

# generate random data with two features
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from util.generate_data import generate_moons
from util.helper import (
    generate_grid2d,
)
import torch

apply_log = True
# apply_log = False

bae_set_seed(123)
# X_train, Y_train, X_test, Y_test = generate_data(n_train=200,train_only=False, n_features=2)
X_train, Y_train, X_test, Y_test = generate_moons(
    train_only=False, n_samples=200, test_size=0.5, outlier_class=1
)
# X_train[:, 0] += (
#     np.random.normal(0, 1, size=X_train[:, 0].shape) * 0.75
# )  ## add random noise
full_X = X_train.copy()
# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.05

# store outliers and inliers in different numpy arrays
x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
x_outliers_test, x_inliers_test = get_outliers_inliers(X_test, Y_test)

X_train = x_inliers_train

# ====================BAE===========================

# USE BAE
input_dim = X_train.shape[-1]

chain_params = [
    {
        "base": "linear",
        "architecture": [input_dim, 100, 10, 1],
        "activation": "selu",
        "norm": "layer",
        "bias": False,
    }
]
weight_decay = 0.000000001
bae_ensemble = BAE_Ensemble(
    chain_params=chain_params,
    # last_activation="sigmoid",
    last_activation="none",
    last_norm="none",
    # twin_output=True,
    # twin_params={"activation": "none", "norm": False},
    # skip=False,
    skip=True,
    # use_cuda=True,
    # scaler_enabled=True,
    learning_rate=0.01,
    num_samples=5,
    likelihood="gaussian",
    homoscedestic_mode="none",
    # homoscedestic_mode="every",
    # weight_decay=0.0000000001,
    use_cuda=True,
    weight_decay=weight_decay,
)


# scaler = MinMaxScaler()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train mu network
train_loader = convert_dataloader(X_train_scaled, batch_size=150, shuffle=True)
bae_ensemble.fit(train_loader, num_epochs=250)
# bae_ensemble.fit(train_loader, num_epochs=1000)


train_nll = bae_ensemble.predict(X_train_scaled, select_keys=["nll"])

# get raw train scores
raw_train_scores = train_nll["nll"].mean(0).mean(-1)
raw_train_scores = np.log(raw_train_scores) if apply_log else raw_train_scores

# visualise grid
grid_2d, grid = generate_grid2d(full_X, span=1)
anomaly_scores_grid = (
    bae_ensemble.predict(scaler.transform(grid_2d), select_keys=["nll"])["nll"]
    .mean(0)
    .mean(-1)
)
bae_scores_grid = np.log(anomaly_scores_grid) if apply_log else anomaly_scores_grid

random_anomaly_index = np.random.randint(0, len(x_outliers_test), 20)

# ====================


def min_plot_decision_boundary(
    inliers,
    anomalies,
    grid_2d,
    anomaly_scores_grid,
    ax,
    levels=35,
    anomaly_threshold=None,
):
    grid = grid_2d.T.reshape(2, 100, 100)
    reshaped_Z = anomaly_scores_grid.reshape(100, 100)

    marker_s = 20
    contour = ax.contourf(grid[0], grid[1], reshaped_Z, levels=levels, cmap="Greys")

    inlier_plot = ax.scatter(
        inliers[:, 0],
        inliers[:, 1],
        c="tab:green",
        marker="o",
        edgecolor="k",
        s=marker_s,
    )
    anomaly_plot = ax.scatter(
        anomalies[:, 0],
        anomalies[:, 1],
        c="tab:orange",
        marker="x",
        s=marker_s,
    )
    # plot decision boundary
    if anomaly_threshold is not None:
        a = ax.contour(
            grid[0],
            grid[1],
            reshaped_Z,
            levels=[anomaly_threshold],
            linewidths=1.25,
            colors="red",
            linestyles="dashed",
        )
        ax.contourf(
            grid[0],
            grid[1],
            reshaped_Z,
            levels=[anomaly_scores_grid.min(), anomaly_threshold],
            colors="tab:blue",
            alpha=0.15,
        )

    # legend
    if anomaly_threshold is not None:
        ax.legend(
            [inlier_plot, anomaly_plot, a.collections[0]],
            ["Inliers", "Anomalies", "Decision boundary"],
        )
    else:
        ax.legend([inlier_plot, anomaly_plot], ["Inliers", "Anomalies"])

    # display
    ylims = (-1.5, 2)
    xlims = (-1.5, 2)
    ax.set_ylim(*ylims)
    ax.set_xlim(*xlims)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


# ==========SHALLOW MODELS==============

clf1 = IForest()
clf1.fit(X_train)
clf1_train_scores = clf1.decision_function(X_train)
clf1_scores = clf1.decision_function(grid_2d)

clf2 = PCA()
clf2.fit(X_train)
clf2_train_scores = clf2.decision_function(X_train)
clf2_scores = clf2.decision_function(grid_2d)

# =============PLOT=============
# contour_levels = 5
contour_levels = 15

decision_perc = 90
figsize = (9.5, 3.25)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

min_plot_decision_boundary(
    x_inliers_train,
    x_outliers_test,
    grid_2d,
    clf1_scores,
    ax=ax1,
    levels=contour_levels,
    anomaly_threshold=np.percentile(clf1_train_scores, decision_perc),
)
min_plot_decision_boundary(
    x_inliers_train,
    x_outliers_test,
    grid_2d,
    np.log(clf2_scores) if apply_log else clf2_scores,
    ax=ax2,
    levels=contour_levels,
    anomaly_threshold=np.percentile(
        np.log(clf2_train_scores) if apply_log else clf2_train_scores, decision_perc
    ),
)
min_plot_decision_boundary(
    x_inliers_train,
    x_outliers_test,
    grid_2d,
    bae_scores_grid,
    ax=ax3,
    levels=contour_levels,
    anomaly_threshold=np.percentile(raw_train_scores, decision_perc),
)

# TITLES
ax1.set_title("(a) Isolation Forest")
ax2.set_title("(b) PCA")
ax3.set_title("(c) BAE")


fig.tight_layout()
fig.savefig("contour_example.png", dpi=500)
