import itertools

import matplotlib.pyplot as plt
import torch
from pyod.utils.data import get_outliers_inliers, generate_data

# generate random data with two features
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4

from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from util.exp_manager import ExperimentManager
from util.generate_data import (
    generate_moons,
    generate_circles,
    generate_blobs,
    generate_aniso,
)
from util.helper import generate_grid2d, plot_decision_boundary
from sklearn.datasets import make_blobs


bae_set_seed(1)
train_samples = 100

# X_train, Y_train, X_test, Y_test = generate_circles(
#     train_only=False, n_samples=train_samples, test_size=0.5, outlier_class=1
# )
#
# # X_train, Y_train, X_test, Y_test = generate_aniso(train_only=False,
# #                                                   n_samples=500,
# #                                                   test_size=0.5,
# #                                                   outlier_class = 1
# #                                                   )
#
# X_train, Y_train, X_test, Y_test = generate_blobs(
#     train_only=False, n_samples=train_samples, test_size=0.5, outlier_class=1
# )
#
# offset = 5
# X_train = X_train + offset
# X_test = X_test + offset
# # X_train, Y_train, X_test, Y_test = generate_moons(train_only=False,
# #                                                   n_samples=500,
# #                                                   test_size=0.5,
# #                                                   outlier_class = 1
# #                                                   )
#
# # by default the outlier fraction is 0.1 in generate data function
# outlier_fraction = 0.01
#
# # store outliers and inliers in different numpy arrays
# x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
# x_outliers_test, x_inliers_test = get_outliers_inliers(X_test, Y_test)
#
# # separate the two features and use it to plot the data
# # F1 = X_train[:,[0]].reshape(-1,1)
# # F2 = X_train[:,[1]].reshape(-1,1)
#
# # create a meshgrid
# # xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
# #
# # # scatter plot
# # plt.scatter(F1,F2)
# # plt.xlabel('F1')
# # plt.ylabel('F2')

# ====================BAE===========================


local_exp_path = "experiments/"
exp_man = ExperimentManager(folder_name=local_exp_path)

# grid = {
#     "skip": [True, False],
#     "overcomplete": [True, False],
#     "bae_type": ["ae", "ens", "vae"],
#     "dataset": ["blob", "circle", "moon"],
# }

# AE
# grid = {
#     "skip": [True, False],
#     "overcomplete": [True, False],
#     "bae_type": ["ae"],
#     "dataset": ["blob", "circle", "moon"],
# }

# BAE & VAE
# grid = {
#     "skip": [True],
#     "overcomplete": [True],
#     "bae_type": ["ens", "vae"],
#     "dataset": ["blob", "circle", "moon"],
# }

grid = {
    "skip": [True],
    "overcomplete": [True],
    "bae_type": ["ens"],
    "dataset": ["moon"],
}

dataset_map = {
    "blob": generate_blobs,
    "circle": generate_circles,
    "moon": generate_moons,
}
bae_type_classes = {
    "ens": BAE_Ensemble,
    "vae": VAE,
    "ae": BAE_Ensemble,
}
n_bae_samples_map = {
    "ens": 10,
    "mcd": 100,
    "sghmc": 100,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

# Loop over all grid search combinations
for rep, values in enumerate(tqdm(itertools.product(*grid.values()))):
    bae_set_seed(1)
    # setup the grid
    exp_params = dict(zip(grid.keys(), values))

    # unpack exp params
    skip = exp_params["skip"]
    overcomplete = exp_params["overcomplete"]
    bae_type = exp_params["bae_type"]
    dataset = exp_params["dataset"]
    generate_dataset = dataset_map[dataset]

    # prepare data
    input_dim = 2
    X_train, Y_train, X_test, Y_test = generate_dataset(
        train_only=False,
        n_samples=train_samples,
        test_size=0.35,
        outlier_class=2 if dataset == "blob" else 1,
    )
    x_outliers_train, x_inliers_train = get_outliers_inliers(X_train, Y_train)
    x_outliers_test, x_inliers_test = get_outliers_inliers(X_test, Y_test)

    # prepare model
    chain_params = [
        {
            "base": "linear",
            # "architecture": [input_dim, input_dim * 2, input_dim * 4],
            "architecture": [input_dim, 50, 50, 50, 100 if overcomplete else 1],
            "activation": "selu",
            "norm": "none",
            "bias": False,
        }
    ]

    lin_autoencoder = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid",
        last_norm="none",
        # twin_output=True,
        # twin_params={"activation": "none", "norm": False},
        # skip=False,
        skip=skip,
        # use_cuda=True,
        # scaler_enabled=True,
        learning_rate=0.0001,
        num_samples=n_bae_samples_map[bae_type],
        likelihood="gaussian",
        homoscedestic_mode="none",
        # weight_decay=0.0000000001,
        use_cuda=True,
    )

    scaler = MinMaxScaler(clip=False)
    x_train_scaled = scaler.fit_transform(x_inliers_train)

    # === FIT AE ===
    # lin_autoencoder.fit(x_train_scaled, num_epochs=3500)

    # run lr_range_finder
    train_dataloader = convert_dataloader(
        x_train_scaled, batch_size=len(x_train_scaled) // 5, drop_last=False
    )

    min_lr, max_lr, half_iter = run_auto_lr_range_v4(
        train_dataloader,
        lin_autoencoder,
        window_size=3,
        num_epochs=10,
        run_full=False,
        save_mecha="copy",
    )
    lin_autoencoder.fit(train_dataloader, num_epochs=100)

    # predict model and visualise grid
    nll_pred_train = lin_autoencoder.predict(x_train_scaled, select_keys=["nll"])["nll"]
    nll_pred_train_mean = nll_pred_train.mean(-1).mean(0)
    nll_pred_train_var = nll_pred_train.mean(-1).var(0)

    # predict grid2d
    grid_2d, grid_map = generate_grid2d(x_inliers_train, span=0.25)

    nll_pred_grid = lin_autoencoder.predict(
        scaler.transform(grid_2d), select_keys=["nll"]
    )["nll"]
    nll_pred_grid_mean = nll_pred_grid.mean(-1).mean(0)
    nll_pred_grid_var = nll_pred_grid.mean(-1).var(0)

    # save outputs
    output_dict = {
        "x_inliers_train": x_inliers_train,
        "x_inliers_test": x_inliers_test,
        "x_outliers_test": x_outliers_test,
        "grid_2d": grid_2d,
        "nll_pred_grid_mean": nll_pred_grid_mean,
        "nll_pred_grid_var": nll_pred_grid_var,
        "nll_pred_train_mean": nll_pred_train_mean,
        "nll_pred_train_var": nll_pred_train_var,
    }

    # exp_man.encode_pickle(exp_params, output_dict)
    # exp_man.update_csv(
    #     exp_params, insert_pickle=True, csv_name="toy_data_bottleneck.csv"
    # )

    # convert proba
    dist = "ecdf"
    norm_scaling = True

    bae_proba_model = BAE_Outlier_Proba(
        dist_type=dist,
        norm_scaling=norm_scaling,
        fit_per_bae_sample=True,
    )
    bae_proba_model.fit(nll_pred_train)

    id_proba_train_mean, id_proba_train_unc = bae_proba_model.predict(
        nll_pred_grid, norm_scaling=norm_scaling
    )

    # # plot
    # figsize = (6,6)
    # fig, ax = plt.subplots(1,1, figsize=figsize)
    # plot_decision_boundary(
    #     x_inliers_train=x_inliers_train,
    #     x_inliers_test=x_inliers_test,
    #     # x_outliers_test=x_outliers_test,
    #     grid_2d=grid_2d,
    #     Z=np.log(nll_pred_grid_mean),
    #     fig=fig,
    #     ax=ax
    # )
    # ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # fig.tight_layout()
    # fig.savefig("uncertainty-2d-a.png",dpi=500)
    #
    # plot_decision_boundary(
    #     x_inliers_train=x_inliers_train,
    #     x_inliers_test=x_inliers_test,
    #     # x_outliers_test=x_outliers_test,
    #     grid_2d=grid_2d,
    #     Z=id_proba_train_mean,
    # )
    #
    # plot_decision_boundary(
    #     x_inliers_train=x_inliers_train,
    #     x_inliers_test=x_inliers_test,
    #     # x_outliers_test=x_outliers_test,
    #     grid_2d=grid_2d,
    #     Z=id_proba_train_unc["epi"]*4,
    #     legend=False
    # )
    # plot_decision_boundary(
    #     x_inliers_train=x_inliers_train,
    #     x_inliers_test=x_inliers_test,
    #     # x_outliers_test=x_outliers_test,
    #     grid_2d=grid_2d,
    #     Z=id_proba_train_unc["alea"]*4,
    #     legend=False
    # )
    # plot_decision_boundary(
    #     x_inliers_train=x_inliers_train,
    #     x_inliers_test=x_inliers_test,
    #     # x_outliers_test=x_outliers_test,
    #     grid_2d=grid_2d,
    #     Z=id_proba_train_unc["total"]*4,
    #     legend=False
    # )

    # plot
    colorbar = False
    figsize = (16, 4)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    plot_decision_boundary(
        x_inliers_train=x_inliers_train,
        x_inliers_test=x_inliers_test,
        # x_outliers_test=x_outliers_test,
        grid_2d=grid_2d,
        Z=np.log(nll_pred_grid_mean),
        fig=fig,
        ax=ax1,
        colorbar=colorbar,
    )
    plot_decision_boundary(
        x_inliers_train=x_inliers_train,
        x_inliers_test=x_inliers_test,
        # x_outliers_test=x_outliers_test,
        grid_2d=grid_2d,
        Z=id_proba_train_unc["epi"] * 4,
        legend=False,
        fig=fig,
        ax=ax2,
        colorbar=colorbar,
    )
    plot_decision_boundary(
        x_inliers_train=x_inliers_train,
        x_inliers_test=x_inliers_test,
        # x_outliers_test=x_outliers_test,
        grid_2d=grid_2d,
        Z=id_proba_train_unc["alea"] * 4,
        legend=False,
        fig=fig,
        ax=ax3,
        colorbar=colorbar,
    )
    plot_decision_boundary(
        x_inliers_train=x_inliers_train,
        x_inliers_test=x_inliers_test,
        # x_outliers_test=x_outliers_test,
        grid_2d=grid_2d,
        Z=id_proba_train_unc["total"] * 4,
        legend=False,
        fig=fig,
        ax=ax4,
        colorbar=colorbar,
    )
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=True,
            labelbottom=True,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax1.set_title("(a) Mean NLL")
    ax2.set_title("(b) Epistemic")
    ax3.set_title("(c) Aleatoric")
    ax4.set_title("(d) Total")

    fig.tight_layout()
    fig.savefig("uncertainty-2d.png", dpi=500)
