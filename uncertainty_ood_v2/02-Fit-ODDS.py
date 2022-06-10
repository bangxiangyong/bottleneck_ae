import itertools
import pickle as pickle

import numpy as np
from tqdm import tqdm

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4, run_auto_lr_range_v5
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
from strathclyde_analysis_v2.evaluate_outlier_uncertainty import evaluate_ood_unc
from util.exp_manager import ExperimentManager

# fmt: off
full_likelihood = ["mse","homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli", "beta"]
homoscedestic_mode_map = { "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every","hetero-gauss": "none", "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none","beta":"none"}
likelihood_map = { "bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian", "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian", "hetero-tgauss": "truncated_gaussian", "mse": "gaussian","beta":"beta"}
twin_output_map = { "bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True, "homo-tgauss": False,"hetero-tgauss": True, "mse": False,"beta":True}
# fmt: on

use_auto_lr = True
eval_ood_unc = False

grid = {
    "skip": [True, False],
    "latent_factor": [0.1, 0.5, 1, 10],
    "bae_type": ["ae", "ens", "mcd", "vi", "vae"],
    "full_likelihood": ["mse"],
}

# grid = {
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
# }

# grid = {
#     "skip": [False],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
# }


bae_type_classes = {
    "ens": BAE_Ensemble,
    "mcd": BAE_MCDropout,
    "sghmc": BAE_SGHMC,
    "vi": BAE_VI,
    "vae": VAE,
    "ae": BAE_Ensemble,
}

n_bae_samples_map = {
    "ens": 10,
    "mcd": 100,
    "sghmc": 50,
    "vi": 100,
    "vae": 100,
    "ae": 1,
}

exp_name = "BENCHMARK_2022_TEST"

exp_man = ExperimentManager(folder_name="experiments")

# load dataset
all_data = pickle.load(open("ad_benchmark.p", "rb"))

# Loop over all grid search combinations
for values in tqdm(itertools.product(*grid.values())):
    for data in all_data:
        # setup the grid
        exp_params = dict(zip(grid.keys(), values))
        print(exp_params)

        # unpack exp params
        skip = exp_params["skip"]
        latent_factor = exp_params["latent_factor"]
        bae_type = exp_params["bae_type"]
        full_likelihood_i = exp_params["full_likelihood"]
        twin_output = twin_output_map[full_likelihood_i]
        homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
        likelihood = likelihood_map[full_likelihood_i]
        n_bae_samples = n_bae_samples_map[bae_type]

        # continue unpacking
        random_seed = data["random_seed"]
        dataset = data["dataset"]
        exp_params.update({"random_seed": random_seed, "dataset": dataset})
        bae_set_seed(random_seed)

        # load data
        x_id_train = data["x_id_train"]
        x_id_test = data["x_id_test"]
        x_ood_test = data["x_ood_test"]

        # ===============FIT BAE===============

        use_cuda = True
        weight_decay = 0.0000000001
        anchored = True if bae_type == "ens" else False
        bias = False
        se_block = False
        norm = "layer"
        # norm = "none"
        self_att = False
        self_att_transpose_only = False
        # num_epochs = 500
        num_epochs = 300
        activation = "selu"
        # activation = "none"
        lr = 0.001
        dropout = {"dropout_rate": 0.01} if bae_type == "mcd" else {}

        input_dim = x_id_train.shape[-1]
        latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)

        chain_params = [
            {
                "base": "linear",
                "architecture": [input_dim, input_dim * 4, input_dim * 4, latent_dim],
                "activation": activation,
                "norm": norm,
                "last_norm": norm,
            }
        ]

        bae_model = bae_type_classes[bae_type](
            chain_params=chain_params,
            last_activation="sigmoid",
            last_norm=norm,
            twin_output=twin_output,
            twin_params={"activation": "selu", "norm": "none"},
            skip=skip,
            use_cuda=use_cuda,
            scaler_enabled=False,
            homoscedestic_mode=homoscedestic_mode,
            likelihood=likelihood,
            weight_decay=weight_decay,
            num_samples=n_bae_samples,
            anchored=anchored,
            learning_rate=lr,
            stochastic_seed=random_seed,
            **dropout,
        )

        x_id_train_loader = convert_dataloader(
            x_id_train, batch_size=len(x_id_train) // 3, shuffle=True, drop_last=True
        )

        if use_auto_lr:
            min_lr, max_lr, half_iter = run_auto_lr_range_v5(
                x_id_train_loader,
                bae_model,
                window_size=1,
                num_epochs=10,
                run_full=False,
                plot=False,
                verbose=False,
                save_mecha="copy",
                set_scheduler=False,
            )

        if isinstance(bae_model, BAE_SGHMC):
            bae_model.fit(
                x_id_train_loader,
                burn_epoch=int(num_epochs * 2 / 3),
                sghmc_epoch=num_epochs // 3,
                clear_sghmc_params=True,
            )
        else:
            time_method(bae_model.fit, x_id_train_loader, num_epochs=num_epochs)

        # predict and evaluate
        (e_nll_id, e_nll_ood, var_nll_id, var_nll_ood), (
            eval_auroc,
            retained_res_all,
            misclas_res_all,
        ) = evaluate_ood_unc(
            bae_model=bae_model,
            x_id_train=x_id_train,
            x_id_test=x_id_test,
            x_ood_test=x_ood_test,
            exp_name=exp_name,
            exp_params=exp_params,
            eval_ood_unc=eval_ood_unc,
            exp_man=exp_man,
            ret_flatten_nll=True,
            cdf_dists=["ecdf", "norm", "uniform", "expon"],
            norm_scalings=[True, False],
            hard_threshold=0.5,
            eval_bce_se=False,
        )
        print(eval_auroc)
